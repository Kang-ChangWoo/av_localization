#!/usr/bin/env python3
"""Use vision to propose and acoustics to choose, and measure the ceiling.

Multiplying the two distributions lets a confident-but-wrong acoustic
likelihood destroy a correct visual answer. Re-ranking cannot: acoustics only
reorders candidates vision already proposed, so the answer never leaves vision's
top-K.

That bounds the gain exactly. The ceiling is vision's *coverage* -- how often the
ground truth is anywhere in the top-K -- which is a much weaker requirement than
vision's accuracy, the fraction of times its argmax is right. A broad, ring-
shaped posterior can have terrible accuracy and excellent coverage.

Reported per K: vision's coverage, the oracle error (pick the best candidate in
the top-K), and what acoustics actually achieves against vision alone.

    python scripts/rerank_topk.py --run-name echoloc_mono
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="echoloc_mono")
    p.add_argument("--net", default="mono", choices=["mono", "mv"])
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--acoustic-temp", type=float, default=0.5)
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100, 250, 500, 1000])
    p.add_argument("--hit-radius-m", type=float, default=1.0,
                   help="a candidate counts as covering the truth within this radius")
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import numpy as np
    import torch
    import tqdm

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.models import MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    scenes = args.scenes or sorted(os.listdir(root / "desdf"))
    Ks = np.array(args.topk)

    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    cls = MonoDepthModule if args.net == "mono" else MVDepthModule
    model = cls.load_from_checkpoint(str(ckpt)).to(device).eval()

    out_rows = {}
    for scene in scenes:
        grid_npz = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
        if not grid_npz.exists():
            continue
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        mrow, mcol = np.nonzero(mask)
        desdf_t = torch.tensor(desdf["desdf"], device=device)

        blob = np.load(grid_npz)
        cand, index = blob["rir"], blob["index"]
        win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
        cand_env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
        cand_n = cand_env / cand_env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
        # map each masked cell to its rendered candidate
        lut = -np.ones(mask.shape, dtype=int)
        lut[index[:, 0], index[:, 1]] = np.arange(len(index))
        cell_cand = lut[mrow, mcol]

        dataset_dir = str(root / args.collection)
        dataset = GridSeqDataset(dataset_dir, [scene], L=L, depth_dir=dataset_dir,
                                 depth_suffix="depth40" if args.net == "mono" else "depth160")
        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene
        picks = np.linspace(0, len(dataset) - 1, min(args.max_samples, len(dataset))).astype(int)

        gt_rank, cover, oracle, rerank, vis_err = [], [], [], [], []
        for chunk in tqdm.tqdm(picks, desc=scene):
            pose_idx = int(chunk) * (L + 1) + L
            f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
            if not f.exists():
                continue
            data = dataset[int(chunk)]
            with torch.no_grad():
                if args.net == "mono":
                    pred, _, _ = model.encoder(
                        torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
                else:
                    batch = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                             for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                    batch["ref_mask"] = batch["src_mask"] = None
                    pred = model.net(batch)["d"]
            rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                device=device, dtype=torch.float32)
            _, prob_dist, _, _ = localize(desdf_t, rays)
            v = np.array(prob_dist, dtype=np.float64)[mask]

            obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
            obs_n = obs / max(obs.sum(), 1e-20)
            a = np.zeros(len(v))
            ok = cell_cand >= 0
            a[ok] = np.exp(-np.abs(cand_n[cell_cand[ok]] - obs_n[None]).sum(axis=(1, 2))
                           / args.acoustic_temp)

            gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
            dist = np.hypot(mcol - gx, mrow - gy) * pg.grid_resolution_m
            order = np.argsort(-v)                      # visual ranking
            gt_cell = int(dist.argmin())
            gt_rank.append(int(np.where(order == gt_cell)[0][0]) + 1)
            vis_err.append(float(dist[order[0]]))

            cov_row, orc_row, rr_row = [], [], []
            for K in Ks:
                sel = order[:K]
                d_sel = dist[sel]
                cov_row.append(bool((d_sel <= args.hit_radius_m).any()))
                orc_row.append(float(d_sel.min()))
                rr_row.append(float(d_sel[int(a[sel].argmax())]))
            cover.append(cov_row); oracle.append(orc_row); rerank.append(rr_row)

        if not gt_rank:
            continue
        cover = np.array(cover); oracle = np.array(oracle); rerank = np.array(rerank)
        vis_err = np.array(vis_err); gt_rank = np.array(gt_rank)
        out_rows[scene] = dict(
            samples=int(len(gt_rank)), candidates=int(mask.sum()),
            visual_recall_1m=float((vis_err < 1).mean()),
            visual_median_m=float(np.median(vis_err)),
            gt_rank_median=float(np.median(gt_rank)),
            gt_rank_p90=float(np.percentile(gt_rank, 90)),
            topk=[dict(K=int(K),
                       coverage=float(cover[:, i].mean()),
                       oracle_recall_1m=float((oracle[:, i] < 1).mean()),
                       rerank_recall_1m=float((rerank[:, i] < 1).mean()),
                       rerank_median_m=float(np.median(rerank[:, i])))
                  for i, K in enumerate(Ks)],
        )
        r = out_rows[scene]
        print(f"\n[{scene}] {r['samples']} samples, {r['candidates']} candidates | "
              f"visual recall@1m {100*r['visual_recall_1m']:.0f}%, median {r['visual_median_m']:.2f} m | "
              f"GT rank median {r['gt_rank_median']:.0f}, p90 {r['gt_rank_p90']:.0f}")
        print(f"    {'K':>5s} {'coverage':>9s} {'oracle@1m':>10s} {'rerank@1m':>10s} {'rerank med':>11s}")
        for t in r["topk"]:
            print(f"    {t['K']:5d} {100*t['coverage']:8.0f}% {100*t['oracle_recall_1m']:9.0f}% "
                  f"{100*t['rerank_recall_1m']:9.0f}% {t['rerank_median_m']:10.2f} m")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / f"rerank_{args.run_name}.json"
    out.write_text(json.dumps({"condition": args.condition, "scenes": out_rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
