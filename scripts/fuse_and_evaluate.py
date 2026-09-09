#!/usr/bin/env python3
"""Fuse the visual posterior with the acoustic likelihood, with calibration.

The two modalities produce scores on wildly different scales -- the visual
matcher uses ``exp(-L1/40)``, the acoustic one ``exp(-L1/0.5)`` -- so a plain
product is not a fusion at all: the sharper distribution simply wins. This is
the failure ``docs/data_contracts.md`` warns about, where an arbitrary numerical
scale becomes an accidental modality weight.

Fusion is therefore done in log space on the shared valid support,

    log p_fused  =  (1 - w) * log p_visual  +  w * log p_acoustic

and ``w`` is swept. Reporting the sweep makes the trade-off visible; the
headline number uses leave-one-scene-out calibration, so the weight is never
chosen on the scene it is scored on.

    python scripts/fuse_and_evaluate.py --run-name echoloc_mono
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
    p.add_argument("--scenes", nargs="+", default=None, help="default: all DESDF scenes")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--acoustic-temp", type=float, default=0.5)
    p.add_argument("--weights", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--max-samples", type=int, default=200)
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
    weights = np.array(args.weights)

    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    cls = MonoDepthModule if args.net == "mono" else MVDepthModule
    model = cls.load_from_checkpoint(str(ckpt)).to(device).eval()

    per_scene: dict[str, np.ndarray] = {}      # scene -> (n_samples, n_weights) errors
    for scene in scenes:
        grid_npz = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
        if not grid_npz.exists():
            print(f"[fuse] {scene}: no candidate grid, skipping")
            continue

        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        desdf_t = torch.tensor(desdf["desdf"], device=device)

        blob = np.load(grid_npz)
        cand, index = blob["rir"], blob["index"]
        win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
        cand_env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
        cand_n = cand_env / cand_env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
        rr, cc = index[:, 0], index[:, 1]

        dataset_dir = str(root / args.collection)
        dataset = GridSeqDataset(dataset_dir, [scene], L=L, depth_dir=dataset_dir,
                                 depth_suffix="depth40" if args.net == "mono" else "depth160")
        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene

        picks = np.linspace(0, len(dataset) - 1, min(args.max_samples, len(dataset))).astype(int)
        errs = []
        for chunk in tqdm.tqdm(picks, desc=f"{scene}"):
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

            # both modalities as log-probabilities on the shared valid support
            v = np.array(prob_dist, dtype=np.float64)[mask]
            v = np.log(np.clip(v / max(v.sum(), 1e-300), 1e-300, None))

            obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
            obs_n = obs / max(obs.sum(), 1e-20)
            score = np.exp(-np.abs(cand_n - obs_n[None]).sum(axis=(1, 2)) / args.acoustic_temp)
            a_full = np.zeros(mask.shape)
            a_full[rr, cc] = score
            a = a_full[mask]
            a = np.log(np.clip(a / max(a.sum(), 1e-300), 1e-300, None))

            gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
            mrow, mcol = np.nonzero(mask)
            row = []
            for w in weights:
                fused = (1 - w) * v + w * a
                k = int(fused.argmax())
                row.append(float(np.hypot(mcol[k] - gx, mrow[k] - gy) * pg.grid_resolution_m))
            errs.append(row)
        if errs:
            per_scene[scene] = np.array(errs)
            med = np.median(per_scene[scene], axis=0)
            print(f"[fuse] {scene}: {len(errs)} samples | median error by w: " +
                  " ".join(f"{w:.1f}={m:.2f}" for w, m in zip(weights, med)))

    if not per_scene:
        print("no scenes scored")
        return 1

    # ---- leave-one-scene-out calibration -----------------------------------
    print("\n[fuse] leave-one-scene-out calibration")
    rows = []
    for held in per_scene:
        others = np.concatenate([per_scene[s] for s in per_scene if s != held], axis=0)
        w_star = weights[int(np.argmin(np.median(others, axis=0)))]
        e = per_scene[held][:, int(np.where(weights == w_star)[0][0])]
        e_vis = per_scene[held][:, 0]
        e_ac = per_scene[held][:, -1]
        rows.append(dict(scene=held, w_star=float(w_star), samples=int(len(e)),
                         visual_median=float(np.median(e_vis)), acoustic_median=float(np.median(e_ac)),
                         fused_median=float(np.median(e)),
                         visual_recall_1m=float((e_vis < 1).mean()),
                         acoustic_recall_1m=float((e_ac < 1).mean()),
                         fused_recall_1m=float((e < 1).mean()),
                         improved_fraction=float((e < e_vis).mean())))
        r = rows[-1]
        print(f"   {held:16s} w*={w_star:.1f}  median  visual {r['visual_median']:.2f} | "
              f"acoustic {r['acoustic_median']:.2f} | fused {r['fused_median']:.2f} m   "
              f"recall@1m  {100*r['visual_recall_1m']:.0f}% -> {100*r['fused_recall_1m']:.0f}%")

    allerr = np.concatenate(list(per_scene.values()), axis=0)
    payload = {
        "run": args.run_name, "condition": args.condition, "window_ms": args.window_ms,
        "weights": weights.tolist(),
        "median_by_weight": np.median(allerr, axis=0).tolist(),
        "recall1m_by_weight": (allerr < 1).mean(axis=0).tolist(),
        "leave_one_scene_out": rows,
    }
    out = args.out or REPO_ROOT / "outputs" / "metrics" / f"fusion_calibrated_{args.run_name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[fuse] pooled median by weight: " +
          " ".join(f"{w:.1f}={m:.2f}" for w, m in zip(weights, np.median(allerr, axis=0))))
    print(f"[fuse] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
