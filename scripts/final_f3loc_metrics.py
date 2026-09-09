#!/usr/bin/env python3
"""Final table: every visual model, with and without the acoustic refinement.

Uses F3Loc's own metric definitions -- recall at 0.1 m, 0.5 m and 1 m, plus the
combined 1 m / 30 deg -- so the rows can be read next to the published table.

The refinement keeps vision in charge. Vision scores the full ``(H, W, 36)``
pose volume, its top-K poses become the shortlist, and the acoustic likelihood
only reorders that shortlist. Orientation therefore comes from vision, which
resolves it to 10 deg bins; the acoustic grid is rendered at one heading and
constrains position only, which is what it was measured to be good for.

    python scripts/final_f3loc_metrics.py --topk 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# The published observation-module numbers, for context only: they are measured
# on Gibson, not on this Replica-derived dataset.
PAPER = {
    "F3Loc paper, Ours_s (gibson_f)": (0.047, 0.286, 0.366, 0.351),
    "F3Loc paper, Ours_m (gibson_f)": (0.132, 0.409, 0.452, 0.437),
    "F3Loc paper, Ours_f (gibson_g)": (0.122, 0.394, 0.445, 0.432),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--models", nargs="+",
                   default=["echoloc_mono_fg:mono", "echoloc_mv_fg:mv", "echoloc_comp_fg:comp"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--n-poses", type=int, default=300)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def featurize(env: np.ndarray) -> np.ndarray:
    per = env / env.sum(axis=-1, keepdims=True).clip(1e-12)
    total = env.sum(axis=(-1, -2))
    split = env.sum(axis=-1) / np.clip(total[..., None], 1e-12, None)
    return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import tqdm

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.models import CompDepthModule, MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    scenes = args.scenes or sorted(os.listdir(root / "desdf"))
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
    CLS = {"mono": MonoDepthModule, "mv": MVDepthModule, "comp": CompDepthModule}

    # per-scene geometry and acoustic candidates, loaded once
    geo = {}
    for scene in scenes:
        g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
        if not g.exists():
            continue
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        blob = np.load(g)
        cand, index = blob["rir"], blob["index"]
        env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
        env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
        acoustic = np.zeros((pg.height, pg.width, env.shape[1], env.shape[2]))
        acoustic[index[:, 0], index[:, 1]] = env
        geo[scene] = dict(pg=pg, mask=mask,
                          desdf_t=torch.tensor(desdf["desdf"], device=device),
                          acoustic=featurize(acoustic),
                          poses=np.array([[float(v) for v in l.split()]
                                          for l in open(root / args.collection / scene / "poses.txt")
                                          if l.strip()]))
    print(f"[final] scenes: {list(geo)}   shortlist K={args.topk}")

    def metrics(pos_err, yaw_err):
        e, y = np.array(pos_err), np.array(yaw_err)
        return dict(n=int(len(e)),
                    r01=float((e < 0.1).mean()), r05=float((e < 0.5).mean()),
                    r1=float((e < 1.0).mean()),
                    r1_30=float(((e < 1.0) & (y < 30)).mean()),
                    median_m=float(np.median(e)))

    table = {}
    for spec in args.models:
        run, net = spec.split(":")
        ckpt = REPO_ROOT / "outputs" / run / f"{net}.ckpt"
        if not ckpt.exists():
            print(f"[final] {spec}: no checkpoint, skipping")
            continue
        kw = dict(mono_ckpt=None, mv_ckpt=None) if net == "comp" else {}
        model = CLS[net].load_from_checkpoint(str(ckpt), **kw).to(device).eval()

        v_pos, v_yaw, a_pos, a_yaw = [], [], [], []
        for scene, G in geo.items():
            pg, mask, poses = G["pg"], G["mask"], G["poses"]
            rir_dir = root / "rir" / args.collection / args.condition / scene
            dsdir = str(root / args.collection)
            dataset = GridSeqDataset(dsdir, [scene], L=L, depth_dir=dsdir,
                                     depth_suffix="depth40" if net == "mono" else "depth160")
            picks = np.linspace(0, len(dataset) - 1,
                                min(args.n_poses, len(dataset))).astype(int)
            for chunk in tqdm.tqdm(picks, desc=f"{run} {scene}", leave=False):
                pose_idx = int(chunk) * (L + 1) + L
                f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
                if not f.exists():
                    continue
                data = dataset[int(chunk)]
                with torch.no_grad():
                    if net == "mono":
                        pred, _, _ = model.encoder(
                            torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
                    else:
                        b = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                             for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                        b["ref_mask"] = b["src_mask"] = None
                        pred = (model.net(b)["d"] if net == "mv"
                                else model.comp_d_net(b)["d_comp"])
                rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                    device=device, dtype=torch.float32)
                prob_vol, _, _, _ = localize(G["desdf_t"], rays)
                vol = np.array(prob_vol, dtype=np.float64)
                vol[~mask] = 0.0

                gx, gy, gyaw = pg.pose_metric_to_grid(poses[pose_idx, :3])
                flat = vol.reshape(-1)
                order = np.argsort(-flat)[: args.topk]
                r_, c_, o_ = np.unravel_index(order, vol.shape)

                # vision alone: the best pose in the volume
                v_pos.append(float(np.hypot(c_[0] - gx, r_[0] - gy) * pg.grid_resolution_m))
                v_yaw.append(PoseGrid.orientation_error_deg(pg.bin_to_yaw(o_[0]), gyaw))

                # acoustics reorders the shortlist; orientation still vision's
                obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
                obs_f = featurize(obs / max(obs.sum(), 1e-20))
                cand = G["acoustic"][r_, c_]
                nw = min(cand.shape[2], obs_f.shape[1])
                sc = -np.abs(cand[:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))
                j = int(sc.argmax())
                a_pos.append(float(np.hypot(c_[j] - gx, r_[j] - gy) * pg.grid_resolution_m))
                a_yaw.append(PoseGrid.orientation_error_deg(pg.bin_to_yaw(o_[j]), gyaw))

        table[f"{net} (vision only)"] = metrics(v_pos, v_yaw)
        table[f"{net} + acoustic re-rank"] = metrics(a_pos, a_yaw)
        r = table[f"{net} (vision only)"]; a = table[f"{net} + acoustic re-rank"]
        print(f"[final] {net}: vision 1m {100*r['r1']:.1f}%  ->  + acoustic {100*a['r1']:.1f}%")

    print(f"\n{'model':34s} {'0.1m':>7s} {'0.5m':>7s} {'1m':>7s} {'1m/30d':>8s} {'median':>8s}")
    print("-" * 76)
    for k, v in PAPER.items():
        print(f"{k:34s} {100*v[0]:6.1f}% {100*v[1]:6.1f}% {100*v[2]:6.1f}% {100*v[3]:7.1f}% "
              f"{'-':>8s}")
    print("-" * 76)
    for k, m in table.items():
        print(f"{k:34s} {100*m['r01']:6.1f}% {100*m['r05']:6.1f}% {100*m['r1']:6.1f}% "
              f"{100*m['r1_30']:7.1f}% {m['median_m']:7.2f}m")
    print(f"\n(rows above the line are the published Gibson numbers; ours are "
          f"{list(table.values())[0]['n'] if table else 0} poses on this Replica dataset)")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / "final_f3loc_metrics.json"
    out.write_text(json.dumps({"topk": args.topk, "condition": args.condition,
                               "paper_reference": {k: list(v) for k, v in PAPER.items()},
                               "ours": table}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
