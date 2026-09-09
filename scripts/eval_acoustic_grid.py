#!/usr/bin/env python3
"""Score any rendered candidate grid against the real recordings.

The candidate grid is a re-render of the floorplan, so it differs from the
recording in two ways at once: the floorplan omits furniture, and its material
absorption is a single guessed constant. Those are separable. Rendering the same
floorplan at several absorption values and scoring each one says how much of the
gap is the material guess rather than the missing geometry -- if a different
absorption closes most of it, the model is mis-tuned; if none of them move the
number, the gap is geometric and no re-render will fix it.

Takes explicit grid paths so any variant can be compared on equal terms:

    python scripts/eval_acoustic_grid.py --scene apartment_2 \
        --grids apartment_2 apartment_2_absorb0.35 apartment_2_absorb0.55
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", required=True)
    p.add_argument("--grids", nargs="+", required=True,
                   help="stems under outputs/acoustic_grid (without .npz)")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--n-poses", type=int, default=150)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def featurize(env: np.ndarray) -> np.ndarray:
    """The `chan_shape` feature: per-channel temporal shape plus the energy
    split across channels, which is what carries direction."""
    per = env / env.sum(axis=-1, keepdims=True).clip(1e-12)
    total = env.sum(axis=(-1, -2))
    split = env.sum(axis=-1) / np.clip(total[..., None], 1e-12, None)
    return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid

    root = Path(args.dataset_root)
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))

    desdf = np.load(root / "desdf" / args.scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / args.scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / args.scene / "poses.txt") if l.strip()])

    rir_dir = root / "rir" / args.collection / args.condition / args.scene
    dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))[: args.n_poses]
    obs, gts = [], []
    for d in dirs:
        f = rir_dir / d / "rir.npy"
        i = int(d.split("_")[1])
        if not f.exists() or i >= len(poses):
            continue
        obs.append(envelope(np.load(f), args.guard_samples, args.usable_samples, win))
        gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
        gts.append((gx, gy))
    print(f"[eval] {args.scene}: {len(obs)} recordings ({args.condition})")

    results = {}
    for stem in args.grids:
        g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{stem}.npz"
        if not g.exists():
            print(f"[eval] {stem}: missing, skipping")
            continue
        blob = np.load(g)
        cand, index = blob["rir"], blob["index"]
        rows, cols = index[:, 0].astype(float), index[:, 1].astype(float)
        env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
        env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
        cf = featurize(env)

        err, pct, rank = [], [], []
        for o, (gx, gy) in zip(obs, gts):
            of = featurize(o / max(o.sum(), 1e-20))
            nw = min(cf.shape[2], of.shape[1])
            score = -np.abs(cf[:, :, :nw] - of[None, :, :nw]).sum(axis=(1, 2))
            b = int(score.argmax())
            gt = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
            err.append(float(np.hypot(cols[b] - gx, rows[b] - gy) * pg.grid_resolution_m))
            pct.append(float((score < score[gt]).mean()) * 100)
            rank.append(int((score > score[gt]).sum()) + 1)
        err = np.array(err)
        results[stem] = dict(n=len(err), candidates=int(len(cand)),
                             gt_percentile=float(np.mean(pct)),
                             gt_rank_median=float(np.median(rank)),
                             recall_1m=float((err < 1).mean()),
                             err_median_m=float(np.median(err)))

    print(f"\n{'grid':32s} {'GT_pct':>7s} {'GT_rank':>8s} {'<1m':>6s} {'err_med':>8s}")
    print("-" * 66)
    for k, r in results.items():
        print(f"{k:32s} {r['gt_percentile']:6.1f} {r['gt_rank_median']:8.0f} "
              f"{100*r['recall_1m']:5.0f}% {r['err_median_m']:7.2f}m")
    print("\n(GT_pct 50 = chance; the candidate count differs per grid so ranks "
          "are only comparable within a scene)")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / f"eval_acoustic_grid_{args.scene}.json"
    out.write_text(json.dumps({"scene": args.scene, "condition": args.condition,
                               "results": results}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
