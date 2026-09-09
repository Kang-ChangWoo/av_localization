#!/usr/bin/env python3
"""One-sided scoring, done so that the asymmetry survives.

The idea is physical: furniture the floorplan does not know about *adds*
reflections to the recording, it does not remove wall returns. So energy the
model predicts and the recording lacks is evidence against a candidate, while
energy the recording has and the model lacks is expected and should cost
nothing.

The earlier attempt could not express that. Both envelopes were normalised to
sum one, which forces sum(model - obs) = 0, so the one-sided penalty collapses
to exactly half the symmetric one and the ranking is identical. Keeping the
asymmetry means keeping a scale, which means fixing the gain between a rendered
candidate and a recording first -- they come from separate renders and differ by
a constant factor.

Three ways to fix that gain are compared, each with a symmetric and a one-sided
penalty, so the pair isolates what the asymmetry is worth.

    python scripts/asymmetric_score_sweep.py --scenes office_4 apartment_2
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
    p.add_argument("--scenes", nargs="+",
                   default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--n-poses", type=int, default=120)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# --- gain fixing: candidate and recording come from separate renders ---------
def gain_sum(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Match total energy. Furniture inflates the recording's total, so this
    systematically shrinks the model and hides the very excess we want."""
    return o.sum() / c.sum(axis=(1, 2)).clip(1e-20)


def gain_peak(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Match the strongest arrival, which is a wall/floor/ceiling return in both."""
    return o.max() / c.max(axis=(1, 2)).clip(1e-20)


def gain_quantile(c: np.ndarray, o: np.ndarray, q: float = 0.9) -> np.ndarray:
    """Match a high quantile: robust to the recording's extra weak arrivals."""
    oq = np.quantile(o, q)
    cq = np.quantile(c.reshape(len(c), -1), q, axis=1).clip(1e-20)
    return oq / cq


GAINS = {"sum": gain_sum, "peak": gain_peak, "q90": gain_quantile}


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid

    root = Path(args.dataset_root)
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
    results = {}

    for scene in args.scenes:
        g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
        if not g.exists():
            continue
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        blob = np.load(g)
        cand, index = blob["rir"], blob["index"]
        rows, cols = index[:, 0].astype(float), index[:, 1].astype(float)
        cenv = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])

        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene
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

        for gname, gfn in GAINS.items():
            for side in ("symmetric", "one_sided"):
                err, pct = [], []
                for o, (gx, gy) in zip(obs, gts):
                    k = gfn(cenv, o)[:, None, None]
                    diff = cenv * k - o[None]
                    d = (np.abs(diff) if side == "symmetric"
                         else np.clip(diff, 0, None)).sum(axis=(1, 2))
                    score = -d
                    b = int(score.argmax())
                    gt = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
                    err.append(float(np.hypot(cols[b] - gx, rows[b] - gy) * pg.grid_resolution_m))
                    pct.append(float((score < score[gt]).mean()) * 100)
                err = np.array(err)
                results[f"{scene}/{gname}/{side}"] = dict(
                    gt_percentile=float(np.mean(pct)),
                    recall_1m=float((err < 1).mean()),
                    err_median_m=float(np.median(err)))

    for scene in args.scenes:
        rows_ = {k: v for k, v in results.items() if k.startswith(scene + "/")}
        if not rows_:
            continue
        print(f"\n=== {scene}")
        print(f"{'gain / penalty':28s} {'GT_pct':>7s} {'<1m':>6s} {'err_med':>8s}")
        for k in sorted(rows_, key=lambda k: -rows_[k]["gt_percentile"]):
            r = rows_[k]
            print(f"{k.split('/',1)[1]:28s} {r['gt_percentile']:6.1f} "
                  f"{100*r['recall_1m']:5.0f}% {r['err_median_m']:7.2f}m")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / "asymmetric_score_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
