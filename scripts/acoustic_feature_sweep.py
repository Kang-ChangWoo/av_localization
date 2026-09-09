#!/usr/bin/env python3
"""Search for an acoustic feature that actually separates candidates.

The candidate model is not the bottleneck any more: with SoundSpaces responses
the truth already sits near the top of the ranking, yet re-ranking recovers only
about a tenth of the available headroom. That gap is a *feature* problem -- the
score cannot tell neighbouring candidates apart -- so this sweeps the two
choices that make up the score:

  representation   what part of the response is compared
  distance         how two of them are compared

Everything else (candidate grid, valid mask, poses, alignment) is held fixed, so
the numbers isolate the feature. Reported per combination: where the truth ranks
and how often the argmax lands within a metre.

    python scripts/acoustic_feature_sweep.py --scene office_4
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
    p.add_argument("--scene", default="office_4")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--grid", type=Path, default=None)
    p.add_argument("--window-ms", type=float, nargs="+", default=[0.5, 0.125])
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--n-poses", type=int, default=120)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------- features --
def feat_energy(env: np.ndarray) -> np.ndarray:
    """Linear energy per window -- the current baseline."""
    return env


def feat_log(env: np.ndarray) -> np.ndarray:
    """Log energy: a late reflection 30 dB down still gets a vote.

    Linear energy is dominated by the first strong arrivals, so two candidates
    that agree on those look identical no matter how their tails differ.
    """
    return np.log10(env + 1e-12)


def feat_power(env: np.ndarray) -> np.ndarray:
    """Compressive power law, between linear and log."""
    return env ** 0.3


def feat_onset(env: np.ndarray) -> np.ndarray:
    """Cumulative energy: encodes *when* energy arrives, not how much.

    Arrival time is the quantity that carries wall distance; comparing
    cumulative curves makes a timing shift a large difference even when the
    totals match.
    """
    c = np.cumsum(env, axis=-1)
    return c / c[..., -1:].clip(1e-12)


def feat_channel_shape(env: np.ndarray) -> np.ndarray:
    """Per-channel normalised envelopes plus the inter-channel energy split.

    Keeps the ring's relative levels as an explicit, small part of the vector
    instead of letting the 6x longer time axis drown them out.
    """
    per = env / env.sum(axis=-1, keepdims=True).clip(1e-12)          # (..., 6, W)
    total = env.sum(axis=(-1, -2))                                   # (...,)
    split = env.sum(axis=-1) / np.clip(total[..., None], 1e-12, None)  # (..., 6)
    return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)


FEATURES = {"energy": feat_energy, "log": feat_log, "power0.3": feat_power,
            "onset": feat_onset, "chan_shape": feat_channel_shape}


# ---------------------------------------------------------------- distances --
def d_l1(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    return np.abs(c - o[None]).sum(axis=(1, 2))


def d_l2(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    return np.sqrt(((c - o[None]) ** 2).sum(axis=(1, 2)))


def d_cosine(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    cf = c.reshape(len(c), -1)
    of = o.reshape(-1)
    num = cf @ of
    den = np.linalg.norm(cf, axis=1) * np.linalg.norm(of) + 1e-12
    return 1.0 - num / den


def d_corr(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    cf = c.reshape(len(c), -1)
    cf = cf - cf.mean(axis=1, keepdims=True)
    of = o.reshape(-1) - o.mean()
    num = cf @ of
    den = np.linalg.norm(cf, axis=1) * np.linalg.norm(of) + 1e-12
    return 1.0 - num / den


DISTANCES = {"L1": d_l1, "L2": d_l2, "cosine": d_cosine, "corr": d_corr}


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask

    root = Path(args.dataset_root)
    scene = args.scene
    grid_path = args.grid or REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
    blob = np.load(grid_path)
    cand, index = blob["rir"], blob["index"]

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
    rows, cols = index[:, 0].astype(float), index[:, 1].astype(float)

    rir_dir = root / args.collection and root / "rir" / args.collection / args.condition / scene
    dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))[: args.n_poses]
    print(f"[feat] {scene} / {args.condition} / {len(cand)} candidates / {len(dirs)} recordings")

    results, best = {}, None
    for win_ms in args.window_ms:
        win = max(1, int(round(args.sample_rate * win_ms / 1000.0)))
        cand_env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win)
                             for c in cand])
        cand_env = cand_env / cand_env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
        obs_env, gts = [], []
        for d in dirs:
            f = rir_dir / d / "rir.npy"
            idx = int(d.split("_")[1])
            if not f.exists() or idx >= len(poses):
                continue
            e = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
            obs_env.append(e / max(e.sum(), 1e-20))
            gx, gy, _ = pg.pose_metric_to_grid(poses[idx, :3])
            gts.append((gx, gy))

        for fname, fn in FEATURES.items():
            cf = fn(cand_env)
            ofs = [fn(o) for o in obs_env]
            for dname, dfn in DISTANCES.items():
                err, pct = [], []
                for o, (gx, gy) in zip(ofs, gts):
                    dist = dfn(cf, o)
                    score = -dist
                    b = int(score.argmax())
                    gt = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
                    err.append(float(np.hypot(cols[b] - gx, rows[b] - gy) * pg.grid_resolution_m))
                    pct.append(float((score < score[gt]).mean()) * 100)
                err = np.array(err)
                key = f"{fname}/{dname}/{win_ms}ms"
                results[key] = dict(recall_1m=float((err < 1).mean()),
                                    recall_2m=float((err < 2).mean()),
                                    err_median_m=float(np.median(err)),
                                    gt_percentile=float(np.mean(pct)))
                if best is None or results[key]["gt_percentile"] > results[best]["gt_percentile"]:
                    best = key

    print(f"\n{'feature/distance/window':34s} {'GT_pct':>7s} {'<1m':>6s} {'<2m':>6s} {'err_med':>8s}")
    for k in sorted(results, key=lambda k: -results[k]["gt_percentile"]):
        r = results[k]
        mark = "  <-- best" if k == best else ""
        print(f"{k:34s} {r['gt_percentile']:6.1f} {100*r['recall_1m']:5.0f}% "
              f"{100*r['recall_2m']:5.0f}% {r['err_median_m']:7.2f}m{mark}")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / f"feature_sweep_{scene}.json"
    out.write_text(json.dumps({"scene": scene, "condition": args.condition,
                               "best": best, "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
