#!/usr/bin/env python3
"""Make the acoustic score robust to what the floorplan model cannot know about.

The model is a floorplan: walls, floor, ceiling. The recording is a real room,
so it also contains everything the floorplan omits -- furniture, clutter, scan
geometry. Those show up as reflections the model has no way to predict, and the
symmetric distance currently in use penalises every one of them.

Three exploits of that asymmetry, none of which need a re-render:

  early window   walls 3-5 m away return within 17-30 ms; the 128 ms window in
                 use is mostly late scatter and reverberation, where furniture
                 dominates
  one-sided      furniture *adds* energy, it does not remove wall returns, so
                 only penalise energy the model predicts and the recording
                 lacks -- the model should be a subset of the observation
  peaks only     large flat surfaces give a few strong specular arrivals;
                 clutter gives many weak diffuse ones

    python scripts/robust_acoustic_sweep.py --scene apartment_2
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
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--spans-ms", type=float, nargs="+", default=[20, 40, 60, 128],
                   help="how much of the response to keep, from the direct peak")
    p.add_argument("--peaks", type=int, nargs="+", default=[0, 8, 16],
                   help="0 keeps the whole envelope; otherwise keep the N strongest windows")
    p.add_argument("--n-poses", type=int, default=120)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def envelope_span(rir: np.ndarray, guard: int, span_samples: int, win: int) -> np.ndarray:
    peak = int(np.abs(rir).max(axis=0).argmax())
    seg = rir[:, peak + guard: peak + guard + span_samples]
    if seg.shape[1] < span_samples:
        seg = np.pad(seg, ((0, 0), (0, span_samples - seg.shape[1])))
    n_win = max(1, span_samples // win)
    return (seg[:, : n_win * win] ** 2).reshape(rir.shape[0], n_win, win).sum(-1)


def keep_peaks(env: np.ndarray, n: int) -> np.ndarray:
    """Zero everything but the n strongest windows of each channel."""
    if n <= 0 or n >= env.shape[-1]:
        return env
    idx = np.argpartition(env, -n, axis=-1)[..., -n:]
    out = np.zeros_like(env)
    np.put_along_axis(out, idx, np.take_along_axis(env, idx, axis=-1), axis=-1)
    return out


def d_sym(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    return np.abs(c - o[None]).sum(axis=(1, 2))


def d_one_sided(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Penalise only what the model predicts and the recording does not have."""
    return np.clip(c - o[None], 0, None).sum(axis=(1, 2))


DISTANCES = {"symmetric": d_sym, "one_sided": d_one_sided}


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    from track1_core.floorplan import PoseGrid, valid_pose_mask

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
        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene
        dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))[: args.n_poses]

        for span_ms in args.spans_ms:
            span = int(round(args.sample_rate * span_ms / 1000.0))
            cenv = np.stack([envelope_span(c, args.guard_samples, span, win) for c in cand])
            cenv /= cenv.sum(axis=(1, 2), keepdims=True).clip(1e-20)
            obs, gts = [], []
            for d in dirs:
                f = rir_dir / d / "rir.npy"
                i = int(d.split("_")[1])
                if not f.exists() or i >= len(poses):
                    continue
                e = envelope_span(np.load(f), args.guard_samples, span, win)
                obs.append(e / max(e.sum(), 1e-20))
                gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
                gts.append((gx, gy))

            for npk in args.peaks:
                cf = keep_peaks(cenv, npk)
                ofs = [keep_peaks(o, npk) for o in obs]
                for dname, dfn in DISTANCES.items():
                    err, pct = [], []
                    for o, (gx, gy) in zip(ofs, gts):
                        score = -dfn(cf, o)
                        b = int(score.argmax())
                        gt = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
                        err.append(float(np.hypot(cols[b] - gx, rows[b] - gy) * pg.grid_resolution_m))
                        pct.append(float((score < score[gt]).mean()) * 100)
                    err = np.array(err)
                    key = (scene, f"{span_ms:g}ms", f"peaks={npk or 'all'}", dname)
                    results["/".join(key)] = dict(
                        gt_percentile=float(np.mean(pct)),
                        recall_1m=float((err < 1).mean()),
                        err_median_m=float(np.median(err)))

    for scene in args.scenes:
        rows_ = {k: v for k, v in results.items() if k.startswith(scene + "/")}
        if not rows_:
            continue
        base = rows_.get(f"{scene}/128ms/peaks=all/symmetric")
        print(f"\n=== {scene}   (baseline = 128ms / all windows / symmetric)")
        print(f"{'span/peaks/distance':40s} {'GT_pct':>7s} {'<1m':>6s} {'err_med':>8s}")
        for k in sorted(rows_, key=lambda k: -rows_[k]["gt_percentile"])[:10]:
            r = rows_[k]
            tag = "  <-- baseline" if base and k.endswith("128ms/peaks=all/symmetric") else ""
            print(f"{k.split('/',1)[1]:40s} {r['gt_percentile']:6.1f} "
                  f"{100*r['recall_1m']:5.0f}% {r['err_median_m']:7.2f}m{tag}")
        if base:
            print(f"{'baseline':40s} {base['gt_percentile']:6.1f} "
                  f"{100*base['recall_1m']:5.0f}% {base['err_median_m']:7.2f}m")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / "robust_acoustic_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
