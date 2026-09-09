#!/usr/bin/env python3
"""Compare candidates in time-frequency, and ask which bands survive the geometry gap.

The feature in use is energy per time window, summed over all frequencies. That
throws away the one axis along which the model's error is structured: the
floorplan omits furniture, and furniture scatters as a function of wavelength.

    wavelength = 343 / f

  500 Hz -> 69 cm   longer than a chair or a table, so the wave diffracts around
                    it and what returns is mostly wall geometry
 3000 Hz ->  11 cm   comparable to furniture, so it scatters off exactly the
                    objects the floorplan does not have

If that reasoning holds, low-frequency bands should rank the truth better than
the broadband envelope, and high-frequency bands worse. If every band behaves
the same, the contamination is not frequency-structured and the time-only
feature was not losing anything.

    python scripts/stft_band_sweep.py --scenes office_4 apartment_2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# (name, low Hz, high Hz); 8 kHz sampling puts Nyquist at 4 kHz
BANDS = [
    ("full", 0, 4000),
    ("lo_0-500", 0, 500),
    ("lo_0-1k", 0, 1000),
    ("mid_500-1.5k", 500, 1500),
    ("mid_1-2k", 1000, 2000),
    ("hi_2-4k", 2000, 4000),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--nfft", type=int, default=64, help="STFT window in samples (8 ms at 8 kHz)")
    p.add_argument("--hop", type=int, default=16)
    p.add_argument("--n-poses", type=int, default=120)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def spectrogram(rir: np.ndarray, guard: int, usable: int, nfft: int, hop: int) -> np.ndarray:
    """Magnitude-squared STFT of the reflection window, ``(6, freq, frame)``."""
    peak = int(np.abs(rir).max(axis=0).argmax())
    seg = rir[:, peak + guard: peak + guard + usable]
    if seg.shape[1] < usable:
        seg = np.pad(seg, ((0, 0), (0, usable - seg.shape[1])))
    w = np.hanning(nfft)
    frames = 1 + (usable - nfft) // hop
    out = np.empty((seg.shape[0], nfft // 2 + 1, frames))
    for t in range(frames):
        chunk = seg[:, t * hop: t * hop + nfft] * w
        out[:, :, t] = np.abs(np.fft.rfft(chunk, axis=-1)) ** 2
    return out


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    from track1_core.floorplan import PoseGrid

    root = Path(args.dataset_root)
    freqs = np.fft.rfftfreq(args.nfft, 1.0 / args.sample_rate)
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
        print(f"[stft] {scene}: {len(cand)} candidates")
        cspec = np.stack([spectrogram(c, args.guard_samples, args.usable_samples,
                                      args.nfft, args.hop) for c in cand])

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
            obs.append(spectrogram(np.load(f), args.guard_samples, args.usable_samples,
                                   args.nfft, args.hop))
            gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
            gts.append((gx, gy))

        for name, lo, hi in BANDS:
            sel = (freqs >= lo) & (freqs < hi)
            if not sel.any():
                continue
            c = cspec[:, :, sel, :].reshape(len(cspec), -1)
            c = c / c.sum(axis=1, keepdims=True).clip(1e-20)
            err, pct = [], []
            for o, (gx, gy) in zip(obs, gts):
                of = o[:, sel, :].reshape(-1)
                of = of / max(of.sum(), 1e-20)
                score = -np.abs(c - of[None]).sum(axis=1)
                b = int(score.argmax())
                gt = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
                err.append(float(np.hypot(cols[b] - gx, rows[b] - gy) * pg.grid_resolution_m))
                pct.append(float((score < score[gt]).mean()) * 100)
            err = np.array(err)
            results[f"{scene}/{name}"] = dict(gt_percentile=float(np.mean(pct)),
                                              recall_1m=float((err < 1).mean()),
                                              err_median_m=float(np.median(err)))

    for scene in args.scenes:
        rows_ = {k: v for k, v in results.items() if k.startswith(scene + "/")}
        if not rows_:
            continue
        print(f"\n=== {scene}")
        print(f"{'band':16s} {'GT_pct':>7s} {'<1m':>6s} {'err_med':>8s}")
        for name, _, _ in BANDS:
            k = f"{scene}/{name}"
            if k not in rows_:
                continue
            r = rows_[k]
            tag = "   <-- broadband" if name == "full" else ""
            print(f"{name:16s} {r['gt_percentile']:6.1f} {100*r['recall_1m']:5.0f}% "
                  f"{r['err_median_m']:7.2f}m{tag}")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / "stft_band_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
