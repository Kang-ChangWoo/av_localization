#!/usr/bin/env python3
"""Which acoustic feature discriminates best *among visual hypotheses*.

The feature was chosen once, on the wrong criterion. The sweep that picked the
2 ms envelope ranked configurations by recall when sound localises alone, and by
that measure the envelope wins: 23.7% against 14.5% for the banded STFT. But
sound does not localise alone in this method. It chooses among a handful of
places vision has already proposed, and the quantity that governs that is where
the true pose sits in the acoustic ranking, not whether it sits first. On that
measure the ordering reverses: the STFT puts truth at median rank 210 of 4507
and every envelope setting is worse, the best being 235.

There is also a stale reason in the repository for preferring the envelope. The
STFT was rejected for smearing each frame over 8 ms, which was correct when the
dataset was 8 kHz and an nfft of 64 spanned 8 ms. The dataset is now 48 kHz and
the same nfft spans 1.33 ms, which is *finer* than the 2 ms envelope, with a
frequency axis on top. That criticism no longer applies and the configuration
deserves to be re-measured rather than inherited.

So this reports three quantities per configuration, and they disagree:

    alone          recall at 1 m when the acoustic argmax is the answer
    GT rank        where truth sits in the acoustic ranking over all cells
    mode accuracy  how often the feature picks the visual hypothesis nearest
                   truth, which is what mode-level fusion actually asks of it

    python scripts/feature_sweep_modes.py --backbone unloc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--backbone", default="unloc")
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--local-radius-m", type=float, default=0.5,
                   help="disc the acoustic evidence of a hypothesis is read over")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "analysis" / "feature_sweep_modes.md")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "feature_sweep_modes.json")
    return p.parse_args()


def read_csv(path: Path) -> dict[str, np.ndarray]:
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if vals[0] in ("True", "False"):
            out[k] = np.array([v == "True" for v in vals]); continue
        try:
            out[k] = np.array([float(v) if v not in ("", "nan") else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def configs(sr: int):
    """The configurations worth comparing, with why each is here."""
    from track1_core.likelihood.grid_score import GridScoreConfig
    base = dict(sample_rate_hz=sr, direct_guard_samples=int(round(sr * 2 / 1000)),
                usable_samples=int(round(sr * 128 / 1000)))
    low = ((0, 500), (500, 1500), (1500, 4000))          # as shipped
    wide = ((0, 1500), (1500, 4000), (4000, 12000))      # uses the 48 kHz bandwidth
    fine = ((0, 375), (375, 750), (750, 1500), (1500, 4000))
    out = [
        ("envelope 1 ms", GridScoreConfig(feature="envelope", window_ms=1.0, **base)),
        ("envelope 2 ms", GridScoreConfig(feature="envelope", window_ms=2.0, **base)),
        # nfft 64 at 48 kHz is a 1.33 ms analysis window, finer than the envelope
        ("stft nfft 64 hop 16", GridScoreConfig(feature="stft_band", nfft=64, hop=16,
                                                bands=low, **base)),
        ("stft nfft 64 hop 32", GridScoreConfig(feature="stft_band", nfft=64, hop=32,
                                                bands=low, **base)),
        # 2.67 ms window, 375 Hz bins: trades the time advantage for frequency
        ("stft nfft 128 hop 32", GridScoreConfig(feature="stft_band", nfft=128, hop=32,
                                                 bands=low, **base)),
        ("stft nfft 128 fine bands", GridScoreConfig(feature="stft_band", nfft=128, hop=32,
                                                     bands=fine, **base)),
        # 5.33 ms, close to the window the 8 kHz criticism was aimed at
        ("stft nfft 256 hop 64", GridScoreConfig(feature="stft_band", nfft=256, hop=64,
                                                 bands=low, **base)),
        ("stft nfft 64 wide bands", GridScoreConfig(feature="stft_band", nfft=64, hop=16,
                                                    bands=wide, **base)),
    ]
    return out


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    import cv2
    import tqdm

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        band_energy, featurise, observation_rate, score,
    )
    from track1_core.provenance import stamp

    M = read_csv(args.analysis_dir / f"modes_{args.condition}_{args.backbone}.csv")
    modes_by_q: dict[str, list] = {}
    for i, qid in enumerate(M["query_id"]):
        modes_by_q.setdefault(str(qid), []).append(
            (int(M["mode"][i]), int(M["x"][i]), int(M["y"][i]), float(M["dist_gt_m"][i])))
    for q in modes_by_q:
        modes_by_q[q].sort()

    root = Path(args.dataset_root)
    acc: dict[str, dict[str, list]] = {}
    inputs = []

    for scene in args.scenes:
        g = REPO_ROOT / args.grid_dir / f"{scene}.npz"
        if not g.exists():
            continue
        inputs.append(g)
        blob = np.load(g)
        sr = int(json.loads(str(blob["config"]))["sample_rate"])
        cfgs = configs(sr)
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        occ = cv2.imread(str(root / args.collections[0] / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        res = pg.grid_resolution_m
        mask = valid_pose_mask(occ, pg)
        rows, cols = np.nonzero(mask)
        idx = blob["index"]
        lut = -np.ones(mask.shape, dtype=np.int64)
        lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
        pick = lut[rows, cols]
        present = pick >= 0
        cell_of = -np.ones(mask.shape, dtype=np.int64)
        cell_of[rows, cols] = np.arange(len(rows))

        cand = {}
        for name, cfg in tqdm.tqdm(cfgs, desc=f"{scene}: featurising", leave=False):
            f = np.stack([featurise(band_energy(c, cfg), cfg) for c in blob["rir"]])
            a = np.zeros((len(rows), f.shape[1], f.shape[2]), dtype=np.float32)
            a[present] = f[pick[present]]
            cand[name] = a
        del blob
        print(f"[geo] {scene}: {len(rows)} cells, {len(cfgs)} configurations")

        for coll in args.collections:
            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rd = root / "rir" / coll / args.condition / scene
            names = sorted(n for n in os.listdir(rd) if n.startswith("pose_"))
            picks = np.linspace(0, len(names) - 1,
                                min(args.n_poses, len(names))).astype(int)
            for k in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                i = int(names[int(k)].split("_")[1])
                f = rd / names[int(k)] / "rir.npy"
                qid = f"{coll}/{scene}/{i:05d}"
                if not f.exists() or i >= len(poses) or qid not in modes_by_q:
                    continue
                rir = np.load(f)
                rate = observation_rate(f)
                gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * res
                gt = int(dist.argmin())
                mm = modes_by_q[qid]
                mcells = np.array([cell_of[y, x] for _, x, y, _ in mm])
                mdist = np.array([d for _, _, _, d in mm])
                keep = mcells >= 0
                discs = [np.nonzero(np.hypot(cols - cols[c], rows - rows[c]) * res
                                    <= args.local_radius_m)[0] for c in mcells[keep]]
                truth_mode = int(np.argmin(mdist[keep])) if keep.any() else -1

                for name, cfg in cfgs:
                    obs = featurise(band_energy(rir, cfg, rate), cfg)
                    s = score(cand[name], obs, present, cfg)
                    a = acc.setdefault(name, {"alone": [], "rank": [], "pct": [],
                                              "mode_hit": [], "mode_err": []})
                    a["alone"].append(dist[int(s.argmax())] < 1.0)
                    r = int((s > s[gt]).sum()) + 1
                    a["rank"].append(r)
                    a["pct"].append(100 * (1 - r / len(rows)))
                    if truth_mode >= 0:
                        ev = np.array([s[d].max() for d in discs])
                        j = int(ev.argmax())
                        a["mode_hit"].append(j == truth_mode)
                        a["mode_err"].append(mdist[keep][j])

    if not acc:
        print("nothing scored")
        return 1
    out: list[str] = []
    W = out.append
    W(f"# Acoustic feature, judged on hypothesis discrimination\n")
    W(f"{len(next(iter(acc.values()))['alone'])} queries, {args.condition}, "
      f"visual hypotheses from `{args.backbone}`. Acoustic evidence per "
      f"hypothesis is the maximum over its {args.local_radius_m:g} m disc, the "
      f"choice the fusion sweep already selected.\n")
    W("\n| feature | alone @1m | GT rank | GT percentile | picks the right "
      "hypothesis | error of the picked hypothesis |")
    W("|---|---|---|---|---|---|")
    js = {}
    for name, a in acc.items():
        row = dict(alone=100 * float(np.mean(a["alone"])),
                   rank=float(np.median(a["rank"])),
                   pct=float(np.median(a["pct"])),
                   mode_hit=100 * float(np.mean(a["mode_hit"])),
                   mode_ok=100 * float(np.mean(np.array(a["mode_err"]) < 1.0)))
        js[name] = row
        W(f"| {name} | {row['alone']:.1f}% | {row['rank']:.0f} | {row['pct']:.2f} | "
          f"{row['mode_hit']:.1f}% | {row['mode_ok']:.1f}% within 1 m |")
    best_alone = max(js, key=lambda k: js[k]["alone"])
    best_rank = min(js, key=lambda k: js[k]["rank"])
    best_mode = max(js, key=lambda k: js[k]["mode_ok"])
    W(f"\nBest by acoustic-alone recall: **{best_alone}**. "
      f"Best by ground-truth rank: **{best_rank}**. "
      f"Best at choosing among visual hypotheses: **{best_mode}**.\n")
    W("\nThe last column is the one the method depends on. If it disagrees with "
      "the first, the feature was selected on the wrong criterion.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        dict(condition=args.condition, backbone=args.backbone, results=js,
             provenance=stamp(inputs=inputs)), indent=2))
    print("\n".join(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
