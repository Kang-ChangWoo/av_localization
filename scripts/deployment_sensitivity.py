#!/usr/bin/env python3
"""Two things a deployment decides, measured on the grids we already rendered.

**How much noise does the query tolerate.** Every impulse response here is a
noiseless simulation, which is the honest limitation we state, but it leaves a
reader with no sense of the margin. Adding white noise to the query at a range
of signal-to-noise ratios says at what point the acoustic evidence stops being
usable, without pretending that white noise is what a real microphone adds.

**How coarse can the candidate grid be.** The grid costs about four seconds of
single-core simulation per cell and is the method's real deployment cost. It is
rendered at 0.1 m. Keeping every second or third cell cuts that by four or nine
times, and the question is what it costs in recall. Nothing is re-rendered: the
coarse grids are subsets of the one on disk, which is exactly what a deployment
would render instead.

Both are run on the same queries and the same policy as the main tables, so the
numbers sit next to them rather than beside them.

    python scripts/deployment_sensitivity.py --backbone f3STFT
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=Path("/root/storage/echoloc_dataset/replica"))
    p.add_argument("--grid-dir", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_v2")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--n-poses", type=int, default=40, help="per scene and collection")
    p.add_argument("--snr-db", nargs="+", type=float, default=[40, 30, 20, 10, 5, 0])
    p.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tex-dir", type=Path, default=REPO_ROOT / "docs" / "tables")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "deployment_sensitivity.md")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "deployment_sensitivity.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import cv2
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, featurise, observation_rate, score,
    )
    from track1_core.provenance import stamp

    rng = np.random.default_rng(args.seed)
    # per (snr, stride): distance from the acoustic argmax to ground truth
    err_snr: dict[float, list] = {s: [] for s in args.snr_db}
    err_snr["clean"] = []
    err_stride: dict[int, list] = {s: [] for s in args.strides}
    cells_kept: dict[int, list] = {s: [] for s in args.strides}

    for scene in args.scenes:
        g = args.grid_dir / f"{scene}.npz"
        if not g.exists():
            continue
        blob = np.load(g, allow_pickle=False)
        sr = int(json.loads(str(blob["config"]))["sample_rate"])
        cfg = GridScoreConfig(feature="stft_band", nfft=256, hop=64, sample_rate_hz=sr,
                              direct_guard_samples=int(round(sr * 2 / 1000)),
                              usable_samples=int(round(sr * 128 / 1000)))
        index = blob["index"]
        F = np.stack([featurise(band_energy(r, cfg), cfg) for r in blob["rir"]])
        del blob
        rows, cols = index[:, 0].astype(int), index[:, 1].astype(int)

        desdf = np.load(args.dataset_root / "desdf" / scene / "desdf.npy",
                        allow_pickle=True).item()
        occ = cv2.imread(str(args.dataset_root / args.collections[0] / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        res = pg.grid_resolution_m
        print(f"[grid] {scene}: {len(index)} cells at {res:.2f} m")

        # a coarser deployment grid is a subset of this one, taken on a lattice
        keep = {s: ((rows % s == 0) & (cols % s == 0)) for s in args.strides}
        for s in args.strides:
            cells_kept[s].append(int(keep[s].sum()))

        for coll in args.collections:
            poses = np.array([[float(v) for v in l.split()]
                              for l in open(args.dataset_root / coll / scene / "poses.txt")
                              if l.strip()])
            rd = args.dataset_root / "rir" / coll / args.condition / scene
            if not rd.is_dir():
                continue
            dirs = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
            picks = np.linspace(0, len(dirs) - 1, min(args.n_poses, len(dirs))).astype(int)
            for k in picks:
                dn = dirs[int(k)]
                i = int(dn.split("_")[1])
                f = rd / dn / "rir.npy"
                if not f.exists() or i >= len(poses):
                    continue
                raw = np.load(f)
                gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * res
                rate = observation_rate(f)

                def best(sig, mask=None):
                    o = featurise(band_energy(sig, cfg, rate), cfg)
                    m = np.ones(len(F), bool) if mask is None else mask
                    sc = score(F[m], o, np.ones(int(m.sum()), bool), cfg)
                    return float(dist[m][int(sc.argmax())])

                err_snr["clean"].append(best(raw))
                # noise is added to the query only: the candidate grid is what a
                # deployment rendered offline and is noiseless by construction
                p_sig = float(np.mean(raw.astype(np.float64) ** 2))
                for snr in args.snr_db:
                    sigma = np.sqrt(p_sig / (10 ** (snr / 10.0)))
                    err_snr[snr].append(best(raw + rng.normal(0, sigma, raw.shape).astype(raw.dtype)))
                for s in args.strides:
                    err_stride[s].append(best(raw, keep[s]))
        del F

    def line(v):
        e = np.asarray(v)
        return (100 * float((e < 1).mean()), 100 * float((e < 2).mean()), float(np.median(e)))

    out, js = [], {}
    W = out.append
    W("# What a deployment can give up\n")
    W("Acoustic score alone, so the effect is not diluted by the visual term. "
      "Same queries and same feature as the main tables.\n")

    W("\n## Query noise\n")
    W("White noise added to the query recording only; the candidate grid is "
      "rendered offline and is noiseless by construction. This is not a claim "
      "that a real microphone adds white noise, only a measure of the margin.\n")
    W("| query SNR | acoustic @1 m | @2 m | median |")
    W("|---|---|---|---|")
    r1, r2, md = line(err_snr["clean"])
    W(f"| clean | {r1:.1f}% | {r2:.1f}% | {md:.2f} m |")
    js["snr"] = {"clean": dict(r1=r1, r2=r2, median=md)}
    for s in args.snr_db:
        r1, r2, md = line(err_snr[s])
        W(f"| {s:g} dB | {r1:.1f}% | {r2:.1f}% | {md:.2f} m |")
        js["snr"][f"{s:g}dB"] = dict(r1=r1, r2=r2, median=md)

    W("\n## Candidate grid resolution\n")
    W("The coarse grids are lattice subsets of the rendered one, which is what a "
      "deployment would render instead. Rendering cost falls with the square of "
      "the stride.\n")
    W("| grid | cells | render cost | acoustic @1 m | @2 m | median |")
    W("|---|---|---|---|---|---|")
    js["stride"] = {}
    base = sum(cells_kept[1]) if 1 in cells_kept else 0
    for s in args.strides:
        n = sum(cells_kept[s])
        r1, r2, md = line(err_stride[s])
        W(f"| {0.1*s:.1f} m | {n} | {n/max(base,1):.0%} | {r1:.1f}% | {r2:.1f}% | {md:.2f} m |")
        js["stride"][f"{0.1*s:.1f}m"] = dict(cells=n, fraction=n / max(base, 1),
                                             r1=r1, r2=r2, median=md)

    # ------------------------------------------------------------------ tex
    args.tex_dir.mkdir(parents=True, exist_ok=True)
    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{What a deployment can give up, measured on the acoustic "
           r"score alone so the visual term does not dilute the effect. Left: "
           r"white noise added to the query only, since the candidate grid is "
           r"rendered offline and is noiseless by construction. Right: coarser "
           r"candidate grids, taken as lattice subsets of the rendered one, with "
           r"rendering cost falling as the square of the spacing.}",
           r"\label{tab:deploy}",
           r"\begin{tabular}{lcc@{\hskip 2em}lccc}\toprule",
           r"Query SNR & $1$\,m & median & Grid & cells & $1$\,m & median \\\midrule"]
    left = [("clean", *line(err_snr["clean"]))] + \
           [(f"${s:g}$\\,dB", *line(err_snr[s])) for s in args.snr_db]
    right = [(f"${0.1*s:.1f}$\\,m", sum(cells_kept[s]), *line(err_stride[s]))
             for s in args.strides]
    for i in range(max(len(left), len(right))):
        a = (rf"{left[i][0]} & {left[i][1]:.1f} & {left[i][3]:.2f}"
             if i < len(left) else " &  & ")
        b = (rf"{right[i][0]} & {right[i][1]} & {right[i][2]:.1f} & {right[i][4]:.2f}"
             if i < len(right) else " &  &  & ")
        tex.append(f"{a} & {b} \\\\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    (args.tex_dir / "tab_deploy.tex").write_text(
        "% Generated by scripts/deployment_sensitivity.py -- do not edit by hand.\n"
        + "\n".join(tex) + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dict(results=js, provenance=stamp()),
                                        indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out} and tab_deploy.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
