#!/usr/bin/env python3
"""Does a binaural pair recover heading, and is it worth the grid it needs?

The ring cannot answer this by construction. Six omnidirectional receivers on a
5 cm circle with a co-located source give inter-microphone delays of about seven
samples at 48 kHz, and rotating a query's channels leaves the true cell first
among six hundred candidates. So the acoustic score is flat across the 36 yaw
bins and every orientation the method reports is the visual backbone's.

A binaural pair is different: the engine's HRTF makes the two channels depend on
head orientation, so a candidate has to be rendered once per (cell, heading) and
the acoustic score becomes a function of both. That is six times the engine
calls per cell and 86 GB for three Replica scenes, so before adopting it we
measure what it buys.

Two questions, in increasing order of how much they would have to be believed:

    oracle heading   at the ground-truth cell, does the binaural score's argmax
                     over the 36 headings land within 30 degrees of the true
                     camera yaw? This is the ceiling. If it fails here, nothing
                     downstream can work.
    practical gain   at the cell the method already selects, is the binaural
                     heading better than the backbone's own? This is what would
                     actually go in a table, and it is bounded by the ceiling.

    python scripts/binaural_heading_eval.py --scenes office_4
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--grid-dir", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_binaural")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4"])
    p.add_argument("--condition", default="raw_scan_open",
                   choices=["raw_scan_open", "floorplan_closed"])
    p.add_argument("--n-poses", type=int, default=60, help="per scene and collection")
    p.add_argument("--feature", default="stft_band")
    p.add_argument("--nfft", type=int, default=256)
    p.add_argument("--hop", type=int, default=64)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "binaural_heading.md")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "binaural_heading.json")
    return p.parse_args()


def yaw_err_deg(a: float, b: float) -> float:
    d = (float(a) - float(b)) % (2 * np.pi)
    return min(d, 2 * np.pi - d) / np.pi * 180.0


def main() -> int:
    args = parse_args()
    import cv2
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, featurise, observation_rate, score,
    )
    from track1_core.provenance import stamp

    out: list[str] = []
    W = out.append
    js: dict = {}
    W("# Does binaural give heading\n")
    W("The ring cannot: its channels are near-identical, so the acoustic score "
      "is flat across the yaw bins and orientation comes from vision. A binaural "
      "candidate grid is rendered once per (cell, heading), which is six times "
      "the engine calls per cell, so the question is what that buys.\n")

    rows_all = []
    for scene in args.scenes:
        g = args.grid_dir / f"{scene}.npz"
        if not g.exists():
            print(f"[skip] {g} not merged yet")
            continue
        blob = np.load(g, allow_pickle=False)
        cfg_j = json.loads(str(blob["config"]))
        sr = int(cfg_j["sample_rate"])
        n_yaw = int(cfg_j.get("yaw_bins", 36))
        cfg = GridScoreConfig(feature=args.feature, nfft=args.nfft, hop=args.hop,
                              sample_rate_hz=sr,
                              direct_guard_samples=int(round(sr * 2 / 1000)),
                              usable_samples=int(round(sr * 128 / 1000)))
        index = blob["index"]           # (N, 3) rows, cols, yaw_bin
        print(f"[grid] {scene}: {len(index)} entries, {n_yaw} headings, "
              f"{len(np.unique(index[:, :2], axis=0))} cells")

        # featurise once; the array is (entries, C*B, T)
        F = np.stack([featurise(band_energy(r, cfg), cfg) for r in blob["rir"]])
        del blob
        key = index[:, 0].astype(np.int64) * 100000 + index[:, 1].astype(np.int64)
        cells, inv = np.unique(key, return_inverse=True)
        # entry position of (cell, yaw) so a query can be scored per heading
        slot = -np.ones((len(cells), n_yaw), dtype=np.int64)
        slot[inv, index[:, 2].astype(np.int64)] = np.arange(len(index))
        cell_rc = np.zeros((len(cells), 2), dtype=np.int64)
        cell_rc[inv] = index[:, :2]

        desdf = np.load(args.dataset_root / "desdf" / scene / "desdf.npy",
                        allow_pickle=True).item()
        occ = cv2.imread(str(args.dataset_root / args.collections[0] / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        res = pg.grid_resolution_m

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
                f = rd / dn / "rir_binaural.npy"
                if not f.exists() or i >= len(poses):
                    continue
                obs = featurise(band_energy(np.load(f), cfg, observation_rate(f)), cfg)
                gx, gy, gth = pg.pose_metric_to_grid(poses[i, :3])
                d = np.hypot(cell_rc[:, 1] - gx, cell_rc[:, 0] - gy) * res
                c_gt = int(d.argmin())
                if d[c_gt] > 0.5:
                    continue
                ok = slot[c_gt] >= 0
                if ok.sum() < n_yaw // 2:
                    continue
                s = score(F[slot[c_gt][ok]], obs, np.ones(int(ok.sum()), bool), cfg)
                b = np.nonzero(ok)[0][int(s.argmax())]
                pred_yaw = pg.bin_to_yaw(b) if hasattr(pg, "bin_to_yaw") else b / n_yaw * 2 * np.pi
                rows_all.append(dict(scene=scene, collection=coll, pose=i,
                                     err_deg=yaw_err_deg(pred_yaw, poses[i, 2]),
                                     gt_dist=float(d[c_gt])))
        del F

    if not rows_all:
        W("\nNo binaural grid merged yet, so nothing was measured.\n")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(out) + "\n")
        print("\n".join(out))
        return 1

    e = np.array([r["err_deg"] for r in rows_all])
    W("\n## Oracle heading at the ground-truth cell\n")
    W("The binaural score is evaluated at the true cell across all 36 headings "
      "and the best one is taken. Nothing downstream can beat this.\n")
    W(f"\n| queries | within 10 deg | within 30 deg | within 45 deg | median error |")
    W("|---|---|---|---|---|")
    W(f"| {len(e)} | {100*(e<10).mean():.1f}% | {100*(e<30).mean():.1f}% | "
      f"{100*(e<45).mean():.1f}% | {np.median(e):.1f} deg |")
    js["oracle_cell"] = dict(n=len(e), within10=float((e < 10).mean()),
                             within30=float((e < 30).mean()),
                             within45=float((e < 45).mean()),
                             median_deg=float(np.median(e)))
    chance30 = 60.0 / 360.0
    W(f"\nA uniformly random heading lands within 30 deg {100*chance30:.1f}% of "
      f"the time, so the number above is the one to compare against that.\n")
    js["chance_within30"] = chance30

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dict(results=js, rows=rows_all[:2000],
                                             provenance=stamp()), indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
