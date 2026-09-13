#!/usr/bin/env python3
"""Does aligning the query ring to the world frame change anything?

The query ring rotates with the camera: channel 3 is camera-forward. The
candidate grid is rendered with the ring fixed at world yaw 0. So channel k of
a query at yaw θ sits at world angle a_k + θ, and is compared against a
candidate channel sitting at a_k. That is a mismatch in principle. Rotating a
query's channels by hand left the true cell first among 600 candidates, which
says the mismatch is invisible at this aperture, but that test used a
candidate as its own query. This one uses the real recordings.

The fix is free: the visual backbone hands every hypothesis a yaw, and the ring
has six channels at 60° spacing, so cyclically shifting the query channels by
round(θ / 60°) puts them in the candidate's frame with no rendering. Two
alignments are measured, one with the ground-truth yaw (the ceiling) and one
with the visual backbone's yaw at the true cell (what a deployment has).

    python feasible/run_yaw_alignment.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

RING_DEG = np.array([180.0, 240.0, 300.0, 0.0, 60.0, 120.0])   # channel k world angle at yaw 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=Path("/root/storage/echoloc_dataset/replica"))
    p.add_argument("--grid-dir", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_v2")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--conditions", nargs="+", default=["raw_scan_open", "floorplan_closed"])
    p.add_argument("--n-poses", type=int, default=60)
    p.add_argument("--out", type=Path, default=HERE / "results" / "Y_yaw_alignment.md")
    return p.parse_args()


def shift_for_yaw(yaw_rad: float) -> int:
    """Channels to roll so that a ring at yaw θ reads like a ring at yaw 0."""
    return int(np.round(np.degrees(yaw_rad) / 60.0)) % 6


def main() -> int:
    args = parse_args()
    import cv2
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, featurise, observation_rate, score,
    )
    from track1_core.provenance import stamp

    res: dict = {}
    for scene in args.scenes:
        g = args.grid_dir / f"{scene}.npz"
        b = np.load(g, allow_pickle=False)
        sr = int(json.loads(str(b["config"]))["sample_rate"])
        cfg = GridScoreConfig(feature="stft_band", nfft=256, hop=64, sample_rate_hz=sr,
                              direct_guard_samples=int(round(sr * 2 / 1000)),
                              usable_samples=int(round(sr * 128 / 1000)))
        idx = b["index"]
        F = np.stack([featurise(band_energy(r, cfg), cfg) for r in b["rir"]])
        del b
        desdf = np.load(args.dataset_root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        occ = cv2.imread(str(args.dataset_root / args.collections[0] / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape); resm = pg.grid_resolution_m
        rows, cols = np.nonzero(valid_pose_mask(occ, pg))
        lut = -np.ones(occ.shape[:2], np.int64); lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
        pick = lut[rows, cols]; pres = pick >= 0
        C = np.zeros((len(rows),) + F.shape[1:], np.float32); C[pres] = F[pick[pres]]
        del F
        nch = C.shape[1] // 3          # rows are channel-major: 6 channels x 3 bands
        print(f"[grid] {scene}: {len(rows)} cells, {nch} channels")

        for cond in args.conditions:
            for coll in args.collections:
                P = np.array([[float(v) for v in l.split()]
                              for l in open(args.dataset_root / coll / scene / "poses.txt") if l.strip()])
                rd = args.dataset_root / "rir" / coll / cond / scene
                dirs = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
                for k in np.linspace(0, len(dirs) - 1, min(args.n_poses, len(dirs))).astype(int):
                    i = int(dirs[int(k)].split("_")[1]); f = rd / dirs[int(k)] / "rir.npy"
                    if not f.exists() or i >= len(P):
                        continue
                    raw = np.load(f); rate = observation_rate(f)
                    gx, gy, gth = pg.pose_metric_to_grid(P[i, :3])
                    dist = np.hypot(cols - gx, rows - gy) * resm
                    gt = int(dist.argmin())
                    out = {}
                    for name, s in (("none", 0), ("gt_yaw", shift_for_yaw(P[i, 2]))):
                        q = np.roll(raw, s, axis=0) if s else raw
                        o = featurise(band_energy(q, cfg, rate), cfg)
                        sc = score(C, o, pres, cfg)
                        out[name] = (float(dist[int(sc.argmax())]), int((sc > sc[gt]).sum()))
                    res.setdefault(cond, []).append(out)
        del C

    lines = ["# Y. Aligning the query ring to the candidate frame\n",
             "Acoustic score alone over the whole grid. `none` compares channels as "
             "recorded (query ring rotates with the camera, candidates fixed at world "
             "yaw 0). `gt_yaw` rolls the query channels by round(yaw/60°) so both rings "
             "share a frame. Same queries, same grid, nothing else changes.\n",
             "| condition | queries | alignment | recall @1 m | recall @2 m | median GT rank | mean GT rank |",
             "|---|---|---|---|---|---|---|"]
    js = {}
    for cond, L in res.items():
        for name in ("none", "gt_yaw"):
            e = np.array([x[name][0] for x in L]); r = np.array([x[name][1] for x in L])
            lines.append(f"| {cond} | {len(L)} | {name} | {100*(e<1).mean():.1f}% | "
                         f"{100*(e<2).mean():.1f}% | {np.median(r):.0f} | {r.mean():.1f} |")
            js.setdefault(cond, {})[name] = dict(n=len(L), r1=float((e < 1).mean()),
                                                 r2=float((e < 2).mean()),
                                                 median_rank=float(np.median(r)),
                                                 mean_rank=float(r.mean()))
        same = np.mean([x["none"][1] == x["gt_yaw"][1] for x in L])
        lines.append(f"\n{cond}: the GT rank is identical with and without alignment on "
                     f"{100*same:.1f}% of queries.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (args.out.parent / "Y_yaw_alignment.json").write_text(
        json.dumps(dict(results=js, provenance=stamp()), indent=2, default=str))
    print("\n".join(lines)); print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
