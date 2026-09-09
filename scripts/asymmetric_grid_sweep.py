#!/usr/bin/env python3
"""Make the one-sided penalty actually do something, and measure whether it helps.

The asymmetry is the right model for clutter: furniture adds arrivals to a
recording and can hide the wall returns a floorplan predicts, but it cannot
invent them. So model energy the recording lacks is evidence against a
candidate, while recording energy the model lacks is expected and should cost
nothing.

Twice in this project that idea was implemented and measured as having exactly
no effect, both times because the features were normalised to sum to one. That
forces ``sum(model - obs) = 0``, which makes the clipped penalty precisely half
the symmetric one and leaves the ranking unchanged. The fix is not a better
penalty, it is a normalisation that does not impose the constraint: dividing by
the strongest arrival, or by a high quantile, keeps a scale and lets the
asymmetry exist.

The sweep crosses that with the two axes already known to matter, the frequency
band and the time window, and reports the acoustic score on its own so the
result cannot be confounded by a fusion rule.

    python scripts/asymmetric_grid_sweep.py --collections replica_f replica_g
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
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import tqdm

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, asymmetry_is_degenerate, candidate_features,
        observation_feature, score,
    )
    from track1_core.provenance import stamp

    base = GridScoreConfig()
    LOW = ((0, 1000),)
    THREE = ((0, 500), (500, 1500), (1500, 4000))
    late = base.frame_of_ms(30.0)

    combos = []
    for bands, bname in ((THREE, "three bands"), (LOW, "low band")):
        for f0, wname in ((0, "whole window"), (late, "late >30 ms")):
            for norm in ("sum", "peak", "quantile"):
                for dist in ("symmetric", "one_sided"):
                    cfg = GridScoreConfig(bands=bands, first_frame=f0,
                                          normalise=norm, distance=dist)
                    if asymmetry_is_degenerate(cfg):
                        continue        # provably identical to its symmetric twin
                    combos.append((f"{bname}, {wname}, {norm}, {dist}", cfg))
    print(f"[sweep] {len(combos)} configurations "
          f"(sum + one_sided pairs dropped as provably degenerate)")

    root = Path(args.dataset_root)
    recs = {lab: [] for lab, _ in combos}
    inputs = []

    for coll in args.collections:
        for scene in args.scenes:
            g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
            if not g.exists() or not (root / coll / scene / "map.png").exists():
                continue
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows, cols = np.nonzero(mask)
            inputs.append(g)

            cache = {}
            for lab, cfg in combos:
                key = (cfg.bands, cfg.first_frame, cfg.normalise, cfg.shape_feature)
                if key not in cache:
                    cache[key] = candidate_features(g, rows, cols, mask.shape, cfg)
            print(f"[geo] {coll}/{scene}: {len(rows)} cells, {len(cache)} feature sets")

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rd = root / "rir" / coll / args.condition / scene
            ds = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
            ds = [ds[i] for i in np.linspace(0, len(ds) - 1,
                                             min(args.n_poses, len(ds))).astype(int)]

            for d in tqdm.tqdm(ds, desc=f"{coll}/{scene}", leave=False):
                i = int(d.split("_")[1])
                f = rd / d / "rir.npy"
                if not f.exists() or i >= len(poses):
                    continue
                rir = np.load(f)
                gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
                gt = int(dist.argmin())
                for lab, cfg in combos:
                    key = (cfg.bands, cfg.first_frame, cfg.normalise, cfg.shape_feature)
                    cand, present = cache[key]
                    s = score(cand, observation_feature(rir, cfg), present, cfg)
                    b = int(s.argmax())
                    recs[lab].append(dict(err=float(dist[b]),
                                          rank=int((s > s[gt]).sum()) + 1,
                                          n=len(rows)))

    if not any(recs.values()):
        print("nothing scored")
        return 1

    print(f"\nacoustic score on its own, {len(next(iter(recs.values())))} poses")
    print(f"{'configuration':52s} {'<1m':>6s} {'<2m':>6s} {'GT rank':>9s} {'pct':>6s}")
    print("-" * 86)
    out = {}
    rows_out = []
    for lab, _ in combos:
        g = recs[lab]
        if not g:
            continue
        e = np.array([x["err"] for x in g])
        rk = np.array([x["rank"] for x in g], dtype=float)
        nc = np.mean([x["n"] for x in g])
        m = dict(recall_1m=float((e < 1).mean()), recall_2m=float((e < 2).mean()),
                 gt_rank_median=float(np.median(rk)),
                 gt_percentile=float(100 * (1 - np.mean(rk) / nc)))
        out[lab] = m
        rows_out.append((m["recall_1m"], lab, m))
    for _, lab, m in sorted(rows_out, reverse=True):
        print(f"{lab:52s} {100*m['recall_1m']:5.1f}% {100*m['recall_2m']:5.1f}% "
              f"{m['gt_rank_median']:9.0f} {m['gt_percentile']:5.1f}")
    print("\nGT percentile 50 = chance. The sum/one_sided pairs are absent because "
          "they are\nprovably identical to their symmetric twins, not because they "
          "were not run.")

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "asymmetric_grid_sweep.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(condition=args.condition, results=out,
                                 provenance=stamp(inputs=inputs)), indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
