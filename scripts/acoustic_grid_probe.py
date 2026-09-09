#!/usr/bin/env python3
"""Measure the acoustic likelihood built from SoundSpaces-rendered candidates.

Where ``acoustic_likelihood_probe.py`` synthesises candidate responses
analytically, this scores against a grid of responses rendered by the same
engine that produced the recordings, on the same floorplan mesh. Model and
observation then share one physics, which removes the model/observation gap
that bounds the analytic proxy.

Two observation conditions answer different questions:

* ``floorplan_closed`` -- the recording was made on the same mesh the candidates
  were rendered on. This is the ceiling of the approach: how well acoustics can
  localise when the map is a perfect description of the room.
* ``raw_scan_open`` -- the recording was made on the real Replica scan, with its
  furniture and scan holes. This is the deployable number: how well a floorplan
  model explains a real room.

    python scripts/acoustic_grid_probe.py --scene office_4 --conditions floorplan_closed raw_scan_open
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
    p.add_argument("--grid", type=Path, default=None,
                   help="default: outputs/acoustic_grid/<scene>.npz")
    p.add_argument("--conditions", nargs="+", default=["floorplan_closed", "raw_scan_open"])
    p.add_argument("--n-poses", type=int, default=50)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--window-ms", type=float, nargs="+", default=[2.0, 0.5, 0.125])
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def envelope(rir: np.ndarray, guard: int, usable: int, win: int) -> np.ndarray:
    """Energy envelope of the reflection window, aligned on the direct peak.

    Both the candidates and the recording are aligned the same way, so a
    constant engine gain and the direct sound both drop out.
    """
    peak = int(np.abs(rir).max(axis=0).argmax())
    seg = rir[:, peak + guard : peak + guard + usable]
    if seg.shape[1] < usable:
        seg = np.pad(seg, ((0, 0), (0, usable - seg.shape[1])))
    n_win = usable // win
    return (seg[:, : n_win * win] ** 2).reshape(rir.shape[0], n_win, win).sum(-1)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    from track1_core.floorplan import PoseGrid

    root = Path(args.dataset_root)
    scene = args.scene
    grid_path = args.grid or REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
    blob = np.load(grid_path, allow_pickle=False)
    cand, index = blob["rir"], blob["index"]
    print(f"[grid] {grid_path.name}: {cand.shape} candidates")

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    rows, cols = index[:, 0].astype(float), index[:, 1].astype(float)

    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / scene / "poses.txt") if l.strip()])

    results = {}
    for cond in args.conditions:
        rir_dir = root / "rir" / args.collection / cond / scene
        if not rir_dir.is_dir():
            print(f"[grid] {cond}: not present, skipping")
            continue
        pose_dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))[: args.n_poses]

        for win_ms in args.window_ms:
            win = max(1, int(round(args.sample_rate * win_ms / 1000.0)))
            cand_env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win)
                                 for c in cand])                       # (N, 6, n_win)
            cand_n = cand_env / cand_env.sum(axis=(1, 2), keepdims=True).clip(1e-20)

            errs, pcts, ranks = [], [], []
            for d in pose_dirs:
                f = rir_dir / d / "rir.npy"
                if not f.exists():
                    continue
                idx = int(d.split("_")[1])
                if idx >= len(poses):
                    continue
                obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
                obs_n = obs / max(obs.sum(), 1e-20)

                l1 = np.abs(cand_n - obs_n[None]).sum(axis=(1, 2))
                score = np.exp(-l1 / args.temperature)

                gx, gy, _ = pg.pose_metric_to_grid(poses[idx, :3])
                gt_cell = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
                best = int(score.argmax())
                errs.append(float(np.hypot(cols[best] - gx, rows[best] - gy) * pg.grid_resolution_m))
                pcts.append(float((score < score[gt_cell]).mean()) * 100)
                ranks.append(int((score > score[gt_cell]).sum()) + 1)

            if not errs:
                continue
            errs = np.array(errs)
            key = f"{cond}_win{win_ms}ms"
            results[key] = dict(
                poses=len(errs),
                position_error_m_median=float(np.median(errs)),
                position_recall_1m=float((errs < 1).mean()),
                position_recall_2m=float((errs < 2).mean()),
                gt_percentile_mean=float(np.mean(pcts)),
                gt_rank_median=float(np.median(ranks)),
                candidates=int(cand.shape[0]),
            )
            r = results[key]
            print(f"[grid] {cond:17s} win {win_ms:>5}ms: err_med={r['position_error_m_median']:.2f} m  "
                  f"<1m={100*r['position_recall_1m']:.0f}%  <2m={100*r['position_recall_2m']:.0f}%  "
                  f"GT_pct={r['gt_percentile_mean']:.1f}  GT_rank_med={r['gt_rank_median']:.0f}/{r['candidates']}")

    payload = {"scene": scene, "grid": str(grid_path), "results": results}
    out = args.out or REPO_ROOT / "outputs" / "metrics" / f"acoustic_grid_{scene}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[grid] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
