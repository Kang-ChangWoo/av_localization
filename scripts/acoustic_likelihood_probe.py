#!/usr/bin/env python3
"""Measure what the geometric acoustic likelihood actually constrains.

Two questions this answers with numbers rather than argument:

1. Does reflection order help? Order 1 is the first-wall fan, which carries the
   same geometry as the visual DESDF cache (over 360 degrees rather than the
   camera's ~106). Order >= 2 adds multi-bounce paths with no visual counterpart.
2. Is the likelihood flat in yaw? The ring is 5 cm at 8 kHz, so the largest
   inter-microphone path difference is ~2.3 samples. Whether that is enough to
   resolve orientation is an empirical question.

Reported per order: position error of the argmax, the ground-truth pose's rank
and percentile in the likelihood (the diagnostic docs/data_contracts.md asks
for), and how peaked the yaw marginal is at the correct bin.

    python scripts/acoustic_likelihood_probe.py --scene office_4 --orders 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", default=None, help="default: the first DESDF scene")
    p.add_argument("--condition", default="floorplan_closed",
                   choices=["floorplan_closed", "raw_scan_open"])
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n-poses", type=int, default=20)
    p.add_argument("--n-rays", type=int, default=72)
    p.add_argument("--energy-window-ms", type=float, nargs="+", default=[2.0],
                   help="time resolution of the envelope; the ring's max path difference is ~0.29 ms")
    p.add_argument("--vertical-orders", type=int, nargs="+", default=[0],
                   help="floor/ceiling image-source orders; 0 is the flat 2D proxy")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--chunk", type=int, default=256, help="candidate cells per batch")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import numpy as np
    import torch

    from track1_core.floorplan import PoseGrid, free_space, mask_summary, valid_pose_mask
    from track1_core.likelihood.acoustic import (
        AcousticProxyConfig, observed_envelope, score_envelopes, synthesize_proxy, trace_echoes,
    )

    root = Path(args.dataset_root)
    scene = args.scene or sorted(os.listdir(root / "desdf"))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    grid = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, grid)
    # room height and receiver height come from the scene, not from defaults:
    # the vertical image sources are wrong at any other height
    fp = json.loads((root / "floorplan_proxy" / scene / "floorplan.json").read_text())
    rir_dir0 = root / "rir" / args.collection / args.condition / scene
    _sample = sorted(d for d in os.listdir(rir_dir0) if d.startswith("pose_"))[0]
    _md = json.loads((rir_dir0 / _sample / "rir_metadata.json").read_text())
    room_h = float(fp["room_height_m"])
    src_h = float(_md["array_center_habitat"][1]) - float(fp["z_floor"])
    print(f"[probe] scene={scene} condition={args.condition} room_h={room_h:.3f} src_h={src_h:.3f}")
    print(f"[probe] {json.dumps(mask_summary(mask, grid))}")

    # --- observed RIRs at ground-truth poses --------------------------------
    rir_dir = root / "rir" / args.collection / args.condition / scene
    if not rir_dir.is_dir():
        print(f"[probe] no RIRs at {rir_dir}")
        return 1
    poses_txt = [l.split() for l in open(root / args.collection / scene / "poses.txt") if l.strip()]
    pose_dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))
    observations = []
    for d in pose_dirs:
        f = rir_dir / d / "rir.npy"
        if not f.exists():
            continue
        idx = int(d.split("_")[1])
        if idx >= len(poses_txt):
            continue
        observations.append((idx, f))
        if len(observations) >= args.n_poses:
            break
    if not observations:
        print("[probe] no complete RIR poses yet")
        return 1
    print(f"[probe] {len(observations)} observed poses")

    # --- candidate cells ----------------------------------------------------
    rows, cols = np.nonzero(mask)
    origins_px = torch.tensor(
        grid.grid_to_map(np.stack([cols, rows], axis=-1)), dtype=torch.float32, device=device
    )
    yaw_bins = np.arange(grid.orientation_bins)
    yaws = torch.tensor(grid.bin_to_yaw(yaw_bins), dtype=torch.float32, device=device)
    free = torch.tensor(free_space(occ), device=device)
    N, O = origins_px.shape[0], len(yaw_bins)
    print(f"[probe] candidates: {N} cells x {O} yaw = {N*O:,} poses")

    results = {}
    for order in args.orders:
      for win_ms in args.energy_window_ms:
       for vo in args.vertical_orders:
        cfg = AcousticProxyConfig(n_rays=args.n_rays, max_order=order, energy_window_ms=win_ms,
                                  vertical_order=vo, room_height_m=room_h, source_height_m=src_h)
        env_chunks = []
        for s in range(0, N, args.chunk):
            o = origins_px[s : s + args.chunk]
            ech = trace_echoes(free, o, cfg, grid.map_resolution_m)
            env_chunks.append(synthesize_proxy(ech, o, yaws, grid, cfg).cpu())
            del ech
            torch.cuda.empty_cache()
        proxy = torch.cat(env_chunks, dim=0)                # (N, O, 6, n_win) on CPU
        del env_chunks

        def scored(obs_env):
            # the envelope grows with time resolution, so score in slices
            out = torch.empty(N, O)
            for s0 in range(0, N, args.chunk):
                blk = proxy[s0 : s0 + args.chunk].to(device, non_blocking=True)
                out[s0 : s0 + args.chunk] = score_envelopes(blk, obs_env, args.temperature).cpu()
                del blk
            torch.cuda.empty_cache()
            return out

        pos_err, yaw_err, gt_pct, yaw_flat = [], [], [], []
        for idx, f in observations:
            obs = torch.tensor(observed_envelope(np.load(f), cfg), dtype=torch.float32, device=device)
            score = scored(obs)                                       # (N, O) on CPU

            gx, gy, gyaw = grid.pose_metric_to_grid(
                np.array([float(v) for v in poses_txt[idx]][:3])
            )
            gt_cell = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
            gt_bin = int(grid.yaw_to_bin(gyaw))

            flat = score.reshape(-1)
            best = int(flat.argmax())
            b_cell, b_bin = best // O, best % O
            pos_err.append(
                float(np.hypot(cols[b_cell] - gx, rows[b_cell] - gy) * grid.grid_resolution_m)
            )
            yaw_err.append(PoseGrid.orientation_error_deg(grid.bin_to_yaw(b_bin), gyaw))
            gt_score = float(score[gt_cell, gt_bin])
            gt_pct.append(float((flat < gt_score).float().mean()) * 100)

            # how peaked is yaw once position is fixed at the truth?
            y = score[gt_cell]
            y = y / y.sum().clamp_min(1e-12)
            ent = float(-(y * y.clamp_min(1e-12).log()).sum())
            yaw_flat.append(ent / np.log(O))          # 1.0 = perfectly flat

        key = f"order{order}_vert{vo}_win{win_ms}ms"
        results[key] = dict(
            position_error_m_median=float(np.median(pos_err)),
            position_recall_1m=float(np.mean(np.array(pos_err) < 1.0)),
            position_recall_2m=float(np.mean(np.array(pos_err) < 2.0)),
            yaw_error_deg_median=float(np.median(yaw_err)),
            gt_percentile_mean=float(np.mean(gt_pct)),
            yaw_entropy_ratio=float(np.mean(yaw_flat)),
        )
        r = results[key]
        print(f"[probe] order {order} vert {vo} win {win_ms:>5}ms: pos_err_med={r['position_error_m_median']:.2f} m  "
              f"recall<1m={100*r['position_recall_1m']:.0f}%  <2m={100*r['position_recall_2m']:.0f}%  "
              f"yaw_err_med={r['yaw_error_deg_median']:.0f} deg  "
              f"GT_pct={r['gt_percentile_mean']:.1f}  yaw_flatness={r['yaw_entropy_ratio']:.3f}")
        del proxy
        torch.cuda.empty_cache()

    payload = {"scene": scene, "collection": args.collection, "condition": args.condition,
               "n_poses": len(observations), "n_rays": args.n_rays,
               "candidates": {"cells": N, "yaw_bins": O}, "results": results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
