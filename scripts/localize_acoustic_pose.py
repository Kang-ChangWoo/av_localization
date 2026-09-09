#!/usr/bin/env python3
"""Localize an SE(2) pose from sound alone, searching position and heading.

The candidate grid must be yaw-resolved: at every cell the microphone ring is
rendered facing each of several headings, so a candidate is a pose, not just a
position. With a single fixed heading the search cannot say anything about
orientation at all.

The source sits at the ring centre, so the room response is decided by position
and the heading only moves the six microphones around a 5 cm circle. Whatever
orientation information exists therefore lives in inter-channel differences
bounded by ~2.3 samples at 8 kHz -- small, which is exactly why it needs
measuring rather than assuming.

    python scripts/localize_acoustic_pose.py --grid outputs/acoustic_grid/office_4_yaw4.npz
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
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", default=None, help="default: parsed from the grid filename")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-poses", type=int, default=100)
    p.add_argument("--n-panels", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tqdm

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask

    root = Path(args.dataset_root)
    scene = args.scene or args.grid.stem.split("_yaw")[0]
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "viz" / "acoustic_pose"
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = np.load(args.grid)
    cand, index = blob["rir"], blob["index"]
    n_yaw = int(index[:, 2].max()) + 1
    yaw_of_bin = np.arange(n_yaw) / n_yaw * 2 * np.pi
    print(f"[pose] {args.grid.name}: {cand.shape[0]} candidates, {n_yaw} headings "
          f"({np.degrees(yaw_of_bin).round(0)} deg)")

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, pg)

    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
    env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
    env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)

    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
    rir_dir = root / "rir" / args.collection / args.condition / scene
    pose_dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))[: args.max_poses]

    rows, cols, ybins = index[:, 0], index[:, 1], index[:, 2]
    records = []
    for d in tqdm.tqdm(pose_dirs, desc="localizing"):
        f = rir_dir / d / "rir.npy"
        if not f.exists():
            continue
        pose_idx = int(d.split("_")[1])
        if pose_idx >= len(poses):
            continue
        obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
        obs /= max(obs.sum(), 1e-20)
        score = np.exp(-np.abs(env - obs[None]).sum(axis=(1, 2)) / args.temperature)

        gx, gy, gyaw = pg.pose_metric_to_grid(poses[pose_idx, :3])
        best = int(score.argmax())
        pos_err = float(np.hypot(cols[best] - gx, rows[best] - gy) * pg.grid_resolution_m)
        yaw_err = PoseGrid.orientation_error_deg(yaw_of_bin[ybins[best]], gyaw)

        # position posterior: best heading at each cell
        heat = np.zeros(mask.shape)
        np.maximum.at(heat, (rows, cols), score)
        # heading posterior at the truth's cell
        near = np.argmin((cols - gx) ** 2 + (rows - gy) ** 2)
        gt_cell = (rows[near], cols[near])
        sel = (rows == gt_cell[0]) & (cols == gt_cell[1])
        head = np.zeros(n_yaw)
        head[ybins[sel]] = score[sel]
        head = head / max(head.sum(), 1e-20)

        records.append(dict(pose_idx=pose_idx, gx=gx, gy=gy, gyaw=gyaw,
                            px=float(cols[best]), py=float(rows[best]),
                            pyaw=float(yaw_of_bin[ybins[best]]),
                            pos_err=pos_err, yaw_err=yaw_err, heat=heat, head=head))

    if not records:
        print("[pose] no recordings matched the grid")
        return 1

    pe = np.array([r["pos_err"] for r in records])
    ye = np.array([r["yaw_err"] for r in records])
    chance = 180.0 / n_yaw * (n_yaw // 2) / max(n_yaw // 2, 1)   # mean |error| of a uniform guess
    chance = float(np.mean([PoseGrid.orientation_error_deg(a, 0.0) for a in yaw_of_bin]))
    summary = dict(scene=scene, condition=args.condition, headings=n_yaw,
                   candidates=int(cand.shape[0]), poses=len(records),
                   position_median_m=float(np.median(pe)),
                   position_recall_1m=float((pe < 1).mean()),
                   position_recall_2m=float((pe < 2).mean()),
                   yaw_median_deg=float(np.median(ye)),
                   yaw_chance_deg=chance,
                   yaw_correct_bin=float((ye <= 180.0 / n_yaw).mean()),
                   pose_recall_1m_90deg=float(((pe < 1) & (ye <= 90)).mean()))
    print(json.dumps(summary, indent=2))
    (REPO_ROOT / "outputs" / "metrics" / f"acoustic_pose_{scene}_{n_yaw}yaw.json").write_text(
        json.dumps(summary, indent=2))

    # ---- panels ------------------------------------------------------------
    order = np.argsort(pe)
    picks = np.unique(np.linspace(0, len(order) - 1, args.n_panels).astype(int))
    chosen = [records[order[i]] for i in picks]
    crop = cv2.resize(occ[pg.top:pg.top + pg.height * 10, pg.left:pg.left + pg.width * 10],
                      (pg.width, pg.height), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 4.0 * len(chosen)),
                             gridspec_kw={"width_ratios": [2, 1]})
    if len(chosen) == 1:
        axes = axes[None, :]
    L = 4.0    # heading arrow length, in grid cells
    for i, r in enumerate(chosen):
        ax = axes[i, 0]
        ax.imshow(crop, cmap="gray", origin="lower", alpha=0.5)
        ax.imshow(np.ma.masked_where(~mask, r["heat"]), cmap="viridis", origin="lower", alpha=0.75)
        ax.arrow(r["gx"], r["gy"], L * np.cos(r["gyaw"]), L * np.sin(r["gyaw"]),
                 color="lime", width=0.35, length_includes_head=True)
        ax.arrow(r["px"], r["py"], L * np.cos(r["pyaw"]), L * np.sin(r["pyaw"]),
                 color="red", width=0.35, length_includes_head=True)
        ax.plot(r["gx"], r["gy"], "o", ms=9, mfc="none", mec="lime", mew=2)
        ax.set_title(f"pose {r['pose_idx']}   pos {r['pos_err']:.2f} m   yaw {r['yaw_err']:.0f} deg",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[i, 1]
        ax.bar(np.degrees(yaw_of_bin), r["head"], width=360 / n_yaw * 0.8, color="steelblue")
        ax.axvline(np.degrees(r["gyaw"] % (2 * np.pi)), color="lime", lw=2, label="truth")
        ax.set_xlim(-30, 360); ax.set_xlabel("heading [deg]", fontsize=8)
        ax.set_ylabel("p(heading | position)", fontsize=8)
        ax.legend(fontsize=7); ax.set_title("heading posterior at the true cell", fontsize=9)

    fig.suptitle(f"acoustic pose from sound alone -- {scene} / {args.condition} / {n_yaw} headings\n"
                 f"median  position {np.median(pe):.2f} m,  heading {np.median(ye):.0f} deg "
                 f"(chance {chance:.0f} deg)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"pose_{scene}_{n_yaw}yaw_{args.condition}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
