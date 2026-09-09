#!/usr/bin/env python3
"""Score every acoustic candidate model through one identical pipeline.

The four models differ only in how a candidate's response is produced:

  A   analytic ray fan on the 2D floorplan, walls only
  A+  the same, plus floor and ceiling image sources (2.5D)
  B   image-source method on the extruded floorplan (pyroomacoustics)
  C   SoundSpaces on the extruded floorplan -- the engine that made the recording

Everything downstream is shared: the same valid-pose mask, the same candidate
cells, the same energy-envelope feature, the same L1 score, the same metrics.
Numbers reported separately by the earlier probes are *not* comparable -- one
ranked over poses including yaw, the other over positions -- so this exists to
put all four on one axis.

    python scripts/compare_acoustic_methods.py --scene office_4
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
    p.add_argument("--scene", default="office_4")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--n-poses", type=int, default=60)
    p.add_argument("--n-maps", type=int, default=3)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, free_space, valid_pose_mask
    from track1_core.likelihood.acoustic import (
        AcousticProxyConfig, synthesize_proxy, trace_echoes,
    )

    root = Path(args.dataset_root)
    scene = args.scene
    device = "cuda" if torch.cuda.is_available() else "cpu"
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, pg)
    rows, cols = np.nonzero(mask)
    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / scene / "poses.txt") if l.strip()])

    fp = json.loads((root / "floorplan_proxy" / scene / "floorplan.json").read_text())
    cdir = root / "rir" / args.collection / "floorplan_closed" / scene
    md = json.loads((cdir / sorted(d for d in os.listdir(cdir) if d.startswith("pose_"))[0]
                     / "rir_metadata.json").read_text())
    room_h = float(fp["room_height_m"])
    src_h = float(md["array_center_habitat"][1]) - float(fp["z_floor"])

    # ---- each model's candidate envelopes, on the same cells ---------------
    models: dict[str, np.ndarray] = {}

    def analytic(vertical_order: int) -> np.ndarray:
        cfg = AcousticProxyConfig(n_rays=72, max_order=2, energy_window_ms=args.window_ms,
                                  vertical_order=vertical_order, room_height_m=room_h,
                                  source_height_m=src_h, usable_samples=args.usable_samples,
                                  direct_guard_samples=args.guard_samples)
        free = torch.tensor(free_space(occ), device=device)
        origins = torch.tensor(pg.grid_to_map(np.stack([cols, rows], axis=-1)),
                               dtype=torch.float32, device=device)
        yaws = torch.zeros(1, device=device)          # match the yaw-1 rendered grids
        out = []
        for s in range(0, len(origins), args.chunk):
            o = origins[s: s + args.chunk]
            ech = trace_echoes(free, o, cfg, pg.map_resolution_m)
            out.append(synthesize_proxy(ech, o, yaws, pg, cfg).squeeze(1).cpu().numpy())
            del ech
            torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)            # (N, 6, n_win)

    print("[cmp] A  analytic, walls only")
    models["A"] = analytic(0)
    print("[cmp] A+ analytic, + floor/ceiling")
    models["A+"] = analytic(1)

    lut = -np.ones(mask.shape, dtype=int)
    for tag, fname in (("B", f"{scene}_pra.npz"), ("C", f"{scene}.npz")):
        f = REPO_ROOT / "outputs" / "acoustic_grid" / fname
        if not f.exists():
            print(f"[cmp] {tag}: {fname} not present, skipping")
            continue
        blob = np.load(f)
        cand, index = blob["rir"], blob["index"]
        env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
        lut[:] = -1
        lut[index[:, 0], index[:, 1]] = np.arange(len(index))
        pick = lut[rows, cols]
        aligned = np.zeros((len(rows), env.shape[1], env.shape[2]), dtype=np.float64)
        aligned[pick >= 0] = env[pick[pick >= 0]]
        models[tag] = aligned
        print(f"[cmp] {tag}  {fname}  {(pick>=0).mean()*100:.0f}% of cells covered")

    for k in models:
        models[k] = models[k] / models[k].sum(axis=(1, 2), keepdims=True).clip(1e-20)

    # ---- score all models against the same recordings ----------------------
    rir_dir = root / "rir" / args.collection / args.condition / scene
    dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))[: args.n_poses]
    results = {k: dict(err=[], pct=[], rank=[]) for k in models}
    keep = []
    for d in dirs:
        f = rir_dir / d / "rir.npy"
        if not f.exists():
            continue
        idx = int(d.split("_")[1])
        if idx >= len(poses):
            continue
        obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
        obs = obs / max(obs.sum(), 1e-20)
        gx, gy, _ = pg.pose_metric_to_grid(poses[idx, :3])
        gt = int(np.argmin((cols - gx) ** 2 + (rows - gy) ** 2))
        keep.append((idx, gx, gy, obs))
        for k, env in models.items():
            n_win = min(env.shape[2], obs.shape[1])
            score = np.exp(-np.abs(env[:, :, :n_win] - obs[None, :, :n_win]).sum(axis=(1, 2))
                           / args.temperature)
            b = int(score.argmax())
            results[k]["err"].append(float(np.hypot(cols[b] - gx, rows[b] - gy) * pg.grid_resolution_m))
            results[k]["pct"].append(float((score < score[gt]).mean()) * 100)
            results[k]["rank"].append(int((score > score[gt]).sum()) + 1)

    summary = {}
    print(f"\n[cmp] {scene} / {args.condition} / {len(keep)} recordings / {len(rows)} candidates")
    print(f"    {'model':5s} {'err_med':>8s} {'<1m':>6s} {'<2m':>6s} {'GT_pct':>7s} {'GT_rank':>8s}")
    for k, r in results.items():
        e = np.array(r["err"])
        summary[k] = dict(err_median_m=float(np.median(e)),
                          recall_1m=float((e < 1).mean()), recall_2m=float((e < 2).mean()),
                          gt_percentile=float(np.mean(r["pct"])),
                          gt_rank_median=float(np.median(r["rank"])),
                          candidates=int(len(rows)))
        s = summary[k]
        print(f"    {k:5s} {s['err_median_m']:7.2f}m {100*s['recall_1m']:5.0f}% "
              f"{100*s['recall_2m']:5.0f}% {s['gt_percentile']:6.1f} "
              f"{s['gt_rank_median']:7.0f}/{s['candidates']}")

    out_dir = args.out_dir or REPO_ROOT / "outputs" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "outputs" / "metrics" / f"compare_acoustic_{scene}.json").write_text(
        json.dumps({"scene": scene, "condition": args.condition, "models": summary}, indent=2))

    # ---- figure ------------------------------------------------------------
    order = [k for k in ("A", "A+", "B", "C") if k in models]
    crop = cv2.resize(occ[pg.top:pg.top + pg.height * 10, pg.left:pg.left + pg.width * 10],
                      (pg.width, pg.height), interpolation=cv2.INTER_NEAREST)
    nmap = min(args.n_maps, len(keep))
    fig = plt.figure(figsize=(4.2 * len(order), 3.6 * (nmap + 1)))
    gs = fig.add_gridspec(nmap + 1, len(order), height_ratios=[1] * nmap + [1.1], hspace=0.3)

    picks = np.linspace(0, len(keep) - 1, nmap).astype(int)
    for r_i, pi in enumerate(picks):
        idx, gx, gy, obs = keep[pi]
        for c_i, k in enumerate(order):
            env = models[k]
            n_win = min(env.shape[2], obs.shape[1])
            score = np.exp(-np.abs(env[:, :, :n_win] - obs[None, :, :n_win]).sum(axis=(1, 2))
                           / args.temperature)
            heat = np.zeros(mask.shape)
            heat[rows, cols] = score
            b = int(score.argmax())
            err = np.hypot(cols[b] - gx, rows[b] - gy) * pg.grid_resolution_m
            ax = fig.add_subplot(gs[r_i, c_i])
            ax.imshow(crop, cmap="gray", origin="lower", alpha=0.5)
            ax.imshow(np.ma.masked_where(~mask, heat), cmap="viridis", origin="lower", alpha=0.8)
            ax.plot(gx, gy, "o", ms=9, mfc="none", mec="lime", mew=2)
            ax.plot(cols[b], rows[b], "x", ms=8, color="red", mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{k}   err {err:.2f} m", fontsize=9)
            if c_i == 0:
                ax.set_ylabel(f"pose {idx}", fontsize=9)

    ax = fig.add_subplot(gs[nmap, :])
    x = np.arange(len(order))
    pct = [summary[k]["gt_percentile"] for k in order]
    rec = [100 * summary[k]["recall_1m"] for k in order]
    ax.bar(x - 0.2, pct, 0.4, label="GT percentile (higher = truth ranks better)", color="#2a9d8f")
    ax.bar(x + 0.2, rec, 0.4, label="recall @ 1 m [%]", color="#e76f51")
    for i, (p, r) in enumerate(zip(pct, rec)):
        ax.text(i - 0.2, p, f"{p:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, r, f"{r:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(
        [f"{k}\n{ {'A':'2D walls','A+':'+floor/ceiling','B':'image sources','C':'SoundSpaces'}[k] }"
         for k in order], fontsize=9)
    ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("candidate model fidelity vs localization quality", fontsize=10)

    fig.suptitle(f"Acoustic candidate models on one pipeline -- {scene}, {args.condition}\n"
                 f"{len(keep)} recordings, {len(rows)} candidate positions "
                 f"(green = truth, red = argmax)", fontsize=12)
    out = out_dir / f"compare_acoustic_{scene}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n[cmp] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
