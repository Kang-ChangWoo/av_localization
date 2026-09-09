#!/usr/bin/env python3
"""Summarise what the SoundSpaces candidate grid (method C) achieved.

Four panels:

  A  where the truth ranks in the acoustic likelihood, per scene, for a matched
     model (the recording made on the same floorplan mesh) and the realistic one
     (the recording made in the real scanned room)
  B  vision proposes, acoustics chooses: recall against the size of the shortlist,
     with vision's own accuracy and the oracle ceiling for context
  C  the product fusion for comparison -- how quickly it degrades as the acoustic
     term gains weight
  D  the acoustic likelihood itself on the floorplan for a few poses

    python scripts/plot_c_results.py
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
    p.add_argument("--run-name", default="echoloc_mono")
    p.add_argument("--map-scene", default="office_4")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--n-maps", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
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

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask

    M = REPO_ROOT / "outputs" / "metrics"
    scenes = ["office_4", "apartment_2", "frl_apartment_5"]
    short = {"office_4": "office_4\n(simple room)",
             "apartment_2": "apartment_2\n(furnished)",
             "frl_apartment_5": "frl_apartment_5\n(furnished)"}

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.32, wspace=0.28)

    # ---- A: where the truth ranks ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    closed, raw, labels = [], [], []
    for s in scenes:
        f = M / f"acoustic_grid_{s}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())["results"]
        best_c = min((v for k, v in r.items() if "floorplan_closed" in k),
                     key=lambda v: v["gt_rank_median"])
        best_r = min((v for k, v in r.items() if "raw_scan_open" in k),
                     key=lambda v: v["gt_rank_median"])
        n = best_c["candidates"]
        closed.append(100 * best_c["gt_rank_median"] / n)
        raw.append(100 * best_r["gt_rank_median"] / n)
        labels.append(short[s])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, closed, 0.4, label="model = observation\n(floorplan mesh)", color="#2a9d8f")
    ax.bar(x + 0.2, raw, 0.4, label="real scanned room\n(furniture, no GT mesh)", color="#e76f51")
    for i, (c, r) in enumerate(zip(closed, raw)):
        ax.text(i - 0.2, c, f"{c:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, r, f"{r:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("truth's rank, as % of all candidates\n(lower is better)")
    ax.set_title("A  Where the truth ranks in the acoustic likelihood", fontsize=10, loc="left")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    # ---- B: vision proposes, acoustics chooses -----------------------------
    ax = fig.add_subplot(gs[0, 1])
    rr = json.loads((M / f"rerank_{args.run_name}.json").read_text())["scenes"]
    colors = {"office_4": "#e76f51", "apartment_2": "#264653", "frl_apartment_5": "#2a9d8f"}
    for s, r in rr.items():
        K = [t["K"] for t in r["topk"]]
        ax.plot(K, [100 * t["rerank_recall_1m"] for t in r["topk"]], "-o", ms=3,
                color=colors[s], label=f"{s}")
        ax.axhline(100 * r["visual_recall_1m"], color=colors[s], ls=":", lw=1)
        ax.plot(K, [100 * t["oracle_recall_1m"] for t in r["topk"]], "--", lw=0.8,
                color=colors[s], alpha=0.5)
    ax.set_xscale("log"); ax.set_xlabel("shortlist size K (log)")
    ax.set_ylabel("recall @ 1 m [%]")
    ax.set_title("B  Vision proposes, acoustics chooses\nsolid = achieved, dotted = vision alone, dashed = oracle",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ---- C: the product fusion, for contrast --------------------------------
    ax = fig.add_subplot(gs[0, 2])
    fc = json.loads((M / f"fusion_calibrated_{args.run_name}.json").read_text())
    w = np.array(fc["weights"])
    ax.plot(w, 100 * np.array(fc["recall1m_by_weight"]), "-o", ms=4, color="#264653")
    best = int(np.argmax(fc["recall1m_by_weight"]))
    ax.axvline(w[best], color="green", ls="--", lw=1)
    ax.annotate(f"best w={w[best]:.1f}\n{100*fc['recall1m_by_weight'][best]:.0f}%",
                (w[best], 100 * fc["recall1m_by_weight"][best]), fontsize=8,
                xytext=(8, -4), textcoords="offset points", color="green")
    ax.set_xlabel("acoustic weight w   (0 = vision only, 1 = acoustics only)")
    ax.set_ylabel("recall @ 1 m [%]")
    ax.set_title("C  Multiplying the two distributions instead\n(degrades once acoustics dominates)",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.3)

    # ---- D: the likelihood on the floorplan ---------------------------------
    root = Path(args.dataset_root)
    scene = args.map_scene
    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, pg)
    blob = np.load(REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz")
    cand, index = blob["rir"], blob["index"]
    win = max(1, int(round(8000 * args.window_ms / 1000.0)))
    env = np.stack([envelope(c, 16, 1024, win) for c in cand])
    env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
    crop = cv2.resize(occ[pg.top:pg.top + pg.height * 10, pg.left:pg.left + pg.width * 10],
                      (pg.width, pg.height), interpolation=cv2.INTER_NEAREST)

    inner = gs[1, :].subgridspec(1, args.n_maps, wspace=0.12)
    rir_dir = root / "rir" / args.collection / "raw_scan_open" / scene
    dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))
    picks = np.linspace(0, len(dirs) - 1, args.n_maps).astype(int)
    for j, pi in enumerate(picks):
        d = dirs[pi]
        idx = int(d.split("_")[1])
        obs = envelope(np.load(rir_dir / d / "rir.npy"), 16, 1024, win)
        obs /= max(obs.sum(), 1e-20)
        score = np.exp(-np.abs(env - obs[None]).sum(axis=(1, 2)) / 0.5)
        heat = np.zeros(mask.shape)
        heat[index[:, 0], index[:, 1]] = score
        gx, gy, _ = pg.pose_metric_to_grid(poses[idx, :3])
        b = int(score.argmax())
        err = np.hypot(index[b, 1] - gx, index[b, 0] - gy) * pg.grid_resolution_m

        ax = fig.add_subplot(inner[0, j])
        ax.imshow(crop, cmap="gray", origin="lower", alpha=0.5)
        ax.imshow(np.ma.masked_where(~mask, heat), cmap="viridis", origin="lower", alpha=0.8)
        ax.plot(gx, gy, "o", ms=10, mfc="none", mec="lime", mew=2)
        ax.plot(index[b, 1], index[b, 0], "x", ms=9, color="red", mew=2)
        ax.set_title(f"pose {idx}   err {err:.2f} m", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("D  acoustic likelihood on the floorplan\n"
                          "(green = truth, red = argmax)", fontsize=9)

    fig.suptitle("Method C -- candidate room impulse responses rendered with SoundSpaces on the "
                 "2D floorplan, matched against a recording made in the real room", fontsize=13)
    out = args.out or REPO_ROOT / "outputs" / "viz" / "method_C_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
