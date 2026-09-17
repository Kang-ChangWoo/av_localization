#!/usr/bin/env python3
"""What the semantic floor plans and semantic rays look like, a few examples.

Left: the query image with two colour bands, the ground-truth class of each
of the 40 rays (top band) and the class the semantic ray network predicts
(bottom band), one band segment per image column group, left to right.
Right: the semantic floor plan around the pose (wall black, window blue, door
red, free white) with the 40 rays drawn to their ground-truth depth and
coloured by their ground-truth class. One row per benchmark, poses chosen
where windows and doors are in view.

    python scripts/plot_semantic_examples.py --gpu 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRL_ROOT = REPO_ROOT.parent / "SemRayLoc"
COL = {0: "#111111", 1: "#1f77ff", 2: "#e6261f", 3: "#ff9ff3"}   # wall, window, door, unknown
NAME = {0: "wall", 1: "window", 2: "door", 3: "unknown"}
EX = [("replica", "replica_f", "apartment_2", 0.375), ("mp3d", "mp3d_f", "EDJbREhghzL_f0", 0.375),
      ("s3d", "s3d", "scene_03250", 0.5959)]
CKPT = {"replica": "outputs/srl/replica/semantic/best.ckpt", "mp3d": "outputs/srl/mp3d/semantic/best.ckpt",
        "s3d": "outputs/srl/s3d/semantic/best.ckpt"}


def center_angs(n, f_w):
    u = np.arange(n)
    return np.flip(np.arctan2(u - u.mean(), n * f_w))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="0")
    p.add_argument("--per-dataset", type=int, default=2)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "figs")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import torch
    sys.path.insert(0, str(SRL_ROOT))
    from modules.semantic.semantic_net_pl import semantic_net_pl

    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, coll, scene, f_w in EX:
        root = Path(f"/root/storage/echoloc_dataset/{ds}")
        sd = root / coll / scene
        poses = np.loadtxt(sd / "poses.txt", ndmin=2)
        D = np.loadtxt(sd / "depth40.txt", ndmin=2)
        S = np.loadtxt(sd / "semantic40.txt", ndmin=2, dtype=int)
        names = sorted(f for f in os.listdir(sd / "rgb") if f.endswith(".png"))
        sem = cv2.imread(str(sd / "semantic_map.png"))[:, :, 0]
        net = semantic_net_pl.load_from_checkpoint(str(REPO_ROOT / CKPT[ds]), map_location="cuda").cuda().eval()
        # poses with both classes in view, the most of them first, spread over the scene
        score = (S == 1).sum(1) * ((S == 2).sum(1) > 0) + (S == 2).sum(1) * ((S == 1).sum(1) > 0)
        order = np.argsort(-score)
        picked = []
        for i in order:
            if all(np.hypot(*(poses[i, :2] - poses[j, :2])) > 2.0 for j in picked):
                picked.append(int(i))
            if len(picked) == a.per_dataset:
                break
        for i in picked:
            img = cv2.cvtColor(cv2.imread(str(sd / "rgb" / names[i])), cv2.COLOR_BGR2RGB)
            x = torch.tensor(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]).cuda()
            m = torch.ones(1, img.shape[0], img.shape[1], dtype=torch.uint8).cuda()
            with torch.no_grad():
                pred = net(x, m)[0].squeeze(0).argmax(-1).cpu().numpy()
            rows.append((ds, scene, i, img, S[i], pred, D[i], poses[i], sem, f_w))
        print(f"[{ds}/{scene}] poses {picked}", flush=True)

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.1 * len(rows)),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    for (ds, scene, i, img, gt, pred, dep, pose, sem, f_w), (ax_img, ax_map) in zip(rows, axes):
        H, W = img.shape[:2]
        ax_img.imshow(img)
        n = len(gt)
        edges = np.linspace(0, W, n + 1)
        for k in range(n):
            ax_img.add_patch(plt.Rectangle((edges[k], 0), edges[k + 1] - edges[k], 0.06 * H, color=COL[gt[k]], lw=0))
            ax_img.add_patch(plt.Rectangle((edges[k], 0.94 * H), edges[k + 1] - edges[k], 0.06 * H, color=COL[pred[k]], lw=0))
        acc = float((gt == pred).mean())
        ax_img.set_title(f"{ds} / {scene} / pose {i}   top band: ground truth, bottom: predicted ({100*acc:.0f}% of rays agree)",
                         fontsize=8)
        ax_img.axis("off")
        # map crop around the pose, rays to their ground-truth depth
        h, w = sem.shape
        r0, c0 = pose[1] / 0.01 + h / 2, pose[0] / 0.01 + w / 2
        rgb = np.ones((h, w, 3))
        for cls, colr in ((1, "#111111"), (2, "#1f77ff"), (3, "#e6261f")):
            rgb[sem == cls] = matplotlib.colors.to_rgb(colr)
        # thicken window/door pixels so they are visible at this scale
        for cls, colr in ((2, "#1f77ff"), (3, "#e6261f")):
            mk = cv2.dilate((sem == cls).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            rgb[mk & (sem > 0)] = matplotlib.colors.to_rgb(colr)
        R = int(6.0 / 0.01)
        r1, r2 = max(0, int(r0) - R), min(h, int(r0) + R)
        c1, c2 = max(0, int(c0) - R), min(w, int(c0) + R)
        ax_map.imshow(rgb[r1:r2, c1:c2], origin="lower", extent=(c1, c2, r1, r2), interpolation="nearest")
        for k, ang in enumerate(center_angs(n, f_w)):
            rng = dep[k] / np.cos(ang) / 0.01
            th = ang + pose[2]
            ax_map.plot([c0, c0 + rng * np.cos(th)], [r0, r0 + rng * np.sin(th)], color=COL[gt[k]], lw=0.9, alpha=0.85)
        ax_map.plot(c0, r0, "o", color="#2ca02c", ms=6)
        ax_map.set_xlim(c1, c2); ax_map.set_ylim(r1, r2); ax_map.set_aspect("equal"); ax_map.axis("off")
        ax_map.set_title("semantic floor plan, 40 ground-truth rays", fontsize=8)
    axes[0, 1].legend(handles=[Patch(color=COL[k], label=NAME[k]) for k in (0, 1, 2)], loc="upper right",
                      fontsize=7, frameon=True)
    fig.tight_layout()
    out = a.out_dir / "semantic_rays_examples.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)

    # the three semantic floor plans whole, one per benchmark
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (ds, coll, scene, _) in zip(axes, EX):
        sem = cv2.imread(f"/root/storage/echoloc_dataset/{ds}/{coll}/{scene}/semantic_map.png")[:, :, 0]
        rgb = np.ones(sem.shape + (3,))
        rgb[sem == 1] = matplotlib.colors.to_rgb("#111111")
        for cls, colr in ((2, "#1f77ff"), (3, "#e6261f")):
            mk = cv2.dilate((sem == cls).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            rgb[mk & (sem > 0)] = matplotlib.colors.to_rgb(colr)
        ys, xs = np.nonzero(sem == 0)
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_xlim(xs.min() - 20, xs.max() + 20); ax.set_ylim(ys.min() - 20, ys.max() + 20)
        cnt = np.bincount(sem.ravel(), minlength=4)
        ax.set_title(f"{ds} / {scene}\nwall {cnt[1]}  window {cnt[2]}  door {cnt[3]} px", fontsize=8)
        ax.axis("off")
    axes[0].legend(handles=[Patch(color=c, label=l) for c, l in (("#111111", "wall"), ("#1f77ff", "window"), ("#e6261f", "door"))],
                   loc="lower left", fontsize=7)
    fig.tight_layout()
    out = a.out_dir / "semantic_maps.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
