#!/usr/bin/env python3
"""Contact sheets of every floorplan, for auditing the geometry by eye.

Four things have to agree per scene and none of them is visible in a number:
the wall mask the DESDF was ray-cast from, the cells the pose grid accepts as
navigable, the cells a candidate grid actually holds an impulse response for,
and where the recorded poses are. A disagreement in any of them surfaces later
as a recall figure nobody can explain.

The panels are drawn so that a correct scene is quiet and a broken one is loud.
Navigable area is a flat tint, walls are the paper colour, and strong colour is
reserved for the two things that must never appear: a navigable cell with no
rendered candidate, and a pose outside the navigable area. A page of correct
scenes therefore carries no red at all, so an audit of 250 rooms is a glance
rather than a reading.

    python scripts/plot_floorplan_grid.py --dataset s3d --split test --cols 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

DATASETS = {
    "replica": dict(root="/root/storage/echoloc_dataset/replica", coll="replica_f",
                    grid="acoustic_grid_v2"),
    "mp3d": dict(root="/root/storage/echoloc_dataset/mp3d", coll="mp3d_f",
                 grid="acoustic_grid_mp3d"),
    "s3d": dict(root="/root/storage/echoloc_dataset/s3d", coll="s3d",
                grid="acoustic_grid_s3d"),
}

# One tint for area, paper for structure, one accent for poses, and red left
# unused unless something is actually wrong.
PAPER = (1.0, 1.0, 1.0)
WALL = (0.05, 0.05, 0.05)         # the wall itself, a thin stroke around the
                                  # free space rather than the whole exterior
BAND = (0.945, 0.949, 0.957)      # free but inside the clearance band, so no
                                  # candidate cell is placed there
NAV = (0.925, 0.949, 0.988)       # navigable, no candidate rendered
REND = (0.831, 0.894, 0.976)       # navigable and rendered
MISSING = (0.937, 0.267, 0.267)   # navigable but never rendered
WALL_PX = 9                       # drawn wall thickness, in map pixels (~9 cm)
POSE = "#1d4ed8"
INK = "#111827"
GREY = "#6b7280"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="s3d", choices=sorted(DATASETS))
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test", "rendered", "all"])
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--rows", type=int, default=10, help="rows per page")
    p.add_argument("--pages", type=int, default=0, help="0 for every page")
    p.add_argument("--panel-in", type=float, default=2.3)
    p.add_argument("--ref-images", type=int, default=1,
                   help="inset one RGB frame per panel; 0 to disable")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "figs")
    return p.parse_args()


def load_scene(root: Path, coll: str, scene: str, grid_dir: Path, have_grid: set):
    """The panel image and the counts the audit turns on, or None if unreadable."""
    import cv2
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.floorplan.pose_grid import MAP_RESOLUTION_M

    mp = root / coll / scene / "map.png"
    dp = root / "desdf" / scene / "desdf.npy"
    if not mp.exists():
        return None
    occ = cv2.imread(str(mp))[:, :, 0]
    # The DESDF is built only for the test split, but the wall mask, the
    # clearance band and the poses exist for every scene and are most of what
    # the audit checks. Without a DESDF there is no 0.1 m lattice and no
    # candidate grid, so those two layers are absent rather than the whole panel
    # being blank: a survey that skips 85% of the rooms is not a survey.
    d = np.load(dp, allow_pickle=True).item() if dp.exists() else None
    pg = PoseGrid.from_desdf(d, occ.shape) if d is not None else None
    nav = valid_pose_mask(occ, pg) if pg is not None else None

    rend = np.zeros_like(nav) if nav is not None else None
    if nav is not None and scene in have_grid:
        b = np.load(grid_dir / f"{scene}.npz", allow_pickle=False)
        gi = np.unique(b["index"][:, :2], axis=0)
        del b
        ok = (gi[:, 0] < nav.shape[0]) & (gi[:, 1] < nav.shape[1])
        rend[gi[ok, 0], gi[ok, 1]] = True

    # The occupancy map marks everything that is not free as occupied, which
    # includes the whole exterior, so painting it directly fills the page. A
    # floorplan reads as a thin wall around white rooms, and that wall is the
    # boundary: dilate the free space and subtract it.
    from track1_core.floorplan.valid_mask import clearance_map_m, DEFAULT_CLEARANCE_M
    free = occ == occ.max()
    mres = pg.map_resolution_m if pg is not None else MAP_RESOLUTION_M
    clear = clearance_map_m(occ, mres)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WALL_PX + 1,) * 2)
    wall = (cv2.dilate(free.astype(np.uint8), ker) > 0) & ~free

    img = np.empty(occ.shape + (3,), np.float32)
    img[...] = np.array(PAPER, np.float32)
    img[free & (clear < DEFAULT_CLEARANCE_M)] = BAND
    img[free & (clear >= DEFAULT_CLEARANCE_M)] = NAV

    if nav is not None:
        k = max(int(round(pg.cells_per_map_pixel)), 1)
        t, l = int(d["t"]), int(d["l"])
        layers = (((nav & rend, REND), (nav & ~rend, MISSING))
                  if scene in have_grid else ((nav, REND),))
        for mask, colour in layers:
            up = np.kron(mask, np.ones((k, k), bool))
            m = np.zeros(occ.shape, bool)
            h = min(up.shape[0], occ.shape[0] - t)
            w = min(up.shape[1], occ.shape[1] - l)
            m[t:t + h, l:l + w] = up[:h, :w]
            img[m] = colour
    img[wall] = WALL

    pp = root / coll / scene / "poses.txt"
    P = (np.array([[float(v) for v in ln.split()] for ln in open(pp) if ln.strip()])
         if pp.exists() else np.zeros((0, 3)))
    # world metres to map pixels needs only the map, so poses are drawn whether
    # or not a DESDF exists
    if len(P):
        centre = np.array([occ.shape[1] / 2.0, occ.shape[0] / 2.0])
        gp = P[:, :2] / mres + centre
        g = pg.metric_to_grid(P[:, :2]) if pg is not None else np.zeros((0, 2))
    else:
        gp = np.zeros((0, 2)); g = np.zeros((0, 2))
    # How far a pose sits from the nearest candidate cell, which is the quantity
    # that matters rather than whether its rounded cell happens to be a wall.
    # Replica and Matterport3D keep this under a tenth of a metre; Structured3D
    # places cameras close to walls and reaches 0.24 m, which is harmless at the
    # 1 m threshold and fatal at 0.1 m. Only a gap large against 1 m is an error.
    gap95 = 0.0
    if len(g) and nav is not None and nav.any():
        from scipy.spatial import cKDTree
        rr, cc = np.nonzero(nav)
        dd, _ = cKDTree(np.stack([cc, rr], axis=-1)).query(g)
        gap95 = float(np.percentile(dd, 95) * pg.grid_resolution_m)

    # crop to the building, so a small room is not lost in the bounding box
    ys, xs = np.nonzero(free | wall)
    if len(ys):
        pad = 12
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, img.shape[0])
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, img.shape[1])
        img = img[y0:y1, x0:x1]
        if len(gp):
            gp = gp - np.array([x0, y0])
    # Reserve a white strip above the plan for the name and the counts. Drawing
    # them over the plan with a translucent box covered the top rooms in every
    # narrow panel, and a Matplotlib title would double the row spacing, which
    # is the whole cost of a 700-scene contact sheet.
    strip = max(int(round(img.shape[0] * 0.22)), 34)
    img = np.concatenate([np.ones((strip,) + img.shape[1:], np.float32), img], axis=0)
    if len(gp):
        gp = gp + np.array([0, strip])
    return dict(img=img, poses=gp,
                n_nav=int(nav.sum()) if nav is not None else 0,
                n_rend=int((nav & rend).sum()) if nav is not None else 0,
                n_pose=len(P), gap95=gap95, has_desdf=nav is not None,
                has_grid=scene in have_grid)


def main() -> int:
    args = parse_args()
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    spec = DATASETS[args.dataset]
    root = Path(spec["root"])
    grid_dir = REPO_ROOT / "outputs" / spec["grid"]
    rendered = sorted(p.stem for p in grid_dir.glob("*.npz") if "shard" not in p.name) \
        if grid_dir.is_dir() else []
    have_grid = set(rendered)

    if args.split == "rendered":
        scenes = rendered
    else:
        sp = yaml.safe_load((root / spec["coll"] / "split.yaml").read_text())
        scenes = ([s for k in ("train", "val", "test") for s in sp.get(k, [])]
                  if args.split == "all" else list(sp[args.split]))
    print(f"[plot] {args.dataset} {args.split}: {len(scenes)} scenes, "
          f"{len(have_grid)} with a candidate grid")

    per_page = args.cols * args.rows
    n_pages = int(np.ceil(len(scenes) / per_page))
    if args.pages:
        n_pages = min(n_pages, args.pages)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flagged: list[str] = []

    for page in range(n_pages):
        chunk = scenes[page * per_page:(page + 1) * per_page]
        rows = int(np.ceil(len(chunk) / args.cols))
        fig, axes = plt.subplots(rows, args.cols,
                                 figsize=(args.cols * args.panel_in,
                                          rows * args.panel_in + 0.6),
                                 facecolor="white")
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.set_axis_off()
        tot = dict(nav=0, rend=0, pose=0)
        for ax, scene in zip(axes, chunk):
            s = load_scene(root, spec["coll"], scene, grid_dir, have_grid)
            if s is None:
                ax.text(0.5, 0.5, f"{scene}\nmissing map or desdf", ha="center",
                        va="center", fontsize=8, color=MISSING, transform=ax.transAxes)
                flagged.append(f"{scene}: missing map or desdf")
                continue
            ax.imshow(s["img"], interpolation="nearest")
            if len(s["poses"]):
                ax.scatter(s["poses"][:, 0], s["poses"][:, 1], s=4.0, c=POSE,
                           linewidths=0, alpha=0.95, zorder=5)
            miss = s["n_nav"] - s["n_rend"] if s["has_grid"] else 0
            bad = (miss > 0) or (s["gap95"] > 0.5)
            if bad:
                flagged.append(f"{scene}: {miss} navigable cells unrendered, "
                               f"pose-to-grid gap {s['gap95']:.2f} m at the 95th pct")
            # the name goes inside the panel: a title row would double the
            # vertical spacing, which is the whole cost of a 250-scene sheet
            ax.text(0.015, 0.995, scene, transform=ax.transAxes, ha="left", va="top",
                    fontsize=11, fontweight="bold",
                    color=(MISSING if bad else INK))
            note = f"#{page * per_page + list(chunk).index(scene) + 1}"
            if miss:
                note += f"   {miss} UNRENDERED"
            if s["gap95"] > 0.5:
                note += f"   gap {s['gap95']*100:.0f} cm"
            ax.text(0.015, 0.918, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, color=(MISSING if bad else GREY))
            tot["nav"] += s["n_nav"]; tot["rend"] += s["n_rend"]; tot["pose"] += s["n_pose"]

            if args.ref_images:
                rgb_dir = root / spec["coll"] / scene / "rgb"
                names = sorted(os.listdir(rgb_dir)) if rgb_dir.is_dir() else []
                if names:
                    im = cv2.imread(str(rgb_dir / names[len(names) // 2]))
                    if im is not None:
                        ih, iw = s["img"].shape[:2]
                        fw = 0.30
                        # the inset is sized in axes fractions, so its aspect has
                        # to be corrected by the panel's own aspect or the frame
                        # is stretched differently in every panel
                        fh = fw * (im.shape[0] / im.shape[1]) * (iw / ih)
                        ia = ax.inset_axes([1 - fw - 0.015, 0.015, fw, fh])
                        ia.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                        ia.set_xticks([]); ia.set_yticks([])
                        for sp_ in ia.spines.values():
                            sp_.set_edgecolor(INK); sp_.set_linewidth(0.6)

        fig.suptitle(
            f"{args.dataset}   {args.split}   page {page+1}/{n_pages}       "
            f"{tot['nav']} navigable cells, {tot['rend']} rendered, {tot['pose']} poses"
            f"\npale blue area = candidate cell    grey fringe = free but inside the "
            f"25 cm clearance band    black = wall    blue dot = recorded pose",
            fontsize=8, color=INK, y=0.998)
        fig.subplots_adjust(left=0.004, right=0.996, top=0.948, bottom=0.004,
                            wspace=0.015, hspace=0.015)
        out = args.out_dir / f"plans_{args.dataset}_{args.split}_p{page+1:02d}.png"
        fig.savefig(out, dpi=170, facecolor="white")
        plt.close(fig)
        print(f"[plot] wrote {out}")

    print(f"[audit] {len(flagged)} scene(s) flagged")
    for f in flagged[:40]:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
