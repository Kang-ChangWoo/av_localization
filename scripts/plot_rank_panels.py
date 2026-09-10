#!/usr/bin/env python3
"""Vision, sound, and the fusion as three comparable rank maps, per pose.

The earlier figures failed at the one thing they were for. Drawn over a whole
floor with a per-panel colour scale, the three fields look identical and the
markers land on top of each other, so neither the change of belief nor the
change of decision is visible. Three fixes, all of them about legibility rather
than about the method:

**One scale for all three panels.** Each field is a rank percentile over the
valid cells, drawn with the same colour map between 0.9 and 1.0. A cell that is
bright in one panel and dark in the next has genuinely changed standing, which
is not something the reader could previously check.

**Zoom to the cells in contention.** The panel is cropped to the bounding box of
the visual pick, the fused pick and the truth. A 3 m disagreement inside a 25 m
apartment is otherwise a dozen pixels.

**Say the numbers.** Each panel prints the rank it assigns to the visual pick
and to the fused pick. Those six numbers are the entire decision: if sound moved
the answer, it is because the pair flipped between the first and second panel.

The arrow on the third panel is drawn only when the fused pick differs from the
visual pick, so a row with no arrow is a row where sound changed nothing.

    /opt/conda/envs/unloc/bin/python scripts/plot_rank_panels.py \
        --checkpoint <replica ckpt> --weight 0.5 --mode fixes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
UNLOC_ROOT = REPO_ROOT.parent / "UnLoc"
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_g")
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--checkpoint", default=str(UNLOC_ROOT / "logs" / "unloc_gibson_vitl.ckpt"))
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--weight", type=float, default=0.5)
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--fix-rows", type=int, default=5)
    p.add_argument("--break-rows", type=int, default=3)
    p.add_argument("--vmin", type=float, default=0.90, help="rank floor of the colour scale")
    p.add_argument("--orn-slice", type=int, default=36)
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def rank_norm(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.shape[0], dtype=np.float64)
    r[np.argsort(x)] = np.arange(x.shape[0])
    return (r + 0.5) / x.shape[0]


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import tqdm

    sys.path.insert(0, str(UNLOC_ROOT))
    from modules.depth_net_pl import UnLocDepthModule
    from utils.localization_utils import (
        get_ray_from_depth_uncertainty, localize_uncertainty,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cwd = os.getcwd()
    os.chdir(UNLOC_ROOT)
    try:
        net = UnLocDepthModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint, strict=False).to(device).eval()
    finally:
        os.chdir(cwd)

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, candidate_features, observation_feature, observation_rate, score,
    )

    def occ_in_grid(occ, pg):
        s = pg.cells_per_map_pixel
        r, c = int(round(pg.height * s)), int(round(pg.width * s))
        crop = occ[pg.top: pg.top + r, pg.left: pg.left + c]
        if crop.shape != (r, c):
            pad = np.zeros((r, c), dtype=occ.dtype)
            pad[: crop.shape[0], : crop.shape[1]] = crop
            crop = pad
        return cv2.resize(crop, (pg.width, pg.height), interpolation=cv2.INTER_AREA)

    root = Path(args.dataset_root)
    samples = []
    for scene in args.scenes:
        g = REPO_ROOT / args.grid_dir / f"{scene}.npz"
        if not g.exists():
            continue
        sr = int(json.loads(str(np.load(g)["config"]))["sample_rate"])
        cfg = GridScoreConfig(feature="envelope", window_ms=args.window_ms,
                              sample_rate_hz=sr,
                              direct_guard_samples=int(round(sr * 2 / 1000)),
                              usable_samples=int(round(sr * 128 / 1000)))
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ_full = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ_full.shape)
        occ = occ_in_grid(occ_full, pg)
        mask = valid_pose_mask(occ_full, pg)
        rows, cols = np.nonzero(mask)
        dt = torch.tensor(desdf["desdf"], device=device)
        cf, present = candidate_features(g, rows, cols, mask.shape, cfg)

        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rd = root / "rir" / args.collection / args.condition / scene
        ds_dirs = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
        picks = np.linspace(0, len(ds_dirs) - 1, min(args.n_poses, len(ds_dirs))).astype(int)

        for k in tqdm.tqdm(picks, desc=scene, leave=False):
            d = ds_dirs[int(k)]
            i = int(d.split("_")[1])
            f = rd / d / "rir.npy"
            img_p = root / args.collection / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"
            if not f.exists() or not img_p.exists() or i >= len(poses):
                continue
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR).astype(np.float32)
            rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            x = torch.tensor(np.transpose(x, (2, 0, 1))[None],
                             dtype=torch.float32, device=device)
            with torch.no_grad():
                loc, sc = net.encoder(x, None)[:2]
            pr, ps = get_ray_from_depth_uncertainty(
                loc.squeeze(0).cpu().numpy(), sc.squeeze(0).cpu().numpy())
            _, pdist, _, _ = localize_uncertainty(
                dt, torch.tensor(pr, device=device), torch.tensor(ps, device=device),
                return_np=False, orn_slice=args.orn_slice)
            vis = np.asarray(pdist.cpu(), dtype=np.float64)[rows, cols]
            ac = score(cf, observation_feature(np.load(f), cfg, observation_rate(f)),
                       present, cfg)
            rv, ra = rank_norm(vis), rank_norm(ac)
            rf = rank_norm(np.log(rv) + args.weight * np.log(ra))

            gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
            dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
            vb, fb = int(vis.argmax()), int(rf.argmax())

            def onto(v):
                a = np.full(mask.shape, np.nan)
                a[rows, cols] = v
                return a

            samples.append(dict(
                scene=scene, occ=occ, rgb=rgb, res=pg.grid_resolution_m,
                gt=(float(gx), float(gy)),
                v_xy=(int(cols[vb]), int(rows[vb])), f_xy=(int(cols[fb]), int(rows[fb])),
                e_vis=float(dist[vb]), e_fu=float(dist[fb]),
                fields=[onto(rv), onto(ra), onto(rf)],
                # the six numbers that are the decision
                at=[(float(rv[vb]), float(rv[fb])), (float(ra[vb]), float(ra[fb])),
                    (float(rf[vb]), float(rf[fb]))]))

    if not samples:
        print("nothing to plot")
        return 1

    fixed = sorted([s for s in samples if s["e_vis"] >= 1.0 and s["e_fu"] < 1.0],
                   key=lambda s: -s["e_vis"])[: args.fix_rows]
    broke = sorted([s for s in samples if s["e_vis"] < 1.0 and s["e_fu"] >= 1.0],
                   key=lambda s: -s["e_fu"])[: args.break_rows]
    show = [("SOUND FIXES IT", s) for s in fixed] + [("SOUND BREAKS IT", s) for s in broke]
    print(f"[plot] {len(samples)} poses; {len(fixed)} repairs and {len(broke)} regressions drawn")

    titles = ["vision: rank of each cell",
              f"sound: rank of each cell",
              f"fused: log(vision) + {args.weight:g} log(sound)"]
    fig, axes = plt.subplots(len(show), 4, figsize=(4.4 * 4, 4.0 * len(show)))
    axes = np.atleast_2d(axes)
    im = None
    for r, (tag, s) in enumerate(show):
        xs = [s["v_xy"][0], s["f_xy"][0], s["gt"][0]]
        ys = [s["v_xy"][1], s["f_xy"][1], s["gt"][1]]
        pad = max(10, int(0.30 * max(max(xs) - min(xs), max(ys) - min(ys))))
        box = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

        axes[r, 0].imshow(s["rgb"]); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"{tag}\n{s['scene']}\n"
                              f"{s['e_vis']:.1f} m  ->  {s['e_fu']:.1f} m",
                              fontsize=10, weight="bold")
        if r == 0:
            axes[r, 0].set_title("camera", fontsize=11)

        for c in range(3):
            ax = axes[r, c + 1]
            ax.imshow(s["occ"], cmap="gray", vmin=0, vmax=255, alpha=0.5, zorder=-1)
            im = ax.imshow(np.ma.masked_invalid(s["fields"][c]), cmap="viridis",
                           vmin=args.vmin, vmax=1.0, alpha=0.95)
            # the two candidates, always both, always on every panel
            ax.scatter(*s["v_xy"], marker="X", s=340, c="#ff1744",
                       edgecolors="k", linewidths=1.4, zorder=6)
            ax.scatter(*s["f_xy"], marker="o", s=340, facecolors="none",
                       edgecolors="#00e5ff", linewidths=3.2, zorder=6)
            ax.scatter(*s["gt"], marker="*", s=430, c="#00e676",
                       edgecolors="k", linewidths=1.0, zorder=7)
            if c == 2 and s["v_xy"] != s["f_xy"]:
                ax.annotate("", xy=s["f_xy"], xytext=s["v_xy"], zorder=8,
                            arrowprops=dict(arrowstyle="-|>", lw=2.6, color="w",
                                            shrinkA=13, shrinkB=13,
                                            path_effects=None))
            a, b = s["at"][c]
            ax.text(0.03, 0.97, f"rank at X  {a:.4f}\nrank at O  {b:.4f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=10,
                    family="monospace", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.4", alpha=0.9))
            ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)

    if im is not None:
        cb = fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), fraction=0.014, pad=0.012)
        cb.set_label(f"rank percentile among valid cells "
                     f"(scale starts at {args.vmin:g})", fontsize=9)
    fig.suptitle(
        "The same three cells under vision, under sound, and under the fusion. "
        "Red X is vision's pick,\ncyan circle the fused pick, green star the truth; "
        "the arrow is the change of answer.\nAll three panels share one rank scale, "
        "so a cell that brightens between panels really did gain standing.",
        fontsize=13)
    p = args.out_dir / f"unloc_rank_panels{args.tag}.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
