#!/usr/bin/env python3
"""What the acoustic term does to UnLoc's likelihood map, drawn pose by pose.

Every claim in this project so far is a recall number, and a recall number
cannot say *how* sound helps. Three mechanisms would all produce the same
column: sound could sharpen a single broad mode, it could kill one of several
competing modes, or it could drag the answer to a different room. They call for
different papers, and they are trivially distinguishable by looking.

Each row is one pose and four panels on the shared pose grid:

    UnLoc alone      the uncertainty-weighted likelihood, max-pooled over the
                     36 heading bins, which is what the recall number reads
    acoustic alone   the candidate grid scored against this recording
    fused            the rank-normalised log-space sum actually used
    difference       fused minus visual

All four are rank percentiles over the valid cells, so they share one scale, and
all four are masked to the cells that reach the top decile under *any* of the
three scores. That mask is what makes the panels comparable: an unmasked field
is flat over most of the floor and renders as a uniform wash, and each panel
would otherwise show a different set of cells.

Poses are drawn spanning the visual margin, the same quantity the gate reads, so
the top rows are cases where vision is unsure and the bottom rows cases where it
is confident and the gate is supposed to do nothing.

    /opt/conda/envs/unloc/bin/python scripts/plot_unloc_audio_maps.py
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
    p.add_argument("--weight", type=float, default=2.0, help="fusion weight, held-out value")
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--n-poses", type=int, default=60, help="scored per scene")
    p.add_argument("--n-rows", type=int, default=10, help="rows in the detail figure")
    p.add_argument("--fix-rows", type=int, default=5, help="repairs shown in figure 2")
    p.add_argument("--break-rows", type=int, default=3, help="regressions shown in figure 2")
    p.add_argument("--tag", default="", help="suffix for the output filenames")
    p.add_argument("--orn-slice", type=int, default=36)
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
    from matplotlib.colors import LogNorm, TwoSlopeNorm

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
        """``map.png`` resampled into the pose grid's frame.

        The grid is a crop of the map at ``(left, top)``, ten times coarser, so
        drawing the map at native resolution under a grid-space field lines up
        on neither scale nor offset.
        """
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
        ys, xs = np.nonzero(mask)
        pad = 6
        extent = (max(xs.min() - pad, 0), min(xs.max() + pad, pg.width - 1),
                  max(ys.min() - pad, 0), min(ys.max() + pad, pg.height - 1))
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
                loc, scale, _, _ = net.encoder(x, None)
            pr, ps = get_ray_from_depth_uncertainty(
                loc.squeeze(0).cpu().numpy(), scale.squeeze(0).cpu().numpy())
            _, pdist, _, _ = localize_uncertainty(
                dt, torch.tensor(pr, device=device), torch.tensor(ps, device=device),
                return_np=False, orn_slice=args.orn_slice)
            vis = np.asarray(pdist.cpu(), dtype=np.float64)[rows, cols]
            s = score(cf, observation_feature(np.load(f), cfg, observation_rate(f)),
                      present, cfg)

            rv, ra = rank_norm(vis), rank_norm(s)
            lv, la = np.log(rv), np.log(ra)
            fu = lv + args.weight * la
            rf = rank_norm(fu)
            # Every panel is masked to the cells that carry belief under any of
            # the three scores. Without this the top decile of one field is drawn
            # against the flat majority of the others and the panels cannot be
            # compared; with it, all four show the same cells and the reader can
            # follow one place across the row.
            live = np.maximum.reduce([rv, ra, rf]) > 0.90
            gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
            dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
            best = int(vis.argmax())
            far = np.hypot(cols - cols[best], rows - rows[best]) \
                * pg.grid_resolution_m > args.mode_sep_m
            margin = float(np.log((vis[best] + 1e-300)
                                  / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))

            # every field is laid back onto the grid so all four panels share axes
            def onto(v, fill=np.nan):
                a = np.full(mask.shape, fill, dtype=np.float64)
                a[rows, cols] = v
                return a

            samples.append(dict(
                scene=scene, occ=occ, extent=extent, rgb=rgb, margin=margin,
                gt=(gx, gy), res=pg.grid_resolution_m,
                vis_pick=(cols[best], rows[best]),
                ac_pick=(cols[int(s.argmax())], rows[int(s.argmax())]),
                fu_pick=(cols[int(fu.argmax())], rows[int(fu.argmax())]),
                e_vis=float(dist[best]), e_ac=float(dist[int(s.argmax())]),
                e_fu=float(dist[int(fu.argmax())]),
                v_vis=onto(np.where(live, rv, np.nan)),
                v_ac=onto(np.where(live, ra, np.nan)),
                v_fu=onto(np.where(live, rf, np.nan)),
                v_d=onto(np.where(live, rf - rv, np.nan))))

    if not samples:
        print("nothing to plot")
        return 1
    tau = float(np.quantile([s["margin"] for s in samples], args.gate_quantile))
    for s in samples:
        s["gate_on"] = s["margin"] < tau
        s["pick"] = s["fu_pick"] if s["gate_on"] else s["vis_pick"]
        s["e_gate"] = s["e_fu"] if s["gate_on"] else s["e_vis"]
    print(f"[plot] {len(samples)} poses, gate fires below margin {tau:.3f} "
          f"({100*np.mean([s['gate_on'] for s in samples]):.0f}% of poses)")

    def heat(ax, v, log=True):
        """A likelihood that spans orders of magnitude and is nearly flat over
        most of the floor renders as one wash under a fixed floor, so the floor
        is a high quantile of the cells that carry any belief at all."""
        nz = v[np.isfinite(v) & (v > 0)]
        if nz.size == 0:
            return
        # the fields arrive already masked to the cells that carry belief, so
        # the colour scale spans those cells rather than the whole floor
        lo, hi = float(np.nanmin(nz)), float(np.nanmax(v))
        if not (hi > lo > 0):
            lo = hi * 0.999
        m = np.ma.masked_where(~np.isfinite(v) | (v <= lo), v)
        ax.imshow(m, cmap="magma_r", norm=LogNorm(vmin=lo, vmax=hi) if log else None,
                  alpha=0.92)

    def diff(ax, v):
        if not np.isfinite(v).any():
            return
        lim = float(np.nanmax(np.abs(v))) or 1.0
        ax.imshow(np.ma.masked_invalid(v), cmap="coolwarm",
                  norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), alpha=0.92)

    def finish(ax, s, marks=True):
        ax.imshow(s["occ"], cmap="gray", vmin=0, vmax=255, alpha=0.35, zorder=-1)
        if marks:
            ax.scatter(*s["gt"], marker="*", s=150, c="#00c853",
                       edgecolors="k", linewidths=0.5, zorder=5)
            ax.scatter(*s["vis_pick"], marker="x", s=70, c="#d50000",
                       linewidths=2.0, zorder=5)
            ax.scatter(*s["fu_pick"], marker="o", s=70, facecolors="none",
                       edgecolors="#2962ff", linewidths=1.8, zorder=5)
        x0, x1, y0, y1 = s["extent"]
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks([]); ax.set_yticks([])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- figure 1: the four fields, poses spanning the margin ---------------
    ordered = sorted(samples, key=lambda s: s["margin"])
    sel = [ordered[i] for i in np.unique(
        np.linspace(0, len(ordered) - 1, args.n_rows).astype(int))]
    cols_t = ["camera", "UnLoc alone", "acoustic alone", "fused", "fused - UnLoc"]
    fig, axes = plt.subplots(len(sel), 5, figsize=(3.0 * 5, 2.9 * len(sel)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(sel):
        axes[r, 0].imshow(s["rgb"]); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"{s['scene']}\nmargin {s['margin']:.2f}"
                              f"\ngate {'ON' if s['gate_on'] else 'off'}", fontsize=8)
        for c in (1, 2, 3):
            heat(axes[r, c], s[("v_vis", "v_ac", "v_fu")[c - 1]], log=False)
            finish(axes[r, c], s)
        diff(axes[r, 4], s["v_d"]); finish(axes[r, 4], s, marks=False)
        for c, e in ((1, s["e_vis"]), (2, s["e_ac"]), (3, s["e_fu"])):
            axes[r, c].set_title((cols_t[c] + "\n" if r == 0 else "")
                                 + f"err {e:.2f} m", fontsize=8)
        if r == 0:
            axes[r, 0].set_title(cols_t[0], fontsize=9)
            axes[r, 4].set_title(cols_t[4], fontsize=9)
    fig.suptitle("UnLoc likelihood, acoustic score, and the fusion, on the shared pose grid\n"
                 "green star truth, red cross UnLoc's pick, blue circle fused pick; "
                 "right column: blue = belief sound removed, red = belief sound added",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p1 = args.out_dir / f"unloc_audio_likelihood_maps{args.tag}.png"
    fig.savefig(p1, dpi=135); plt.close(fig)

    # ---- figure 2: the cases the fusion actually changes --------------------
    fixed = sorted([s for s in samples if s["e_vis"] >= 1.0 and s["e_fu"] < 1.0],
                   key=lambda s: -s["e_vis"])
    broke = sorted([s for s in samples if s["e_vis"] < 1.0 and s["e_fu"] >= 1.0],
                   key=lambda s: -s["e_fu"])
    print(f"[plot] sound fixes {len(fixed)} poses, breaks {len(broke)}, "
          f"of {len(samples)}")
    # membership by identity: these dicts hold arrays, so `in` would compare
    # them element-wise and raise
    show = ([("FIXED", s) for s in fixed[: args.fix_rows]]
            + [("BROKEN", s) for s in broke[: args.break_rows]])
    if show:
        fig, axes = plt.subplots(len(show), 5, figsize=(3.0 * 5, 2.9 * len(show)))
        axes = np.atleast_2d(axes)
        for r, (tag, s) in enumerate(show):
            axes[r, 0].imshow(s["rgb"]); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_ylabel(f"{tag}\n{s['scene']}\n"
                                  f"{s['e_vis']:.1f} m -> {s['e_fu']:.1f} m", fontsize=8)
            for c in (1, 2, 3):
                heat(axes[r, c], s[("v_vis", "v_ac", "v_fu")[c - 1]], log=False)
                finish(axes[r, c], s)
            diff(axes[r, 4], s["v_d"]); finish(axes[r, 4], s, marks=False)
            if r == 0:
                for c, t in enumerate(cols_t):
                    axes[r, c].set_title(t, fontsize=9)
        fig.suptitle(f"Where the acoustic term changes the answer: the "
                     f"{len(fixed[:args.fix_rows])} biggest repairs and the "
                     f"{len(broke[:args.break_rows])} worst regressions "
                     f"(sound repairs {len(fixed)} poses and breaks {len(broke)} of "
                     f"{len(samples)})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        p2 = args.out_dir / f"unloc_audio_fix_and_break{args.tag}.png"
        fig.savefig(p2, dpi=135); plt.close(fig)
        print(f"wrote {p2}")
    print(f"wrote {p1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
