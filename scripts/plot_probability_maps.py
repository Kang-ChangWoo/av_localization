#!/usr/bin/env python3
"""Draw what the two modalities actually believe, over many poses.

Every number reported so far is a recall rate, which says how often the answer
was right but nothing about the shape of the belief that produced it. These
figures show the shape: where the visual posterior puts its mass, where the
acoustic likelihood puts its own, and whether the two disagree in a way that a
re-rank can exploit.

Two figures:

  grid    one panel per pose, many poses, sorted by visual margin so the
          ambiguous cases sit together at the top-left. Each panel carries the
          visual posterior as a heat map, the shortlist as dots shaded by
          acoustic score, and the three answers (truth, vision, gated).
  detail  a few poses at full size, with the visual posterior, the acoustic
          likelihood over the whole candidate grid, and the gated result side
          by side, so the interaction is legible rather than inferred.

    python scripts/plot_probability_maps.py --n-panels 24 --scene office_4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

STFT_BANDS = [(0, 500), (500, 1500), (1500, 4000)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--run-name", default="echoloc_mono_fg")
    p.add_argument("--net", default="mono", choices=["mono", "mv", "comp"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--nfft", type=int, default=64)
    p.add_argument("--hop", type=int, default=16)
    p.add_argument("--n-poses", type=int, default=120, help="poses scored per scene")
    p.add_argument("--n-panels", type=int, default=24, help="panels in the grid figure")
    p.add_argument("--n-detail", type=int, default=5)
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
    from matplotlib.colors import LogNorm
    import torch
    import tqdm

    from scripts.margin_gated_fusion import band_envelope, featurize_band
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.models import CompDepthModule, MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    CLS = {"mono": MonoDepthModule, "mv": MVDepthModule, "comp": CompDepthModule}

    scenes = args.scenes or sorted(
        s for s in os.listdir(root / "desdf")
        if (REPO_ROOT / "outputs" / "acoustic_grid" / f"{s}.npz").exists())

    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    kw = dict(mono_ckpt=None, mv_ckpt=None) if args.net == "comp" else {}
    model = CLS[args.net].load_from_checkpoint(str(ckpt), **kw).to(device).eval()

    def describe(rir):
        return band_envelope(rir, args.guard_samples, args.usable_samples,
                             args.nfft, args.hop, args.sample_rate, STFT_BANDS)

    def occ_in_grid(occ, pg):
        """``map.png`` resampled into the pose grid's own frame.

        The grid is a crop of the map at ``(left, top)``, ten times coarser
        (0.1 m per cell against 0.01 m per pixel). Drawing the map at its native
        resolution underneath a grid-space posterior lines up on neither scale
        nor offset, so it is brought into the grid frame once, here.
        """
        s = pg.cells_per_map_pixel
        r, c = int(round(pg.height * s)), int(round(pg.width * s))
        crop = occ[pg.top: pg.top + r, pg.left: pg.left + c]
        if crop.shape != (r, c):
            pad = np.zeros((r, c), dtype=occ.dtype)
            pad[: crop.shape[0], : crop.shape[1]] = crop
            crop = pad
        return cv2.resize(crop, (pg.width, pg.height), interpolation=cv2.INTER_AREA)

    samples = []
    for scene in scenes:
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ_full = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ_full.shape)
        occ = occ_in_grid(occ_full, pg)
        mask = valid_pose_mask(occ_full, pg)
        # crop every panel to the walkable extent, so the room fills the frame
        ys, xs = np.nonzero(mask)
        pad = 6
        extent_xy = (max(xs.min() - pad, 0), min(xs.max() + pad, pg.width - 1),
                     max(ys.min() - pad, 0), min(ys.max() + pad, pg.height - 1))
        desdf_t = torch.tensor(desdf["desdf"], device=device)

        blob = np.load(REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz")
        cand, index = blob["rir"], blob["index"]
        env = np.stack([describe(c) for c in cand])
        env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
        cand_f = featurize_band(env)

        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene
        dsdir = str(root / args.collection)
        dataset = GridSeqDataset(dsdir, [scene], L=L, depth_dir=dsdir,
                                 depth_suffix="depth40" if args.net == "mono" else "depth160")
        picks = np.linspace(0, len(dataset) - 1, min(args.n_poses, len(dataset))).astype(int)

        for chunk in tqdm.tqdm(picks, desc=scene, leave=False):
            pose_idx = int(chunk) * (L + 1) + L
            f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
            if not f.exists():
                continue
            data = dataset[int(chunk)]
            with torch.no_grad():
                if args.net == "mono":
                    pred, _, _ = model.encoder(
                        torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
                else:
                    b = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                         for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                    b["ref_mask"] = b["src_mask"] = None
                    pred = (model.net(b)["d"] if args.net == "mv"
                            else model.comp_d_net(b)["d_comp"])
            rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                device=device, dtype=torch.float32)
            prob_vol, _, _, _ = localize(desdf_t, rays)
            vol = np.array(prob_vol, dtype=np.float64)
            vol[~mask] = 0.0
            vis_map = vol.sum(-1)            # marginal over orientation
            vis_map /= max(vis_map.sum(), 1e-300)

            flat = vol.reshape(-1)
            order = np.argsort(-flat)[: args.topk]
            r_, c_, o_ = np.unravel_index(order, vol.shape)
            far = np.hypot(c_ - c_[0], r_ - r_[0]) * pg.grid_resolution_m > args.mode_sep_m
            second = flat[order][far].max() if far.any() else 0.0
            margin = float(np.log((flat[order[0]] + 1e-300) / (second + 1e-300)))

            obs_f = featurize_band(describe(np.load(f)))
            nw = min(cand_f.shape[2], obs_f.shape[1])
            ac_all = -np.abs(cand_f[:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))
            ac_map = np.full(vis_map.shape, np.nan)
            ac_map[index[:, 0], index[:, 1]] = ac_all

            lut = -np.ones(vis_map.shape, dtype=int)
            lut[index[:, 0], index[:, 1]] = np.arange(len(index))
            short = lut[r_, c_]
            ac_short = np.where(short >= 0, ac_all[np.clip(short, 0, None)], -np.inf)

            gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
            samples.append(dict(
                scene=scene, occ=occ, extent_xy=extent_xy,
                res=pg.grid_resolution_m, vis_map=vis_map,
                ac_map=ac_map, rows=r_, cols=c_, ac_short=ac_short, margin=margin,
                gt=(gx, gy), vis_pick=(c_[0], r_[0]),
                ac_pick=(c_[int(np.argmax(ac_short))], r_[int(np.argmax(ac_short))])))

    if not samples:
        print("nothing to plot")
        return 1
    margins = np.array([s["margin"] for s in samples])
    tau = float(np.quantile(margins, args.gate_quantile))
    ranks = np.argsort(np.argsort(margins)) / max(len(margins) - 1, 1)
    for s, q in zip(samples, ranks):
        s["margin_q"] = float(q)
    for s in samples:
        s["gated"] = s["ac_pick"] if s["margin"] < tau else s["vis_pick"]
        s["gate_on"] = s["margin"] < tau
        s["e_vis"] = np.hypot(*(np.array(s["vis_pick"]) - s["gt"])) * s["res"]
        s["e_gate"] = np.hypot(*(np.array(s["gated"]) - s["gt"])) * s["res"]
    print(f"[plot] {len(samples)} poses, gate fires below margin {tau:.3f} "
          f"({100*np.mean([s['gate_on'] for s in samples]):.0f}% of poses)")

    def draw_map(ax, s):
        ax.imshow(s["occ"], cmap="gray", vmin=0, vmax=255, alpha=0.4)

    def markers(ax, s, size=70):
        """Draws the three answers, then fixes the view. Called last on every
        panel: imshow and scatter both autoscale, so the crop has to be applied
        after them or it is silently undone."""
        ax.scatter(*s["gt"], marker="*", s=size * 2.2, c="#00c853",
                   edgecolors="k", linewidths=0.5, zorder=5)
        ax.scatter(*s["vis_pick"], marker="x", s=size, c="#d50000",
                   linewidths=2.0, zorder=5)
        ax.scatter(*s["gated"], marker="o", s=size, facecolors="none",
                   edgecolors="#2962ff", linewidths=1.8, zorder=5)
        x0, x1, y0, y1 = s["extent_xy"]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

    def heat(ax, v, alpha=0.9):
        """The posterior spans many orders of magnitude and is nearly flat over
        most of the floor, so a floor at a fixed fraction of the peak renders as
        one uniform wash. Clipping at a high quantile of the occupied cells
        shows the few cells that actually carry the belief."""
        nz = v[v > 0]
        if nz.size == 0:
            return
        lo = float(np.quantile(nz, 0.97))
        hi = float(v.max())
        if not (hi > lo > 0):
            lo, hi = max(hi * 1e-3, 1e-300), max(hi, 1e-299)
        ax.imshow(np.ma.masked_where(v <= lo, v), cmap="magma_r",
                  norm=LogNorm(vmin=lo, vmax=hi), alpha=alpha)

    # ---------------- figure 1: poses spanning the ambiguity range ----------
    ordered = sorted(samples, key=lambda s: s["margin"])
    idx = np.unique(np.linspace(0, len(ordered) - 1, args.n_panels).astype(int))
    sel = [ordered[i] for i in idx]
    ncol = 6
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.75 * nrow))
    for ax, s in zip(np.ravel(axes), sel):
        draw_map(ax, s)
        heat(ax, s["vis_map"])
        a = s["ac_short"]
        fin = np.isfinite(a)
        if fin.any():
            ax.scatter(s["cols"][fin], s["rows"][fin], s=16,
                       c=a[fin], cmap="winter", edgecolors="none", alpha=0.85, zorder=4)
        markers(ax, s, size=45)
        ax.set_title(f"{s['scene'][:12]}  margin pct {100*s['margin_q']:.0f}"
                     f"{'  GATE ON' if s['gate_on'] else ''}\n"
                     f"vis {s['e_vis']:.1f}m  ->  {s['e_gate']:.1f}m",
                     fontsize=7,
                     color=("#1b5e20" if s["e_gate"] < s["e_vis"] - 1e-9 else
                            "#b71c1c" if s["e_gate"] > s["e_vis"] + 1e-9 else "#37474f"))
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.ravel(axes)[len(sel):]:
        ax.axis("off")
    fig.suptitle(
        f"Visual posterior (dark = high, top 3% of cells) with the top-{args.topk} "
        f"shortlist shaded by acoustic score (green = better match)\n"
        f"green star = truth,  red x = vision's answer,  blue circle = after the gate.  "
        f"{len(sel)} poses spanning the margin range, most ambiguous first, "
        f"of {len(samples)} scored.  {args.run_name} ({args.net})",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p1 = out_dir / f"probability_maps_grid_{args.run_name}.png"
    fig.savefig(p1, dpi=135); plt.close(fig)
    print(f"wrote {p1}")

    # ---------------- figure 2: a few poses, each modality separately -------
    step = max(1, len(samples) // args.n_detail)
    det = sorted(samples, key=lambda s: s["margin"])[::step][: args.n_detail]
    fig, axes = plt.subplots(len(det), 3, figsize=(11.0, 3.5 * len(det)))
    axes = np.atleast_2d(axes)
    for row, s in zip(axes, det):
        v = s["vis_map"]
        draw_map(row[0], s)
        heat(row[0], v)
        row[0].set_ylabel(f"{s['scene']}\nmargin pct {100*s['margin_q']:.0f}", fontsize=8)
        row[0].set_title("visual posterior, marginal over yaw", fontsize=9)

        draw_map(row[1], s)
        a = s["ac_map"]
        fin = np.isfinite(a)
        if fin.any():
            lo, hi = np.nanpercentile(a[fin], [2, 100])
            row[1].imshow(np.ma.masked_invalid(a), cmap="winter", vmin=lo, vmax=hi, alpha=0.9)
        row[1].set_title("acoustic score, whole candidate grid", fontsize=9)

        draw_map(row[2], s)
        heat(row[2], v, alpha=0.45)
        sa = s["ac_short"]
        f2 = np.isfinite(sa)
        if f2.any():
            row[2].scatter(s["cols"][f2], s["rows"][f2], s=42, c=sa[f2],
                           cmap="winter", edgecolors="k", linewidths=0.3, zorder=4)
        row[2].set_title(f"shortlist re-ranked  ({'gate on' if s['gate_on'] else 'gate off'})"
                         f"\nvision {s['e_vis']:.2f} m  ->  gated {s['e_gate']:.2f} m", fontsize=9)
        for ax in row:
            markers(ax, s)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Where each modality puts its belief, and what the gate does with it\n"
                 "green star = truth,  red x = vision's answer,  blue circle = after the gate",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = out_dir / f"probability_maps_detail_{args.run_name}.png"
    fig.savefig(p2, dpi=135); plt.close(fig)
    print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
