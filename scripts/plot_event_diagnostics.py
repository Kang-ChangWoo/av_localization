#!/usr/bin/env python3
"""Diagnostics for one-sided acoustic evidence matching.

Five panels per observation, laid out so a failure can be attributed rather
than merely observed:

  A  the observed delay-energy profile with the extracted arrivals
  B  predicted arrivals at the true pose, at the strongest wrong candidate,
     and at a random valid candidate, on the same time axis as A
  C  which predicted arrivals found support and which did not, by path type
  D  the acoustic score over the valid pose grid
  E  the visual likelihood, the acoustic likelihood, and their product

Panels A to C share one x axis in milliseconds of window time, so a systematic
offset between the two sides is visible directly rather than inferred from a
ranking that got worse.

    python scripts/plot_event_diagnostics.py --scene office_4 --n-figures 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

KIND_STYLE = {0: ("radial wall return", "o", "#1565c0"),
              1: ("higher order", "s", "#ef6c00"),
              2: ("corner NLOS", "^", "#6a1b9a"),
              3: ("vertical", "v", "#2e7d32")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", default="office_4")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--n-figures", type=int, default=4)
    p.add_argument("--n-rays", type=int, default=72)
    p.add_argument("--max-order", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--top-fraction", type=float, default=0.6)
    p.add_argument("--with-corner", action="store_true")
    p.add_argument("--run-name", default="echoloc_mono_fg")
    p.add_argument("--net", default="mono")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import torch

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.acoustic import AcousticProxyConfig, trace_echoes
    from track1_core.likelihood.events import (
        EventMatchConfig, event_match_score, observed_events, observed_profile,
        pad_event_sets, predicted_events_from_echoes,
    )

    root = Path(args.dataset_root)
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "figures" / "event_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene, coll = args.scene, args.collection

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, pg)
    rows_g, cols_g = np.nonzero(mask)
    sel = (rows_g % args.stride == 0) & (cols_g % args.stride == 0)
    rows_g, cols_g = rows_g[sel], cols_g[sel]
    free = torch.tensor(occ == 255, device="cuda" if torch.cuda.is_available() else "cpu")
    origins_px = pg.grid_to_map(np.stack([cols_g, rows_g], axis=1))

    ecfg = EventMatchConfig(top_fraction=args.top_fraction)
    acfg = AcousticProxyConfig(n_rays=args.n_rays, max_order=args.max_order)
    echoes = trace_echoes(free, torch.tensor(origins_px, dtype=torch.float32, device=free.device),
                          acfg, pg.map_resolution_m)
    ev = predicted_events_from_echoes({k: v.cpu().numpy() for k, v in echoes.items()},
                                      ecfg, origins_px, pg.map_resolution_m, args.max_order)
    pd_, pw_, pk_ = pad_event_sets(ev)
    print(f"[diag] {scene}: {len(rows_g)} candidates, "
          f"{(pw_ > 0).sum(1).mean():.1f} predicted events each")

    if args.with_corner:
        from track1_core.likelihood.corners import (
            CornerConfig, corner_events, corner_remote_distances, extract_corners)
        ccfg = CornerConfig()
        fn = free.cpu().numpy()
        cpts = extract_corners(fn, pg.map_resolution_m, ccfg)
        rho, ang = corner_remote_distances(fn, cpts, pg.map_resolution_m, ccfg)
        cd, cw, ck = corner_events(fn, origins_px, cpts, rho, ang, pg.map_resolution_m, ccfg, ecfg)
        pd_ = np.concatenate([pd_, cd], axis=1)
        pw_ = np.concatenate([pw_, cw], axis=1)
        pk_ = np.concatenate([pk_, ck], axis=1)

    # optional visual likelihood for panel E
    vis_model = None
    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    if ckpt.exists():
        from track1_core.models import MonoDepthModule
        from utils.data_utils import GridSeqDataset
        from utils.localization_utils import get_ray_from_depth, localize
        d2 = dict(desdf)
        d2["desdf"] = np.clip(desdf["desdf"], None, 10)
        vis_model = dict(model=MonoDepthModule.load_from_checkpoint(str(ckpt)).to(free.device).eval(),
                         desdf_t=torch.tensor(d2["desdf"], device=free.device),
                         ds=GridSeqDataset(str(root / coll), [scene], L=3,
                                           depth_dir=str(root / coll), depth_suffix="depth40"),
                         rays=get_ray_from_depth, loc=localize)
        print("[diag] visual likelihood available")

    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / coll / scene / "poses.txt") if l.strip()])
    rir_dir = root / "rir" / coll / args.condition / scene
    dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))
    dirs = [dirs[i] for i in np.linspace(0, len(dirs) - 1, args.n_figures).astype(int)]

    for d in dirs:
        i = int(d.split("_")[1])
        f = rir_dir / d / "rir.npy"
        if not f.exists() or i >= len(poses):
            continue
        rir = np.load(f)
        obs = observed_events(rir, ecfg)
        times, prof = observed_profile(rir, ecfg)
        score = event_match_score(pd_, pw_, obs, ecfg)

        gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
        gt = int(np.argmin((cols_g - gx) ** 2 + (rows_g - gy) ** 2))
        far = np.hypot(cols_g - gx, rows_g - gy) * pg.grid_resolution_m > 2.0
        wrong = int(np.flatnonzero(far)[np.argmax(score[far])]) if far.any() else 0
        rnd = int(np.random.default_rng(i).integers(len(score)))

        fig = plt.figure(figsize=(15.5, 9.5))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.35], hspace=0.45, wspace=0.25)

        # ---- A: observed profile and extracted arrivals
        axA = fig.add_subplot(gs[0, :])
        axA.fill_between(times, prof / max(prof.max(), 1e-30), color="#90a4ae", lw=0,
                         label="observed energy")
        axA.vlines(obs.delay_ms, 0, obs.weight / max(obs.weight.max(), 1e-30),
                   color="#c62828", lw=1.6, label=f"extracted arrivals (n={len(obs)})")
        axA.set_title(f"A  observed delay-energy profile   [{scene} {d}, {args.condition}]",
                      loc="left", fontsize=10)
        axA.set_ylabel("normalised energy")
        axA.legend(fontsize=8, loc="upper right")

        # ---- B: predicted arrivals at three candidates
        axB = fig.add_subplot(gs[1, :], sharex=axA)
        for row, (idx, name, col) in enumerate([
                (gt, "ground truth", "#2e7d32"),
                (wrong, "strongest wrong (>2 m away)", "#c62828"),
                (rnd, "random valid candidate", "#607d8b")]):
            m = pw_[idx] > 0
            axB.vlines(pd_[idx][m], row, row + 0.8, color=col, lw=1.6)
            axB.text(0.995, row + 0.4, f"{name}   score {score[idx]:.3f}",
                     transform=axB.get_yaxis_transform(), ha="right", va="center",
                     fontsize=8, color=col)
        axB.set_yticks([]); axB.set_ylim(0, 3)
        axB.set_title("B  predicted arrivals from the 2D floorplan", loc="left", fontsize=10)
        axB.set_xlabel("delay within the reflection window (ms)")

        # ---- C: supported vs unsupported predicted arrivals at the truth
        axC = fig.add_subplot(gs[2, 0])
        m = pw_[gt] > 0
        dly, wts, knd = pd_[gt][m], pw_[gt][m], pk_[gt][m]
        near = (np.abs(dly[:, None] - obs.delay_ms[None, :]) / ecfg.tau_tolerance_ms)
        sup = np.exp(-0.5 * near.min(axis=1) ** 2) if len(obs) else np.zeros_like(dly)
        for k in np.unique(knd):
            lab, mk, col = KIND_STYLE.get(int(k), ("other", "x", "k"))
            s = knd == k
            axC.scatter(dly[s], sup[s], marker=mk, c=col, s=34, label=lab, zorder=3)
        axC.axhline(np.exp(-0.5), color="0.5", ls="--", lw=0.8)
        axC.text(0.02, np.exp(-0.5) + 0.02, "one sigma", transform=axC.get_yaxis_transform(),
                 fontsize=7, color="0.4")
        axC.set_ylim(-0.05, 1.05)
        axC.set_xlabel("predicted delay (ms)"); axC.set_ylabel("support")
        axC.set_title("C  support per predicted arrival, at the truth", loc="left", fontsize=10)
        axC.legend(fontsize=7, loc="lower right")

        # ---- D: acoustic score over the grid
        axD = fig.add_subplot(gs[2, 1])
        amap = np.full(mask.shape, np.nan)
        amap[rows_g, cols_g] = score
        axD.imshow(occ_grid(occ, pg), cmap="gray", vmin=0, vmax=255, alpha=0.4)
        im = axD.imshow(np.ma.masked_invalid(amap), cmap="viridis")
        axD.scatter(gx, gy, marker="*", s=150, c="#00e676", edgecolors="k", lw=0.5)
        axD.scatter(cols_g[wrong], rows_g[wrong], marker="x", s=70, c="#ff1744", lw=2)
        axD.set_xticks([]); axD.set_yticks([])
        axD.set_title("D  acoustic score (star = truth)", loc="left", fontsize=10)
        fig.colorbar(im, ax=axD, fraction=0.045)

        # ---- E: vision, acoustics, product
        axE = fig.add_subplot(gs[2, 2])
        if vis_model is not None and (i - 3) % 4 == 0 and (i - 3) // 4 < len(vis_model["ds"]):
            chunk = (i - 3) // 4
            data = vis_model["ds"][chunk]
            with torch.no_grad():
                pred, _, _ = vis_model["model"].encoder(
                    torch.tensor(data["ref_img"], device=free.device).unsqueeze(0), None)
            r = torch.tensor(vis_model["rays"](pred.squeeze(0).float().cpu().numpy()),
                             device=free.device, dtype=torch.float32)
            _, prob, _, _ = vis_model["loc"](vis_model["desdf_t"], r)
            v = np.array(prob, dtype=np.float64)
            v[~mask] = 0.0
            vv = v[rows_g, cols_g]
            vv = vv / max(vv.sum(), 1e-300)
            aa = score / max(score.sum(), 1e-30)
            fused = vv * aa
            fmap = np.full(mask.shape, np.nan)
            fmap[rows_g, cols_g] = fused / max(fused.sum(), 1e-30)
            axE.imshow(occ_grid(occ, pg), cmap="gray", vmin=0, vmax=255, alpha=0.4)
            axE.imshow(np.ma.masked_invalid(fmap), cmap="magma_r",
                       norm=LogNorm(vmin=max(np.nanmax(fmap) * 1e-3, 1e-12),
                                    vmax=max(np.nanmax(fmap), 1e-11)))
            axE.scatter(gx, gy, marker="*", s=150, c="#00e676", edgecolors="k", lw=0.5)
            rv = int((vv > vv[gt]).sum()) + 1
            rf = int((fused > fused[gt]).sum()) + 1
            axE.set_title(f"E  vision x acoustics   GT rank {rv} -> {rf}", loc="left", fontsize=10)
        else:
            axE.text(0.5, 0.5, "no visual likelihood\nfor this pose index",
                     ha="center", va="center", fontsize=9, color="0.4")
            axE.set_title("E  vision x acoustics", loc="left", fontsize=10)
        axE.set_xticks([]); axE.set_yticks([])

        rank = int((score > score[gt]).sum()) + 1
        fig.suptitle(f"One-sided event matching   {scene} {d}   "
                     f"GT acoustic rank {rank} of {len(score)}   "
                     f"order<={args.max_order}, top_fraction={args.top_fraction}", fontsize=12)
        p = out_dir / f"{scene}_{d}_{args.condition}.png"
        fig.savefig(p, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")
    return 0


def occ_grid(occ, pg):
    """``map.png`` resampled into the pose grid frame (a 10x coarser crop)."""
    import cv2
    s = pg.cells_per_map_pixel
    r, c = int(round(pg.height * s)), int(round(pg.width * s))
    crop = occ[pg.top: pg.top + r, pg.left: pg.left + c]
    if crop.shape != (r, c):
        pad = np.zeros((r, c), dtype=occ.dtype)
        pad[: crop.shape[0], : crop.shape[1]] = crop
        crop = pad
    return cv2.resize(crop, (pg.width, pg.height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    raise SystemExit(main())
