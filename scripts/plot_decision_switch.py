#!/usr/bin/env python3
"""Why the answer changed, shown as a decision between competing modes.

A heat map over the whole floor shows where belief is, but not why one place
won. Both the visual likelihood and the acoustic score are nearly flat over most
cells and concentrated in a handful of separated modes, and the fusion is
decided entirely by the ranking *among those modes*. Everything else is noise
that the picture spends its area on.

So this reduces each pose to that decision. Modes are extracted from the visual
posterior by greedy non-maximum suppression at ``--mode-sep-m``, which is the
same separation the confidence margin uses, and each surviving mode is scored
three ways, all as rank percentiles over the valid cells so they share an axis:

    visual      where UnLoc puts it
    acoustic    where the candidate grid puts it
    fused       log(visual rank) + w * log(acoustic rank), the rule in use

The bars are drawn in log units rather than as percentiles, because a percentile
hides the decision: every mode vision shortlists sits above the 99.9th, so those
bars are all full and identical. In log units the visual term among shortlisted
modes is a few hundredths while the acoustic term is whole units, which is the
point. Vision chooses the shortlist and sound orders it.

The bar chart is the whole argument for one pose. The visual winner is the top
bar; if the fused winner is a different bar, the acoustic column is why, and the
distance-to-truth column says whether that was the right call.

The map panel is cropped to the modes actually in contention rather than the
whole room, so a 30 cm difference is visible instead of being three pixels.

    /opt/conda/envs/unloc/bin/python scripts/plot_decision_switch.py \
        --checkpoint <replica ckpt> --weight 0.5
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
    p.add_argument("--n-modes", type=int, default=5, help="competing modes drawn per pose")
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--fix-rows", type=int, default=6)
    p.add_argument("--break-rows", type=int, default=2)
    p.add_argument("--orn-slice", type=int, default=36)
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def rank_norm(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.shape[0], dtype=np.float64)
    r[np.argsort(x)] = np.arange(x.shape[0])
    return (r + 0.5) / x.shape[0]


def separated_modes(vis: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    res: float, sep_m: float, k: int) -> list[int]:
    """Greedy non-maximum suppression on the visual posterior.

    Taking the top-k cells outright would return k neighbours of one peak and
    describe no decision at all, so each pick suppresses everything within
    ``sep_m`` before the next is taken.
    """
    order = np.argsort(-vis)
    picked: list[int] = []
    taken = np.zeros(len(vis), dtype=bool)
    for i in order:
        if taken[i]:
            continue
        picked.append(int(i))
        taken |= np.hypot(cols - cols[i], rows - rows[i]) * res <= sep_m
        if len(picked) == k:
            break
    return picked


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
    from matplotlib.colors import LogNorm

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
            fu = np.log(rv) + args.weight * np.log(ra)

            gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
            dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
            modes = separated_modes(vis, rows, cols, pg.grid_resolution_m,
                                    args.mode_sep_m, args.n_modes)
            vb, fb = int(vis.argmax()), int(fu.argmax())
            if fb not in modes:               # the fused winner must be on the chart
                modes = modes[:-1] + [fb]
            samples.append(dict(
                scene=scene, occ=occ, rgb=rgb, res=pg.grid_resolution_m,
                gt=(gx, gy), e_vis=float(dist[vb]), e_fu=float(dist[fb]),
                vb=vb, fb=fb, modes=modes,
                xy=[(int(cols[m]), int(rows[m])) for m in modes],
                d=[float(dist[m]) for m in modes],
                rv=[float(rv[m]) for m in modes], ra=[float(ra[m]) for m in modes],
                rf=[float(fu[m]) for m in modes],
                v_vis=np.where(mask, np.nan, np.nan), rows=rows, cols=cols,
                vis_field=vis, ac_field=ra, mask_shape=mask.shape))

    if not samples:
        print("nothing to plot")
        return 1
    fixed = sorted([s for s in samples if s["e_vis"] >= 1.0 and s["e_fu"] < 1.0],
                   key=lambda s: -s["e_vis"])[: args.fix_rows]
    broke = sorted([s for s in samples if s["e_vis"] < 1.0 and s["e_fu"] >= 1.0],
                   key=lambda s: -s["e_fu"])[: args.break_rows]
    show = [("SOUND FIXES IT", s) for s in fixed] + [("SOUND BREAKS IT", s) for s in broke]
    print(f"[plot] {len(samples)} poses; drawing {len(fixed)} repairs, {len(broke)} regressions")

    def onto(v, s):
        a = np.full(s["mask_shape"], np.nan)
        a[s["rows"], s["cols"]] = v
        return a

    fig, axes = plt.subplots(len(show), 4, figsize=(4.1 * 4, 3.3 * len(show)),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.25, 1.5]})
    axes = np.atleast_2d(axes)
    for r, (tag, s) in enumerate(show):
        xs = [p[0] for p in s["xy"]] + [s["gt"][0]]
        ys = [p[1] for p in s["xy"]] + [s["gt"][1]]
        pad = max(8, int(0.25 * (max(max(xs) - min(xs), max(ys) - min(ys)))))
        box = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

        axes[r, 0].imshow(s["rgb"]); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"{tag}\n{s['scene']}\n"
                              f"{s['e_vis']:.1f} m -> {s['e_fu']:.1f} m", fontsize=9)
        if r == 0:
            axes[r, 0].set_title("camera", fontsize=10)

        for c, (field, name, log) in enumerate(
                ((s["vis_field"], "UnLoc likelihood", True),
                 (s["ac_field"], "acoustic score (rank)", False)), start=1):
            ax = axes[r, c]
            v = onto(field, s)
            nz = v[np.isfinite(v) & (v > 0)]
            if nz.size:
                lo, hi = float(np.quantile(nz, 0.90)), float(np.nanmax(v))
                if hi > lo > 0:
                    ax.imshow(np.ma.masked_where(~np.isfinite(v) | (v <= lo), v),
                              cmap="magma_r",
                              norm=LogNorm(vmin=lo, vmax=hi) if log else None, alpha=0.9)
            ax.imshow(s["occ"], cmap="gray", vmin=0, vmax=255, alpha=0.35, zorder=-1)
            # every competing mode is numbered, so the bars can be read off the map
            for j, (px, py) in enumerate(s["xy"]):
                ax.annotate(str(j + 1), (px, py), color="k", fontsize=9, weight="bold",
                            ha="center", va="center", zorder=6,
                            bbox=dict(boxstyle="circle,pad=0.15", fc="white",
                                      ec="#2962ff" if s["modes"][j] == s["fb"] else
                                         ("#d50000" if s["modes"][j] == s["vb"] else "0.5"),
                                      lw=1.8))
            ax.scatter(*s["gt"], marker="*", s=190, c="#00c853",
                       edgecolors="k", linewidths=0.6, zorder=7)
            ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)

        # The decision, drawn as the arithmetic that actually produces it. A
        # rank percentile hides the decision, because every mode vision short-
        # lists sits at 0.999-something and the bars all look full. The fusion
        # adds log(rank), and it is in log units that the two terms are
        # comparable: the visual term is a few hundredths, the acoustic term is
        # whole units. That gap is the finding, so the figure has to show it.
        ax = axes[r, 3]
        n = len(s["modes"])
        y = np.arange(n)
        lv = np.log(np.array(s["rv"]))
        la = args.weight * np.log(np.array(s["ra"]))
        tot = lv + la
        ax.barh(y - 0.26, lv, height=0.24, color="#d50000", label="log visual rank")
        ax.barh(y, la, height=0.24, color="#7b1fa2",
                label=f"{args.weight:g} x log acoustic rank")
        ax.barh(y + 0.26, tot, height=0.24, color="#2962ff", label="sum, the fused score")
        lo = float(min(tot.min(), la.min())) * 1.35 - 0.05
        for j in range(n):
            mark = ""
            if s["modes"][j] == s["vb"]:
                mark += "  <- vision picks"
            if s["modes"][j] == s["fb"]:
                mark += "  <- fusion picks"
            ax.text(0.02 * abs(lo), y[j], f"{s['d'][j]:.1f} m{mark}", fontsize=8,
                    va="center", ha="left",
                    color="#00a040" if s["d"][j] < 1.0 else "0.25")
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([str(j + 1) for j in range(n)])
        ax.invert_yaxis(); ax.set_xlim(lo, abs(lo) * 0.55)
        ax.set_xlabel("log score, less negative is better", fontsize=8)
        ax.tick_params(labelsize=8)
        if r == 0:
            ax.set_title("the decision: log(visual) + w log(acoustic)", fontsize=10)
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Why the answer changes. Numbered circles are spatially separated modes of the visual "
        "posterior;\nred outline is UnLoc's pick, blue is the fused pick, green star is truth. "
        "Right: each mode's rank\nunder vision, under sound, and under the fusion. "
        "The label gives its distance to truth.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = args.out_dir / f"unloc_decision_switch{args.tag}.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
