#!/usr/bin/env python3
"""Put the acoustic scores next to the F3Loc visual baseline on identical poses.

Every method here scores the same candidate set -- the valid cells of the shared
``valid_pose_mask`` -- for the same observations, so recalls and ranks are
directly comparable and no method can gain from a different support.

Five things are compared:

  vision            the F3Loc observation model, position marginal of its
                    (H, W, 36) posterior
  acoustic 2D       the training-free one-sided event match against paths
                    predicted from the 2D floorplan alone
  acoustic rendered a SoundSpaces-rendered candidate grid, which is what the
                    same comparison can do when the predicted response is not a
                    2D proxy. It needs a per-scene render, so it is not a
                    deployable method; it is the ceiling that says how much of
                    the gap is representation.
  fused             vision and one acoustic score combined in log space after
                    rank normalisation, no learning and no fitted weight
  gated             the acoustic term applied only where the visual margin is
                    below a threshold fitted on the other collection

    python scripts/vision_vs_acoustic.py --nets mono mv comp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STFT_BANDS = [(0, 500), (500, 1500), (1500, 4000)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--nets", nargs="+", default=["mono", "mv", "comp"])
    p.add_argument("--run-prefix", default="echoloc_")
    p.add_argument("--run-suffix", default="_fg")
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--n-rays", type=int, default=72)
    p.add_argument("--fuse-weight", type=float, default=0.25)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out-dir", type=Path, default=None)
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

    from scripts.margin_gated_fusion import band_envelope, featurize_band
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.acoustic import AcousticProxyConfig, trace_echoes
    from track1_core.likelihood.events import (
        EventMatchConfig, event_match_score, observed_events, pad_event_sets,
        predicted_events_from_echoes,
    )
    from track1_core.models import CompDepthModule, MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CLS = {"mono": MonoDepthModule, "mv": MVDepthModule, "comp": CompDepthModule}
    ecfg = EventMatchConfig(top_fraction=1.0)       # full support, the best row
    L = 3

    # ---- per (collection, scene): candidates, 2D predicted events, rendered grid
    geo = {}
    for coll in args.collections:
        for scene in args.scenes:
            if not (root / coll / scene / "map.png").exists():
                continue
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows_g, cols_g = np.nonzero(mask)
            free = torch.tensor(occ == 255, device=device)
            origins_px = pg.grid_to_map(np.stack([cols_g, rows_g], axis=1))

            acfg = AcousticProxyConfig(n_rays=args.n_rays, max_order=1)
            ech = trace_echoes(free, torch.tensor(origins_px, dtype=torch.float32, device=device),
                               acfg, pg.map_resolution_m)
            ev = predicted_events_from_echoes({k: v.cpu().numpy() for k, v in ech.items()},
                                              ecfg, origins_px, pg.map_resolution_m, 1)
            pd_, pw_, _ = pad_event_sets(ev)

            # rendered candidate grid, aligned onto the same valid cells
            gpath = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
            rend = None
            if gpath.exists():
                blob = np.load(gpath)
                env = np.stack([band_envelope(c, ecfg.direct_guard_samples, 1024, 64, 16,
                                              ecfg.sample_rate_hz, STFT_BANDS)
                                for c in blob["rir"]])
                env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
                feat = featurize_band(env)
                lut = -np.ones(mask.shape, dtype=int)
                idx = blob["index"]
                lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
                pick = lut[rows_g, cols_g]
                aligned = np.zeros((len(rows_g), feat.shape[1], feat.shape[2]), dtype=np.float32)
                aligned[pick >= 0] = feat[pick[pick >= 0]]
                rend = dict(feat=aligned, present=pick >= 0)

            geo[(coll, scene)] = dict(
                pg=pg, mask=mask, rows=rows_g, cols=cols_g,
                desdf_t=torch.tensor(desdf["desdf"], device=device),
                pd=pd_, pw=pw_, rend=rend,
                poses=np.array([[float(v) for v in l.split()]
                                for l in open(root / coll / scene / "poses.txt") if l.strip()]))
            print(f"[geo] {coll}/{scene}: {len(rows_g)} cells, "
                  f"{(pw_ > 0).sum(1).mean():.1f} 2D events, "
                  f"rendered grid {'yes' if rend else 'no'}")

    recs: dict[str, list] = {n: [] for n in args.nets}
    for net in args.nets:
        ckpt = REPO_ROOT / "outputs" / f"{args.run_prefix}{net}{args.run_suffix}" / f"{net}.ckpt"
        if not ckpt.exists():
            print(f"[skip] {net}: no checkpoint")
            continue
        kw = dict(mono_ckpt=None, mv_ckpt=None) if net == "comp" else {}
        model = CLS[net].load_from_checkpoint(str(ckpt), **kw).to(device).eval()

        for (coll, scene), G in geo.items():
            rir_dir = root / "rir" / coll / args.condition / scene
            if not rir_dir.is_dir():
                continue
            dsdir = str(root / coll)
            ds = GridSeqDataset(dsdir, [scene], L=L, depth_dir=dsdir,
                                depth_suffix="depth40" if net == "mono" else "depth160")
            picks = np.linspace(0, len(ds) - 1, min(args.n_poses, len(ds))).astype(int)
            rows_g, cols_g, pg = G["rows"], G["cols"], G["pg"]

            for chunk in tqdm.tqdm(picks, desc=f"{net} {coll}/{scene}", leave=False):
                pose_idx = int(chunk) * (L + 1) + L
                f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
                if not f.exists() or pose_idx >= len(G["poses"]):
                    continue
                data = ds[int(chunk)]
                with torch.no_grad():
                    if net == "mono":
                        pred, _, _ = model.encoder(
                            torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
                    else:
                        b = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                             for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                        b["ref_mask"] = b["src_mask"] = None
                        pred = (model.net(b)["d"] if net == "mv"
                                else model.comp_d_net(b)["d_comp"])
                rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                    device=device, dtype=torch.float32)
                _, prob_dist, _, _ = localize(G["desdf_t"], rays)
                vis = np.array(prob_dist, dtype=np.float64)[rows_g, cols_g]

                rir = np.load(f)
                ac2d = event_match_score(G["pd"], G["pw"], observed_events(rir, ecfg), ecfg)

                acr = None
                if G["rend"] is not None:
                    obs_f = featurize_band(band_envelope(rir, ecfg.direct_guard_samples, 1024,
                                                         64, 16, ecfg.sample_rate_hz, STFT_BANDS))
                    nw = min(G["rend"]["feat"].shape[2], obs_f.shape[1])
                    acr = -np.abs(G["rend"]["feat"][:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))
                    acr = np.where(G["rend"]["present"], acr, acr.min() - 1.0)

                gx, gy, _ = pg.pose_metric_to_grid(G["poses"][pose_idx, :3])
                dist = np.hypot(cols_g - gx, rows_g - gy) * pg.grid_resolution_m
                gt = int(dist.argmin())

                lv = np.log(np.clip(rank_norm(vis), 1e-9, None))
                best = int(vis.argmax())
                far = np.hypot(cols_g - cols_g[best], rows_g - rows_g[best]) \
                    * pg.grid_resolution_m > args.mode_sep_m
                margin = float(np.log((vis[best] + 1e-300)
                                      / (vis[far].max() + 1e-300 if far.any() else 1e-300)))

                r = dict(coll=coll, scene=scene, margin=margin, n_cand=len(rows_g),
                         vision=float(dist[best]),
                         vision_rank=int((vis > vis[gt]).sum()) + 1,
                         acoustic2d=float(dist[int(ac2d.argmax())]),
                         acoustic2d_rank=int((ac2d > ac2d[gt]).sum()) + 1,
                         fused2d=float(dist[int((lv + args.fuse_weight
                                                 * np.log(np.clip(rank_norm(ac2d), 1e-9, None))).argmax())]))
                if acr is not None:
                    lr = np.log(np.clip(rank_norm(acr), 1e-9, None))
                    r.update(acoustic_rendered=float(dist[int(acr.argmax())]),
                             acoustic_rendered_rank=int((acr > acr[gt]).sum()) + 1,
                             fused_rendered=float(dist[int((lv + args.fuse_weight * lr).argmax())]),
                             dist_all=None)
                    r["_lr_best"] = int(acr.argmax())
                r["_dist"] = dist
                recs[net].append(r)

    # ---- margin gate: threshold fitted on replica_f, applied to replica_g ----
    for net, rs in recs.items():
        if not rs:
            continue
        fit = [r for r in rs if r["coll"] != args.collections[-1]]
        tau = float(np.quantile([r["margin"] for r in (fit or rs)], args.gate_quantile))
        for r in rs:
            on = r["margin"] < tau
            r["gated2d"] = r["fused2d"] if on else r["vision"]
            if "fused_rendered" in r:
                r["gated_rendered"] = r["fused_rendered"] if on else r["vision"]
            r["gate_on"] = on

    METHODS = [("vision", "F3Loc vision only"),
               ("acoustic2d", "acoustic 2D floorplan, alone"),
               ("acoustic_rendered", "acoustic rendered grid, alone"),
               ("fused2d", "vision + acoustic 2D"),
               ("gated2d", "vision + acoustic 2D, gated"),
               ("fused_rendered", "vision + rendered"),
               ("gated_rendered", "vision + rendered, gated")]

    table: dict[str, dict] = {}
    print(f"\n{'method':36s} {'0.1m':>7s} {'0.5m':>7s} {'1m':>7s} {'med err':>9s} "
          f"{'GT rank':>9s} {'vs vision':>10s}")
    print("-" * 92)
    for net, rs in recs.items():
        if not rs:
            continue
        base = float(np.mean([r["vision"] < 1 for r in rs]))
        print(f"[{net}]  {len(rs)} poses, {int(np.mean([r['n_cand'] for r in rs]))} candidates each")
        for key, label in METHODS:
            vals = [r[key] for r in rs if key in r and r[key] is not None]
            if not vals:
                continue
            e = np.array(vals)
            rk = [r.get(key + "_rank") for r in rs if r.get(key + "_rank") is not None]
            m = dict(n=len(e), r01=float((e < 0.1).mean()), r05=float((e < 0.5).mean()),
                     r1=float((e < 1).mean()), median_m=float(np.median(e)),
                     gt_rank_median=float(np.median(rk)) if rk else None)
            table[f"{net}/{key}"] = m
            rank_s = "        -" if m["gt_rank_median"] is None else f"{m['gt_rank_median']:9.0f}"
            delta_s = "" if key == "vision" else f"{100 * (m['r1'] - base):+9.1f}"
            print(f"  {label:34s} {100*m['r01']:6.1f}% {100*m['r05']:6.1f}% "
                  f"{100*m['r1']:6.1f}% {m['median_m']:8.2f}m {rank_s} {delta_s:>10s}")

    # ---------------- figure ------------------------------------------------
    net0 = next((n for n in args.nets if recs.get(n)), None)
    if net0 is None:
        return 1
    rs = recs[net0]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # A: recall at 1 m per net and method
    ax = axes[0, 0]
    keys = [k for k, _ in METHODS if any(f"{n}/{k}" in table for n in args.nets)]
    w = 0.8 / max(len(keys), 1)
    for j, k in enumerate(keys):
        vals = [100 * table.get(f"{n}/{k}", {}).get("r1", np.nan) for n in args.nets]
        ax.bar(np.arange(len(args.nets)) + j * w, vals, w,
               label=dict(METHODS)[k], edgecolor="k", linewidth=0.4)
    ax.set_xticks(np.arange(len(args.nets)) + 0.4 - w / 2)
    ax.set_xticklabels(args.nets)
    ax.set_ylabel("recall at 1 m (%)")
    ax.set_title("A  localization recall, F3Loc metric", loc="left", fontsize=10)
    # headroom so the legend never sits on top of a bar
    ax.set_ylim(0, max(60.0, ax.get_ylim()[1] * 1.55))
    ax.legend(fontsize=6.6, ncol=2, loc="upper center", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # B: error CDF
    ax = axes[0, 1]
    for k, label in METHODS:
        v = np.array([r[k] for r in rs if k in r and r[k] is not None])
        if v.size == 0:
            continue
        ax.plot(np.sort(v), np.arange(1, v.size + 1) / v.size, lw=1.6, label=label)
    ax.axvline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlim(0, 8); ax.set_xlabel("position error (m)"); ax.set_ylabel("fraction of poses")
    ax.set_title(f"B  error CDF ({net0})", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

    # C: GT rank CDF, the ranking view rather than the top-1 view
    ax = axes[1, 0]
    for k in ("vision", "acoustic2d", "acoustic_rendered"):
        v = np.array([r[k + "_rank"] for r in rs if r.get(k + "_rank") is not None], dtype=float)
        if v.size == 0:
            continue
        v = v / np.mean([r["n_cand"] for r in rs]) * 100
        ax.plot(np.sort(v), np.arange(1, v.size + 1) / v.size, lw=1.8,
                label=dict(METHODS)[k])
    ax.plot([0, 100], [0, 1], color="0.6", ls=":", lw=1.2, label="chance")
    ax.set_xscale("log"); ax.set_xlabel("GT rank, % of candidates (log)")
    ax.set_ylabel("fraction of poses")
    ax.set_title("C  how high does each modality rank the truth", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

    # D: gain against visual ambiguity
    ax = axes[1, 1]
    margins = np.array([r["margin"] for r in rs])
    q = np.quantile(margins, [1 / 3, 2 / 3])
    labels = ["ambiguous", "middle", "confident"]
    groups = [margins <= q[0], (margins > q[0]) & (margins <= q[1]), margins > q[1]]
    x = np.arange(3)
    for j, k in enumerate(["vision", "fused2d", "gated2d", "fused_rendered"]):
        vals = []
        for g in groups:
            v = [rs[i][k] for i in np.flatnonzero(g) if k in rs[i] and rs[i][k] is not None]
            vals.append(100 * np.mean(np.array(v) < 1) if v else np.nan)
        ax.bar(x + j * 0.2, vals, 0.2, label=dict(METHODS)[k], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x + 0.3); ax.set_xticklabels(labels)
    ax.set_xlabel("visual margin stratum (computed without ground truth)")
    ax.set_ylabel("recall at 1 m (%)")
    ax.set_title("D  does acoustics help where vision is unsure", loc="left", fontsize=10)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"F3Loc vision against training-free acoustic scoring   "
                 f"{len(rs)} poses, {args.condition}, identical candidate set", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out_dir / "vision_vs_acoustic.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {p}")

    # The metrics path has to follow --out-dir, not sit at a fixed location:
    # a second run with different checkpoints silently overwrote the first
    # because only the figure path was parameterised.
    o = (out_dir / "vision_vs_acoustic.json") if args.out_dir \
        else (REPO_ROOT / "outputs" / "metrics" / "vision_vs_acoustic.json")
    slim = {n: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rs]
            for n, rs in recs.items() if rs}
    o.write_text(json.dumps({"condition": args.condition, "fuse_weight": args.fuse_weight,
                             "table": table, "per_pose": slim}, indent=2))
    print(f"wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
