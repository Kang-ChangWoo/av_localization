#!/usr/bin/env python3
"""One pass over the queries that produces the tables every later analysis reads.

The diagnosis this repository needs asks a dozen different questions of the same
underlying quantities: the visual posterior, the acoustic score, the separated
visual modes, and their relation to ground truth. Recomputing those per question
would mean running the visual backbone and re-featurising a multi-gigabyte
candidate grid a dozen times, and would let the questions drift apart. So this
runs once per (condition, backbone) and writes three flat tables.

    queries_*.csv       one row per query: visual ambiguity measures, acoustic
                        discriminativeness measures, ground-truth coverage,
                        smoothness of the acoustic field around truth, and the
                        outcome of every existing decision rule
    modes_*.csv         one row per query and separated visual mode: position,
                        raw and rank scores under both modalities, local
                        aggregates over the mode's disc, distance to truth
    mode_ranges_*.csv   one row per query, mode and post-direct time range, for
                        the early-versus-late ablation

Two things make this affordable. The 2 ms envelope is computed once per grid cell
at the full 128 ms window, and every time range is then a slice of those 64
frames rather than a re-read of the impulse response. And the visual posterior is
computed once and shared by every acoustic variant, since none of them touch the
image.

Run it twice, once per acoustic condition, to get the domain-gap decomposition:

    python scripts/extract_mode_table.py --condition raw_scan_open
    python scripts/extract_mode_table.py --condition floorplan_closed
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
DISCO_ROOT = REPO_ROOT.parent / "DisCo-FLoc"
sys.path.insert(0, str(REPO_ROOT))

# post-direct time ranges, in milliseconds. The envelope frame is 2 ms, so each
# range is an exact frame slice; nothing is resampled or re-windowed.
TIME_RANGES = [("0-16", 0, 16), ("16-32", 16, 32), ("32-64", 32, 64),
               ("64-128", 64, 128), ("0-32", 0, 32), ("0-64", 0, 64),
               ("0-128", 0, 128)]
PRIMARY_RANGE = "0-128"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open",
                   choices=["raw_scan_open", "floorplan_closed"])
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--backbone", default="unloc", choices=["unloc", "f3loc_mono"])
    p.add_argument("--checkpoint", default=None,
                   help="visual weights; defaults per backbone")
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--feature", default="envelope", choices=["envelope", "stft_band"])
    p.add_argument("--nfft", type=int, default=64)
    p.add_argument("--hop", type=int, default=16)
    p.add_argument("--tag", default="", help="suffix on the output filenames")
    p.add_argument("--nms-radius-m", type=float, default=1.5)
    p.add_argument("--n-modes", type=int, default=10)
    p.add_argument("--local-radius-m", type=float, default=0.5)
    p.add_argument("--peak-rank", type=float, default=0.99,
                   help="acoustic rank above which a cell counts as a strong peak")
    p.add_argument("--legacy-weight", type=float, default=0.5,
                   help="weight of the existing cell-wise log-rank fusion, "
                        "recorded so the new rules can be compared against it")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--n-poses", type=int, default=100, help="per scene and collection")
    p.add_argument("--orn-slice", type=int, default=36)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def rank_norm(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.shape[0], dtype=np.float64)
    r[np.argsort(x)] = np.arange(x.shape[0])
    return (r + 0.5) / x.shape[0]


def orn_err_deg(pred_rad: float, gt_rad: float) -> float:
    d = (float(pred_rad) - float(gt_rad)) % (2 * np.pi)
    return min(d, 2 * np.pi - d) / np.pi * 180.0


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import tqdm

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, featurise, observation_rate, score, to_config_rate,
    )
    from track1_core.modes import (
        ModeConfig, aggregate, entropy, extract_modes, local_discs, margins,
        relative_evidence, softmax,
    )
    from track1_core.provenance import stamp

    mcfg = ModeConfig(nms_radius_m=args.nms_radius_m, n_modes=args.n_modes,
                      local_radius_m=args.local_radius_m)

    # ---- visual backbone -------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.backbone == "unloc":
        sys.path.insert(0, str(UNLOC_ROOT))
        from modules.depth_net_pl import UnLocDepthModule
        from utils.localization_utils import (
            get_ray_from_depth_uncertainty, localize_uncertainty,
        )
        ck = args.checkpoint or str(
            UNLOC_ROOT / "tb_logs/my_model/version_1/checkpoints/epoch=19-step=1040.ckpt")
        cwd = os.getcwd()
        os.chdir(UNLOC_ROOT)
        try:
            net = UnLocDepthModule.load_from_checkpoint(
                checkpoint_path=ck, strict=False).to(device).eval()
        finally:
            os.chdir(cwd)

        def posterior(img_bgr, desdf_t):
            x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            x = torch.tensor(np.transpose(x, (2, 0, 1))[None],
                             dtype=torch.float32, device=device)
            with torch.no_grad():
                loc, sc = net.encoder(x, None)[:2]
            pr, ps = get_ray_from_depth_uncertainty(
                loc.squeeze(0).cpu().numpy(), sc.squeeze(0).cpu().numpy())
            _, pd_, orn, _ = localize_uncertainty(
                desdf_t, torch.tensor(pr, device=device),
                torch.tensor(ps, device=device), return_np=False,
                orn_slice=args.orn_slice)
            return (np.asarray(pd_.cpu(), dtype=np.float64),
                    np.asarray(orn.cpu()))
    else:
        # the vendored F3Loc tree keeps upstream's absolute imports, so its root
        # has to be on the path before `utils` resolves
        from track1_core._vendor import ensure_on_path
        ensure_on_path()
        from utils.localization_utils import get_ray_from_depth, localize
        from track1_core.models import MonoDepthModule
        ck = args.checkpoint or str(REPO_ROOT / "outputs/echoloc_mono_fg/mono.ckpt")
        net = MonoDepthModule.load_from_checkpoint(ck).to(device).eval()
        F_W = 1 / (2 * np.tan(np.deg2rad(106.2602) / 2))

        def posterior(img_bgr, desdf_t):
            x = img_bgr[:, :, ::-1].astype(np.float64) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            x = torch.tensor(np.transpose(x, (2, 0, 1))[None],
                             dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = net.encoder(x, None)[0].squeeze(0).float().cpu().numpy()
            rays = torch.tensor(get_ray_from_depth(pred, V=11, F_W=F_W),
                                device=device, dtype=torch.float32)
            _, pd_, orn, _ = localize(desdf_t, rays, return_np=False)
            return (np.asarray(pd_.cpu(), dtype=np.float64), np.asarray(orn.cpu()))
    print(f"[visual] {args.backbone}: {ck}")

    root = Path(args.dataset_root)
    q_rows, m_rows, r_rows, inputs = [], [], [], []

    for coll in args.collections:
        for scene in args.scenes:
            g = REPO_ROOT / args.grid_dir / f"{scene}.npz"
            if not g.exists() or not (root / coll / scene / "map.png").exists():
                continue
            inputs.append(g)
            blob = np.load(g)
            sr = int(json.loads(str(blob["config"]))["sample_rate"])
            full = GridScoreConfig(feature=args.feature, window_ms=args.window_ms,
                                   nfft=args.nfft, hop=args.hop,
                                   sample_rate_hz=sr,
                                   direct_guard_samples=int(round(sr * 2 / 1000)),
                                   usable_samples=int(round(sr * 128 / 1000)))
            # frames per millisecond, whichever feature is in use: the envelope
            # frame is window_ms long, the STFT frame is hop/sample_rate long
            fpm = 1.0 / full.ms_per_frame

            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            res = pg.grid_resolution_m
            mask = valid_pose_mask(occ, pg)
            rows, cols = np.nonzero(mask)
            dt = torch.tensor(desdf["desdf"], device=device)

            # the whole grid's envelope, once, at the full window; every time
            # range below is a slice of these frames
            E = np.stack([band_energy(c, full) for c in blob["rir"]])  # (cells, 6, 64)
            idx = blob["index"]
            lut = -np.ones(mask.shape, dtype=np.int64)
            lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
            pick = lut[rows, cols]
            present = pick >= 0
            cand = {}
            for name, lo, hi in TIME_RANGES:
                f = featurise(E[:, :, int(lo * fpm): int(hi * fpm)], full)
                a = np.zeros((len(rows), f.shape[1], f.shape[2]), dtype=np.float32)
                a[present] = f[pick[present]]
                cand[name] = a
            del E
            print(f"[geo] {coll}/{scene}: {len(rows)} cells, "
                  f"{present.sum()} with a rendered candidate, {len(TIME_RANGES)} ranges")

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rd = root / "rir" / coll / args.condition / scene
            ds_dirs = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
            picks = np.linspace(0, len(ds_dirs) - 1,
                                min(args.n_poses, len(ds_dirs))).astype(int)

            for k in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                dname = ds_dirs[int(k)]
                i = int(dname.split("_")[1])
                f = rd / dname / "rir.npy"
                img_p = root / coll / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"
                if not f.exists() or not img_p.exists() or i >= len(poses):
                    continue
                qid = f"{coll}/{scene}/{i:05d}"
                img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
                pdist, orns = posterior(img, dt)
                vis = pdist[rows, cols]
                yaw = pg.bin_to_yaw(orns[rows, cols])

                obs_full = band_energy(np.load(f), full, observation_rate(f))
                ac_by_range = {}
                for name, lo, hi in TIME_RANGES:
                    o = featurise(obs_full[:, int(lo * fpm): int(hi * fpm)], full)
                    ac_by_range[name] = score(cand[name], o, present, full)
                ac = ac_by_range[PRIMARY_RANGE]

                rv, ra = rank_norm(vis), rank_norm(ac)
                gx, gy, gth = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * res
                gt_cell = int(dist.argmin())

                modes = extract_modes(vis, rows, cols, res, mcfg)
                discs = local_discs(modes, rows, cols, res, mcfg.local_radius_m)
                vis_agg = aggregate(np.log(np.clip(vis, 1e-300, None)), modes, discs, mcfg)
                ac_agg = aggregate(ac, modes, discs, mcfg)
                ac_rank_agg = aggregate(ra, modes, discs, mcfg)

                # ---- existing decision rules, for comparison -------------
                lv, la = np.log(rv), np.log(ra)
                fused_cell = int((lv + args.legacy_weight * la).argmax())
                order = np.argsort(-vis)[: args.topk]
                rerank_cell = int(order[int(ac[order].argmax())])
                vb = int(vis.argmax())
                far = np.hypot(cols - cols[vb], rows - rows[vb]) * res > mcfg.nms_radius_m
                vis_margin_log = float(np.log((vis[vb] + 1e-300)
                                              / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))

                # ---- A2 visual ambiguity ---------------------------------
                vm = vis[modes]
                m_diff, m_ratio = margins(vm)
                p_modes = softmax(np.log(np.clip(vm, 1e-300, None)))
                n_within = {t: int((vm >= t * vm[0]).sum()) for t in (0.5, 0.8, 0.9, 0.95)}

                # ---- A3 acoustic discriminativeness among visual modes ---
                am = ac[modes]
                a_diff, _ = margins(am)
                ar = ra[modes]
                ar_diff, _ = margins(ar)
                a_rel = relative_evidence(am / max(np.std(am), 1e-9))
                p_ac = softmax(am / max(np.std(am), 1e-9))

                # ---- A6 coverage -----------------------------------------
                cov = {f"gt_in_top{K}_cells": bool((dist[order[:K]] < 1.0).any())
                       for K in (1, 2, 3, 5, 10, 20, 50)}
                cov.update({f"gt_in_top{K}_modes": bool((dist[modes[:K]] < 1.0).any())
                            for K in (1, 2, 3, 5, 10)})

                # ---- A7 spatial smoothness of the acoustic field ---------
                near = {f"ac_rank_best_within_{r}m": float(ra[dist <= r].max())
                        if (dist <= r).any() else np.nan
                        for r in (0.25, 0.5, 1.0)}
                strong = ra >= args.peak_rank
                near["dist_to_strong_ac_peak_m"] = (
                    float(dist[strong].min()) if strong.any() else np.nan)

                q_rows.append(dict(
                    query_id=qid, collection=coll, scene=scene, pose_index=i,
                    condition=args.condition, backbone=args.backbone,
                    n_cells=len(rows), n_modes=len(modes),
                    # visual ambiguity
                    vis_margin_log=vis_margin_log,
                    mode_raw_margin=m_diff, mode_ratio=m_ratio,
                    mode_log_ratio=float(np.log(max(vm[0], 1e-300))
                                         - np.log(max(vm[1], 1e-300))) if len(vm) > 1 else np.inf,
                    mode_entropy=entropy(p_modes),
                    **{f"n_modes_ge_{int(100*t)}pct": v for t, v in n_within.items()},
                    # acoustic discriminativeness among those modes
                    ac_mode_margin=a_diff,
                    ac_rank_margin=ar_diff,
                    ac_mode_ratio=float(am[0] / am[1]) if len(am) > 1 and am[1] != 0 else np.nan,
                    ac_mode_entropy=entropy(p_ac),
                    ac_rel_best=float(a_rel.max()),
                    # ground truth
                    gt_x=float(gx), gt_y=float(gy), gt_yaw=float(gth),
                    gt_cell_ac_rank=float(ra[gt_cell]),
                    gt_cell_vis_rank=float(rv[gt_cell]),
                    gt_vis_rank_pos=int((vis > vis[gt_cell]).sum()) + 1,
                    gt_ac_rank_pos=int((ac > ac[gt_cell]).sum()) + 1,
                    gt_mode_index=int(np.argmin(dist[modes])),
                    gt_mode_dist_m=float(dist[modes].min()),
                    **cov, **near,
                    # outcomes of the existing rules
                    e_vis=float(dist[vb]), orn_vis=orn_err_deg(yaw[vb], gth),
                    e_ac=float(dist[int(ac.argmax())]),
                    e_rerank=float(dist[rerank_cell]), orn_rerank=orn_err_deg(yaw[rerank_cell], gth),
                    e_fused=float(dist[fused_cell]), orn_fused=orn_err_deg(yaw[fused_cell], gth),
                ))

                for j, mi in enumerate(modes):
                    m_rows.append(dict(
                        query_id=qid, scene=scene, collection=coll, mode=j,
                        x=int(cols[mi]), y=int(rows[mi]),
                        vis_raw=float(vis[mi]), vis_rank=float(rv[mi]),
                        ac_raw=float(ac[mi]), ac_rank=float(ra[mi]),
                        vis_log_centre=float(vis_agg["centre"][j]),
                        vis_log_max=float(vis_agg["max"][j]),
                        vis_log_lse=float(vis_agg["lse"][j]),
                        ac_centre=float(ac_agg["centre"][j]),
                        ac_max=float(ac_agg["max"][j]),
                        ac_quantile=float(ac_agg["quantile"][j]),
                        ac_lse=float(ac_agg["lse"][j]),
                        ac_rank_max=float(ac_rank_agg["max"][j]),
                        ac_rank_quantile=float(ac_rank_agg["quantile"][j]),
                        ac_rel=float(a_rel[j]),
                        dist_gt_m=float(dist[mi]), within_1m=bool(dist[mi] < 1.0),
                        dist_to_best_mode_m=float(np.hypot(cols[mi] - cols[modes[0]],
                                                           rows[mi] - rows[modes[0]]) * res),
                        yaw_err_deg=orn_err_deg(yaw[mi], gth),
                        disc_cells=int(discs[j].size)))

                for name, _, _ in TIME_RANGES:
                    s = ac_by_range[name]
                    sr_ = rank_norm(s)
                    for j, mi in enumerate(modes):
                        r_rows.append(dict(query_id=qid, scene=scene, mode=j, range_ms=name,
                                           ac_raw=float(s[mi]), ac_rank=float(sr_[mi]),
                                           dist_gt_m=float(dist[mi])))
                    # acoustic-alone behaviour of this range, once per query
                    r_rows.append(dict(query_id=qid, scene=scene, mode=-1, range_ms=name,
                                       ac_raw=float(s.max()), ac_rank=float(sr_[gt_cell]),
                                       dist_gt_m=float(dist[int(s.argmax())])))

    if not q_rows:
        print("nothing extracted")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.condition}_{args.backbone}{args.tag}"
    # written with the standard library: the two conda environments this has to
    # run in do not share a pandas, and a dependency here would silently make
    # one backbone unrunnable
    import csv
    for name, rowset in (("queries", q_rows), ("modes", m_rows), ("mode_ranges", r_rows)):
        p = args.out_dir / f"{name}_{tag}.csv"
        keys = list(rowset[0].keys())
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rowset)
        print(f"wrote {p}  ({len(rowset)} rows)")
    (args.out_dir / f"provenance_{tag}.json").write_text(
        json.dumps(stamp(inputs=inputs, extra=dict(
            checkpoint=str(ck), condition=args.condition, backbone=args.backbone,
            nms_radius_m=args.nms_radius_m, n_modes=args.n_modes,
            local_radius_m=args.local_radius_m, window_ms=args.window_ms,
            feature=args.feature, nfft=args.nfft, hop=args.hop,
            time_ranges=[list(t) for t in TIME_RANGES])), indent=2))
    print(f"{len(q_rows)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
