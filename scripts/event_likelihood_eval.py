#!/usr/bin/env python3
"""Ablation: symmetric envelope L1 against one-sided geometric event matching.

Everything scored here predicts its candidate evidence from the 2D floorplan
alone, with no rendering, no learning, and no fitting. The observation is a real
SoundSpaces recording made on the *scanned* mesh, so it carries the furniture,
clutter, and vertical geometry the floorplan does not have. That gap is the
whole point: the question is which comparison still ranks the true pose when the
model is a floorplan and the room is not.

Candidate geometry is traced once per scene and reused for every observation,
because the paths depend on the floorplan and the candidate position, never on
the recording. The delay-only score is a function of position alone, since the
source sits at the array centre, so it is computed once per cell and broadcast
across yaw rather than recomputed 36 times.

    python scripts/event_likelihood_eval.py --scenes office_4 --n-poses 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from track1_core.likelihood.events import vertical_nuisance_cutoff_ms  # noqa: E402

# Worst-case first-order floor/ceiling arrival for a 1.25 m mounting height and
# ordinary ceilings. Derived from geometry, not chosen by ranking performance.
CUTOFF_MS = vertical_nuisance_cutoff_ms()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--condition", default="raw_scan_open",
                   help="raw_scan_open is the real room; floorplan_closed is the "
                        "matched-domain control")
    p.add_argument("--n-poses", type=int, default=60)
    p.add_argument("--n-rays", type=int, default=72)
    p.add_argument("--max-order", type=int, default=1)
    p.add_argument("--vertical-order", type=int, default=0)
    p.add_argument("--stride", type=int, default=2,
                   help="candidate cells sampled per axis; 1 scores every valid cell")
    p.add_argument("--modes", nargs="+", default=None, help="subset of the ablation to run")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# (label, kwargs applied to EventMatchConfig, extra flags)
def ablation_table(args) -> list[tuple[str, dict, dict]]:
    """The ablation, with one physically-chosen default per row.

    No row's parameters were selected by maximising recall on these poses. The
    top-fraction row exists precisely to report that sensitivity rather than
    hide it.
    """
    rows = [
        ("l1 (baseline, symmetric)", {}, {"score": "l1"}),
        ("asymmetric dense energy", {}, {"score": "asym"}),
        ("event match, full support", {"top_fraction": 1.0}, {"score": "event"}),
        ("event match, top 0.6", {"top_fraction": 0.6}, {"score": "event"}),
        ("event match, top 0.4", {"top_fraction": 0.4}, {"score": "event"}),
        ("event match, top 0.8", {"top_fraction": 0.8}, {"score": "event"}),
        ("event match + corner NLOS", {"top_fraction": 0.6}, {"score": "event", "corner": True}),
        # 'uniform' and 'solid_angle' are deliberately absent as separate rows:
        # the aggregation is a weighted mean, so multiplying every weight by the
        # constant 1/n_rays cancels exactly and the two are the same estimator.
        # Only an amplitude-shaped confidence can actually change the ranking.
        ("event match, 1/r^2 conf", {"top_fraction": 0.6, "confidence_mode": "inverse_square"},
         {"score": "event"}),
        ("event match, no uniqueness", {"top_fraction": 0.6, "uniqueness": "none"},
         {"score": "event"}),
        ("event match, 2 ms bins", {"top_fraction": 0.6, "profile_resolution_ms": 2.0,
                                    "min_peak_separation_ms": 4.0}, {"score": "event"}),
        ("event match, tol 0.5 ms", {"top_fraction": 0.6, "tau_tolerance_ms": 0.5},
         {"score": "event"}),
        ("event match, tol 2.0 ms", {"top_fraction": 0.6, "tau_tolerance_ms": 2.0},
         {"score": "event"}),
        # decay compensation recovers late observed arrivals but triples the
        # observed event count, which is the density failure again
        ("event match, decay comp", {"top_fraction": 0.6, "decay_window_ms": 6.0},
         {"score": "event"}),
        ("event match, decay + full", {"top_fraction": 1.0, "decay_window_ms": 6.0},
         {"score": "event"}),
        ("event match, 20 dB range", {"top_fraction": 0.6, "peak_dynamic_range_db": 20.0},
         {"score": "event"}),
        ("event match, 6 dB range", {"top_fraction": 0.6, "peak_dynamic_range_db": 6.0},
         {"score": "event"}),
        # The diagnostic showed the strongest observed arrivals sit at 6-10 ms
        # in every pose, where the floor and ceiling bounces land, while the 2D
        # wall returns move with the room. These rows remove that window.
        ("event match, no vertical window", {"top_fraction": 0.6, "min_delay_ms": CUTOFF_MS},
         {"score": "event"}),
        ("asym dense, no vertical window", {"min_delay_ms": CUTOFF_MS}, {"score": "asym"}),
        ("event match, no vert + corner", {"top_fraction": 0.6, "min_delay_ms": CUTOFF_MS},
         {"score": "event", "corner": True}),
        ("event match, no vert, top 1.0", {"top_fraction": 1.0, "min_delay_ms": CUTOFF_MS},
         {"score": "event"}),
        ("event match, no vert, tol 2ms", {"top_fraction": 0.6, "min_delay_ms": CUTOFF_MS,
                                           "tau_tolerance_ms": 2.0}, {"score": "event"}),
    ]
    if args.modes:
        rows = [r for r in rows if any(m in r[0] for m in args.modes)]
    return rows


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.acoustic import (
        AcousticProxyConfig, observed_envelope, trace_echoes,
    )
    from track1_core.likelihood.corners import (
        CornerConfig, corner_events, corner_remote_distances, extract_corners,
    )
    from track1_core.likelihood.events import (
        EventMatchConfig, asymmetric_energy_score, event_match_score,
        observed_events, observed_profile, pad_event_sets,
        predicted_events_from_echoes, pooled_observed_profile,
    )

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = args.scenes or sorted(os.listdir(root / "desdf"))
    rows = ablation_table(args)
    results: dict[str, dict] = {}
    counts: dict[str, list] = {"predicted_events": [], "observed_events": [], "corner_events": []}

    for scene in scenes:
        if not (root / args.collection / scene / "map.png").exists():
            continue
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        rows_g, cols_g = np.nonzero(mask)
        sel = (rows_g % args.stride == 0) & (cols_g % args.stride == 0)
        rows_g, cols_g = rows_g[sel], cols_g[sel]
        if rows_g.size == 0:
            continue
        # map.png free space is exactly 255; anything else is an obstacle
        free = torch.tensor(occ == 255, device=device)
        origins_px = pg.grid_to_map(np.stack([cols_g, rows_g], axis=1))
        origins_t = torch.tensor(origins_px, dtype=torch.float32, device=device)

        print(f"[{scene}] {len(rows_g)} candidate cells (stride {args.stride})")

        # ---- floorplan geometry, traced once and reused for every recording
        t0 = time.time()
        acfg = AcousticProxyConfig(n_rays=args.n_rays, max_order=args.max_order,
                                   vertical_order=args.vertical_order)
        echoes = trace_echoes(free, origins_t, acfg, pg.map_resolution_m)
        echoes_np = {k: v.cpu().numpy() for k, v in echoes.items()}
        trace_s = time.time() - t0
        print(f"[{scene}] traced order<={args.max_order} in {trace_s:.1f}s")

        # ---- the envelope baseline needs a synthesised proxy; the event scores
        # ---- do not, which is itself part of the result
        from track1_core.likelihood.acoustic import synthesize_proxy
        t0 = time.time()
        yaw0 = torch.zeros(1, device=device)
        proxy_env = synthesize_proxy(echoes, origins_t, yaw0, pg, acfg)[:, 0]   # (N, 6, W)
        synth_s = time.time() - t0

        geom_cache: dict[str, tuple] = {}
        corner_cache: dict | None = None

        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene
        if not rir_dir.is_dir():
            print(f"[{scene}] no {args.condition} recordings, skipping")
            continue
        dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))
        dirs = [dirs[i] for i in np.linspace(0, len(dirs) - 1,
                                             min(args.n_poses, len(dirs))).astype(int)]

        obs_list, gt_list = [], []
        for d in dirs:
            f = rir_dir / d / "rir.npy"
            i = int(d.split("_")[1])
            if not f.exists() or i >= len(poses):
                continue
            gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
            obs_list.append(np.load(f))
            gt_list.append((gx, gy))
        if not obs_list:
            continue
        print(f"[{scene}] {len(obs_list)} recordings ({args.condition})")

        for label, cfg_kw, flags in rows:
            ecfg = EventMatchConfig(**cfg_kw)
            key = json.dumps(cfg_kw, sort_keys=True)
            t0 = time.time()

            if flags["score"] == "l1":
                # existing baseline, unchanged: normalise both envelopes, L1
                env = proxy_env.cpu().numpy()
                env = env / np.clip(env.sum(axis=(1, 2), keepdims=True), 1e-30, None)
            else:
                if key not in geom_cache:
                    ev = predicted_events_from_echoes(echoes_np, ecfg, origins_px,
                                                      pg.map_resolution_m, args.max_order)
                    geom_cache[key] = pad_event_sets(ev)
                pd_, pw_, pk_ = geom_cache[key]
                if flags.get("corner"):
                    if corner_cache is None:
                        ccfg = CornerConfig()
                        fn = free.cpu().numpy()
                        cpts = extract_corners(fn, pg.map_resolution_m, ccfg)
                        rho, ang = corner_remote_distances(fn, cpts, pg.map_resolution_m, ccfg)
                        cd, cw, ck = corner_events(fn, origins_px, cpts, rho, ang,
                                                   pg.map_resolution_m, ccfg, ecfg)
                        corner_cache = dict(d=cd, w=cw, k=ck, n=cpts.shape[0])
                        counts["corner_events"].append(float(np.isfinite(cd).sum(1).mean()))
                        print(f"[{scene}] {cpts.shape[0]} corners, "
                              f"{counts['corner_events'][-1]:.1f} NLOS paths per candidate")
                    pd_ = np.concatenate([pd_, corner_cache["d"]], axis=1)
                    pw_ = np.concatenate([pw_, corner_cache["w"]], axis=1)

            errs, pcts, ranks, margins = [], [], [], []
            for obs, (gx, gy) in zip(obs_list, gt_list):
                if flags["score"] == "l1":
                    o = observed_envelope(obs, acfg)
                    o = o / max(o.sum(), 1e-30)
                    score = -np.abs(env - o[None]).sum(axis=(1, 2))
                elif flags["score"] == "asym":
                    t, pooled = pooled_observed_profile(obs, ecfg)
                    score = asymmetric_energy_score(pd_, pw_, t, pooled, ecfg)
                else:
                    oe = observed_events(obs, ecfg, with_ring=bool(flags.get("ring")))
                    counts["observed_events"].append(len(oe))
                    score = event_match_score(pd_, pw_, oe, ecfg)

                gt = int(np.argmin((cols_g - gx) ** 2 + (rows_g - gy) ** 2))
                b = int(score.argmax())
                errs.append(float(np.hypot(cols_g[b] - gx, rows_g[b] - gy) * pg.grid_resolution_m))
                pcts.append(float((score < score[gt]).mean()) * 100)
                ranks.append(int((score > score[gt]).sum()) + 1)
                # margin between GT and the best candidate at least 1 m away
                far = np.hypot(cols_g - gx, rows_g - gy) * pg.grid_resolution_m > 1.0
                margins.append(float(score[gt] - score[far].max()) if far.any() else 0.0)

            if flags["score"] != "l1" and key in geom_cache:
                counts["predicted_events"].append(float((geom_cache[key][1] > 0).sum(1).mean()))
            errs = np.array(errs)
            r = results.setdefault(label, dict(scenes={}, wall_s=0.0))
            r["scenes"][scene] = dict(
                n=len(errs), candidates=int(len(rows_g)),
                gt_percentile=float(np.mean(pcts)), gt_rank_median=float(np.median(ranks)),
                gt_rank_mean=float(np.mean(ranks)),
                top10pct=float(np.mean(np.array(pcts) >= 90.0)),
                recall_1m=float((errs < 1).mean()), recall_2m=float((errs < 2).mean()),
                err_median_m=float(np.median(errs)), margin_mean=float(np.mean(margins)))
            r["wall_s"] += time.time() - t0

    if not results:
        print("nothing scored")
        return 1

    def agg(label, field):
        v = [s[field] for s in results[label]["scenes"].values()]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{'score mode':32s} {'GT_pct':>7s} {'GTrank':>7s} {'top10%':>7s} "
          f"{'<1m':>6s} {'<2m':>6s} {'margin':>9s} {'sec':>7s}")
    print("-" * 92)
    for label, _, _ in rows:
        if label not in results:
            continue
        print(f"{label:32s} {agg(label,'gt_percentile'):6.1f} "
              f"{agg(label,'gt_rank_median'):7.0f} {100*agg(label,'top10pct'):6.0f}% "
              f"{100*agg(label,'recall_1m'):5.0f}% {100*agg(label,'recall_2m'):5.0f}% "
              f"{agg(label,'margin_mean'):+9.4f} {results[label]['wall_s']:7.1f}")
    print("\nGT_pct 50 = chance. margin is GT score minus the best candidate over "
          "1 m away;\npositive means the truth is genuinely on top, not merely close.")
    for k, v in counts.items():
        if v:
            print(f"{k}: {np.mean(v):.1f} mean")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / \
        f"event_likelihood_{args.collection}_{args.condition}_o{args.max_order}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        collection=args.collection, condition=args.condition, max_order=args.max_order,
        vertical_order=args.vertical_order, stride=args.stride, n_rays=args.n_rays,
        counts={k: (float(np.mean(v)) if v else None) for k, v in counts.items()},
        results=results), indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
