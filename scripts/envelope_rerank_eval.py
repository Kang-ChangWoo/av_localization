#!/usr/bin/env python3
"""Rerank vision's candidates with the acoustic feature that actually resolves time.

The training-free reranking in this project stopped working when the acoustic
feature was changed, and the reason is time resolution rather than anything
about vision or the fusion rule. The original feature summed squared samples in
disjoint 0.5 ms windows, so one arrival lands in one window and 0.5 ms is 17 cm
of path. It reached GT percentile 96.6 on office_4. The replacement analysed the
same window with a 64-sample Hann STFT, which smears every frame over 8 ms
whatever the hop, and 8 ms is 2.7 m of path, or 27 cells of a 0.1 m pose grid.
That feature scores at chance.

So this sweeps the envelope window from 0.125 ms to 4 ms with everything else
fixed, and reports both what the acoustic score does alone and what it does to
vision's ranking. If the diagnosis is right the curve should peak near 0.5 ms
and collapse toward the STFT's behaviour as the window grows.

    python scripts/envelope_rerank_eval.py --windows 0.125 0.25 0.5 1 2 4
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--run-name", default="echoloc_mono_fg")
    p.add_argument("--windows", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--stft-control", action="store_true", default=True,
                   help="also score the banded STFT, as the thing being replaced")
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--fuse-weight", type=float, default=0.25)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--grid-dir", default="outputs/acoustic_grid",
                   help="acoustic_grid_v2 holds the re-render that matches the "
                        "shipped recordings: 48 kHz, diffraction to order 10")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
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
    import torch
    import tqdm

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, candidate_features, observation_feature, observation_rate, score,
    )
    from track1_core.models import MonoDepthModule
    from track1_core.provenance import stamp
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    import json as _json
    probe = REPO_ROOT / args.grid_dir / f"{args.scenes[0]}.npz"
    sr = int(_json.loads(str(np.load(probe)["config"]))["sample_rate"])
    # guard and window are 2 ms and 128 ms whatever the rate; storing them as
    # sample counts is what silently broke when the dataset went to 48 kHz
    base = dict(sample_rate_hz=sr, direct_guard_samples=int(round(sr * 2 / 1000)),
                usable_samples=int(round(sr * 128 / 1000)))
    variants = [(f"envelope {w:g} ms",
                 GridScoreConfig(feature="envelope", window_ms=w, **base))
                for w in args.windows]
    if args.stft_control:
        variants.append(("stft 3 bands (nfft 64)",
                         GridScoreConfig(feature="stft_band", **base)))
    print(f"[sweep] grid {args.grid_dir} at {sr} Hz, guard "
          f"{base['direct_guard_samples']} samples = 2 ms, window "
          f"{base['usable_samples']} = 128 ms")
    print(f"[sweep] {len(variants)} features; a window of w ms is "
          f"{343.0 * min(args.windows) / 1000 * 100:.1f} cm of path at the finest")

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mono = MonoDepthModule.load_from_checkpoint(
        str(REPO_ROOT / "outputs" / args.run_name / "mono.ckpt")).to(device).eval()

    recs = {lab: [] for lab, _ in variants}
    vis_rec, inputs = [], []

    for coll in args.collections:
        for scene in args.scenes:
            g = REPO_ROOT / args.grid_dir / f"{scene}.npz"
            if not g.exists() or not (root / coll / scene / "map.png").exists():
                continue
            inputs.append(g)
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows, cols = np.nonzero(mask)
            dt = torch.tensor(desdf["desdf"], device=device)
            cand = {lab: candidate_features(g, rows, cols, mask.shape, c)
                    for lab, c in variants}
            print(f"[geo] {coll}/{scene}: {len(rows)} cells, {len(cand)} features built")

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rd = root / "rir" / coll / args.condition / scene
            dsdir = str(root / coll)
            ds = GridSeqDataset(dsdir, [scene], L=3, depth_dir=dsdir, depth_suffix="depth40")
            picks = np.linspace(0, len(ds) - 1, min(args.n_poses, len(ds))).astype(int)

            for chunk in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                pi = int(chunk) * 4 + 3
                f = rd / f"pose_{pi:05d}" / "rir.npy"
                if not f.exists() or pi >= len(poses):
                    continue
                d = ds[int(chunk)]
                with torch.no_grad():
                    pred, _, _ = mono.encoder(
                        torch.tensor(d["ref_img"], device=device).unsqueeze(0), None)
                rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                    device=device, dtype=torch.float32)
                _, prob_dist, _, _ = localize(dt, rays)
                vis = np.asarray(prob_dist, dtype=np.float64)[rows, cols]

                gx, gy, _ = pg.pose_metric_to_grid(poses[pi, :3])
                dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
                gt = int(dist.argmin())
                best = int(vis.argmax())
                far = np.hypot(cols - cols[best], rows - rows[best]) \
                    * pg.grid_resolution_m > args.mode_sep_m
                margin = float(np.log((vis[best] + 1e-300)
                                      / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))
                lv = np.log(np.clip(rank_norm(vis), 1e-9, None))
                order = np.argsort(-vis)[: args.topk]
                vis_rec.append(dict(err=float(dist[best]), margin=margin, n=len(rows)))

                rir, rate = np.load(f), observation_rate(f)
                for lab, cfg in variants:
                    cf, present = cand[lab]
                    s = score(cf, observation_feature(rir, cfg, rate), present, cfg)
                    la = np.log(np.clip(rank_norm(s), 1e-9, None))
                    recs[lab].append(dict(
                        alone=float(dist[int(s.argmax())]),
                        rank=int((s > s[gt]).sum()) + 1,
                        fused=float(dist[int((lv + args.fuse_weight * la).argmax())]),
                        rerank=float(dist[order][int(s[order].argmax())]),
                        margin=margin, vis=float(dist[best])))

    if not vis_rec:
        print("nothing scored")
        return 1
    margins = np.array([r["margin"] for r in vis_rec])
    tau = float(np.quantile(margins, args.gate_quantile))
    v1 = float(np.mean([r["err"] < 1 for r in vis_rec]))
    nc = float(np.mean([r["n"] for r in vis_rec]))

    print(f"\n{len(vis_rec)} poses, {nc:.0f} candidates each, vision alone "
          f"{100*v1:.1f}% at 1 m, gate at margin < {tau:.3f}")
    print(f"chance GT rank is {nc/2:.0f}\n")
    print(f"{'feature':26s} {'alone':>6s} {'GTrank':>7s} {'pct':>5s} "
          f"{'rerank':>7s} {'fused':>7s} {'gated':>7s} {'vs vis':>8s}")
    print("-" * 84)
    out = {"vision": v1, "candidates": nc, "n": len(vis_rec), "tau": tau}
    for lab, _ in variants:
        g = recs[lab]
        if not g:
            continue
        a = np.mean([x["alone"] < 1 for x in g])
        rk = float(np.median([x["rank"] for x in g]))
        rr = np.mean([x["rerank"] < 1 for x in g])
        fu = np.mean([x["fused"] < 1 for x in g])
        ga = np.mean([(x["fused"] if x["margin"] < tau else x["vis"]) < 1 for x in g])
        out[lab] = dict(alone=float(a), gt_rank_median=rk,
                        gt_percentile=float(100 * (1 - rk / nc)),
                        rerank=float(rr), fused=float(fu), gated=float(ga),
                        gain=float(ga - v1))
        print(f"{lab:26s} {100*a:5.1f}% {rk:7.0f} {100*(1-rk/nc):4.1f} "
              f"{100*rr:6.1f}% {100*fu:6.1f}% {100*ga:6.1f}% {100*(ga-v1):+7.1f}")

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "envelope_rerank_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(condition=args.condition, results=out,
                                 provenance=stamp(inputs=inputs)), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
