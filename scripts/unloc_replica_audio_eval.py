#!/usr/bin/env python3
"""UnLoc single-frame on Replica, with and without the acoustic term.

UnLoc reproduces its published single-frame Gibson row exactly here, so it is a
trustworthy third visual backbone, and it is the interesting one for this
project: unlike F3Loc and DisCo's ray predictor it emits a *per-ray uncertainty*
alongside the depth, and its likelihood downweights rays it does not trust,

    p(x) proportional to exp( -|| (desdf_rays - pred_rays) / scales ||_1 / lambd )

That is already a soft version of what the margin gate does at the pose level.
If sound still helps a backbone that models its own uncertainty, the gain is not
merely a correction for overconfident depth, which is the most obvious deflating
explanation for the F3Loc result.

Rows are the same five the other Replica scripts report, on the same metric set
as f3loc/eval_observation.py, so the three backbones sit on one axis.

Runs in the `unloc` environment, which is why the acoustic code is imported from
this repository rather than the scoring being duplicated:

    /opt/conda/envs/unloc/bin/python scripts/unloc_replica_audio_eval.py --per-scene
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
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--checkpoint", default=str(UNLOC_ROOT / "logs" / "unloc_gibson_vitl.ckpt"),
                   help="released Gibson weights by default, so this is zero-shot; "
                        "point it at the Replica run to remove the domain gap")
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--weights", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--orn-slice", type=int, default=36)
    p.add_argument("--per-scene", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def rank_norm(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.shape[0], dtype=np.float64)
    r[np.argsort(x)] = np.arange(x.shape[0])
    return (r + 0.5) / x.shape[0]


def orn_err_deg(pred_rad: float, gt_rad: float) -> float:
    d = (float(pred_rad) - float(gt_rad)) % (2 * np.pi)
    return min(d, 2 * np.pi - d) / np.pi * 180.0


def held_out(rows, weights, gate_q, fit_coll):
    """Fit the fusion weight and gate threshold on one collection, report on the other.

    Both knobs are otherwise chosen by maximising the number being reported.
    The two collections are different pose sets over the same rooms, so this
    holds out the protocol rather than the room, but it is the honest version.
    """
    fit = [r for r in rows if r["coll"] == fit_coll]
    rep = [r for r in rows if r["coll"] != fit_coll]
    if not fit or not rep:
        return None
    w = max(weights, key=lambda w: np.mean([r[f"w{w}_err"] < 1 for r in fit]))
    tau = float(np.quantile([r["margin"] for r in fit], gate_q))
    for r in rep:
        on = r["margin"] < tau
        r["ho_err"] = r[f"w{w}_err"] if on else r["vision_err"]
        r["ho_orn"] = r[f"w{w}_orn"] if on else r["vision_orn"]
        r["hof_err"], r["hof_orn"] = r[f"w{w}_err"], r[f"w{w}_orn"]
    return rep, w, tau


def recalls(rows: list[dict], key: str) -> dict:
    e = np.array([r[f"{key}_err"] for r in rows])
    o = np.array([r.get(f"{key}_orn", np.nan) for r in rows], dtype=np.float64)
    out = {"r0.1": 100 * float((e < 0.1).mean()),
           "r0.5": 100 * float((e < 0.5).mean()),
           "r1.0": 100 * float((e < 1.0).mean()),
           "median_m": float(np.median(e))}
    out["r1.0_30deg"] = None if np.isnan(o).all() else \
        100 * float(np.logical_and(e < 1.0, o < 30).mean())
    return out


def print_table(title: str, table: dict, baseline: str = "vision only") -> None:
    print(f"\n=== {title}")
    print(f"{'method':22s} {'0.1 m':>7s} {'0.5 m':>7s} {'1 m':>7s} "
          f"{'1m/30deg':>9s} {'median':>8s}   vs vision")
    print("-" * 78)
    base = table.get(baseline)
    for name, m in table.items():
        d = ""
        if base is not None and name != baseline:
            d = f"  {m['r1.0'] - base['r1.0']:+5.1f} @1m"
            if m["r1.0_30deg"] is not None and base["r1.0_30deg"] is not None:
                d += f" {m['r1.0_30deg'] - base['r1.0_30deg']:+5.1f} @30deg"
        od = "     --" if m["r1.0_30deg"] is None else f"{m['r1.0_30deg']:8.1f}%"
        print(f"{name:22s} {m['r0.1']:6.1f}% {m['r0.5']:6.1f}% {m['r1.0']:6.1f}% "
              f"{od} {m['median_m']:7.2f}m{d}")


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import tqdm

    sys.path.insert(0, str(UNLOC_ROOT))
    from modules.depth_net_pl import UnLocDepthModule
    from utils.localization_utils import (
        get_ray_from_depth_uncertainty, localize_uncertainty,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # the frozen Depth Anything backbone is loaded from a path relative to the
    # UnLoc checkout, so construction has to happen from there
    cwd = os.getcwd()
    os.chdir(UNLOC_ROOT)
    try:
        net = UnLocDepthModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint, strict=False).to(device).eval()
    finally:
        os.chdir(cwd)
    print(f"[unloc] {args.checkpoint}")

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, candidate_features, observation_feature, observation_rate, score,
    )
    from track1_core.provenance import stamp

    root = Path(args.dataset_root)
    recs, inputs = [], []
    for coll in args.collections:
        for scene in args.scenes:
            g = REPO_ROOT / args.grid_dir / f"{scene}.npz"
            if not g.exists() or not (root / coll / scene / "map.png").exists():
                continue
            inputs.append(g)
            sr = int(json.loads(str(np.load(g)["config"]))["sample_rate"])
            cfg = GridScoreConfig(feature="envelope", window_ms=args.window_ms,
                                  sample_rate_hz=sr,
                                  direct_guard_samples=int(round(sr * 2 / 1000)),
                                  usable_samples=int(round(sr * 128 / 1000)))
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows, cols = np.nonzero(mask)
            dt = torch.tensor(desdf["desdf"], device=device)
            cf, present = candidate_features(g, rows, cols, mask.shape, cfg)
            print(f"[geo] {coll}/{scene}: {len(rows)} cells, grid at {sr} Hz")

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rd = root / "rir" / coll / args.condition / scene
            ds_dirs = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
            picks = np.linspace(0, len(ds_dirs) - 1,
                                min(args.n_poses, len(ds_dirs))).astype(int)

            for k in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                d = ds_dirs[int(k)]
                i = int(d.split("_")[1])
                f = rd / d / "rir.npy"
                img_p = root / coll / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"
                if not f.exists() or not img_p.exists() or i >= len(poses):
                    continue
                # exactly GibsonFrameDataset's preprocessing: BGR to RGB, /255,
                # ImageNet statistics, channel-first
                img = cv2.imread(str(img_p), cv2.IMREAD_COLOR).astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img -= (0.485, 0.456, 0.406)
                img /= (0.229, 0.224, 0.225)
                x = torch.tensor(np.transpose(img, (2, 0, 1))[None],
                                 dtype=torch.float32, device=device)

                with torch.no_grad():
                    loc, scale, _, _ = net.encoder(x, None)
                pr, ps = get_ray_from_depth_uncertainty(
                    loc.squeeze(0).cpu().numpy(), scale.squeeze(0).cpu().numpy())
                _, pdist, orns, _ = localize_uncertainty(
                    dt, torch.tensor(pr, device=device),
                    torch.tensor(ps, device=device),
                    return_np=False, orn_slice=args.orn_slice)
                vis = np.asarray(pdist.cpu(), dtype=np.float64)[rows, cols]
                yaw = pg.bin_to_yaw(np.asarray(orns.cpu())[rows, cols])

                s = score(cf, observation_feature(np.load(f), cfg, observation_rate(f)),
                          present, cfg)
                la = np.log(np.clip(rank_norm(s), 1e-9, None))
                lv = np.log(np.clip(rank_norm(vis), 1e-9, None))

                gx, gy, gth = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
                best = int(vis.argmax())
                far = np.hypot(cols - cols[best], rows - rows[best]) \
                    * pg.grid_resolution_m > args.mode_sep_m
                margin = float(np.log((vis[best] + 1e-300)
                                      / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))
                order = np.argsort(-vis)[: args.topk]
                rr = int(order[int(s[order].argmax())])
                r = {"scene": scene, "coll": coll, "margin": margin,
                     "vision_err": float(dist[best]), "vision_orn": orn_err_deg(yaw[best], gth),
                     "audio_err": float(dist[int(s.argmax())]),
                     "rerank_err": float(dist[rr]), "rerank_orn": orn_err_deg(yaw[rr], gth),
                     "covered": bool((dist[order] < 1.0).any())}
                for w in args.weights:
                    j = int((lv + w * la).argmax())
                    r[f"w{w}_err"] = float(dist[j])
                    r[f"w{w}_orn"] = orn_err_deg(yaw[j], gth)
                recs.append(r)

    if not recs:
        print("nothing scored")
        return 1

    best_w = max(args.weights, key=lambda w: np.mean([r[f"w{w}_err"] < 1 for r in recs]))
    tau = float(np.quantile([r["margin"] for r in recs], args.gate_quantile))
    for r in recs:
        on = r["margin"] < tau
        r["gated_err"] = r[f"w{best_w}_err"] if on else r["vision_err"]
        r["gated_orn"] = r[f"w{best_w}_orn"] if on else r["vision_orn"]

    table = {"vision only": recalls(recs, "vision"),
             "audio only": recalls(recs, "audio"),
             f"rerank top-{args.topk}": recalls(recs, "rerank"),
             f"fused (w={best_w:g})": recalls(recs, f"w{best_w}"),
             "gated fusion": recalls(recs, "gated")}
    cov = 100 * float(np.mean([r["covered"] for r in recs]))
    print_table(f"UnLoc single-frame   {len(recs)} poses, {len(args.scenes)} Replica scenes, "
                f"envelope {args.window_ms:g} ms", table)
    print(f"  truth inside vision's top-{args.topk} {cov:.0f}% of the time")
    print(f"  {'weight':10s}" + "".join(f"{w:>9g}" for w in args.weights))
    print(f"  {'1 m':10s}" + "".join(
        f"{100*np.mean([r[f'w{w}_err'] < 1 for r in recs]):8.1f}%" for w in args.weights))

    ho = held_out(recs, args.weights, args.gate_quantile, args.collections[0])
    if ho is not None:
        rep, w_ho, tau_ho = ho
        t_ho = {"vision only": recalls(rep, "vision"),
                f"fused (w={w_ho:g})": recalls(rep, "hof"),
                "gated fusion": recalls(rep, "ho")}
        print_table(f"HELD OUT: weight and gate fitted on {args.collections[0]}, "
                    f"reported on the rest ({len(rep)} poses, w={w_ho:g}, "
                    f"tau={tau_ho:.3f})", t_ho)

    out = {"n": len(recs), "best_weight": best_w, "gate_tau": tau,
           "shortlist_coverage": cov, "overall": table,
           "by_weight": {str(w): 100 * float(np.mean([r[f"w{w}_err"] < 1 for r in recs]))
                         for w in args.weights}}
    if ho is not None:
        out["held_out"] = {"fit_on": args.collections[0], "weight": w_ho,
                           "tau": tau_ho, "n": len(rep), "table": t_ho}
    if args.per_scene:
        out["per_scene"] = {}
        for scene in args.scenes:
            sub = [r for r in recs if r["scene"] == scene]
            if not sub:
                continue
            t = {"vision only": recalls(sub, "vision"),
                 "audio only": recalls(sub, "audio"),
                 f"rerank top-{args.topk}": recalls(sub, "rerank"),
                 f"fused (w={best_w:g})": recalls(sub, f"w{best_w}"),
                 "gated fusion": recalls(sub, "gated")}
            print_table(f"UnLoc / {scene}   {len(sub)} poses", t)
            out["per_scene"][scene] = t

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "unloc_replica_audio.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(checkpoint=args.checkpoint, scenes=args.scenes,
                                 condition=args.condition, window_ms=args.window_ms,
                                 results=out, provenance=stamp(inputs=inputs)), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
