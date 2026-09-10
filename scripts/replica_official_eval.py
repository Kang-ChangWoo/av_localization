#!/usr/bin/env python3
"""Replica, scored the way Gibson and Structured3D are scored in the papers.

Every number this project has reported on Replica so far is recall at 1 m, which
is one row of the four that F3Loc, SemRayLoc and DisCo-FLoc all report. That row
is also the most forgiving one: at 0.1 m a method has to put the pose in the
right cell of a 0.1 m grid, and at 1 m / 30 deg it has to get the heading right
as well. A method can gain at 1 m by moving mass into roughly the right room
while getting no better at the thing the tight thresholds measure, so reporting
only 1 m can flatter a fusion rule. This runs the upstream metric set unchanged:

    recall < 0.1 m, < 0.5 m, < 1 m, and < 1 m with heading error < 30 deg

with the error definitions copied from ``f3loc/eval_observation.py``: distance in
the DESDF grid frame times 0.1, and heading wrapped to the shorter arc.

Five rows are reported per visual backbone, so the comparison the papers make is
available on the acoustic axis too:

    vision only     the backbone alone, the reproduction baseline
    audio only      the candidate grid alone, no image at all
    rerank          audio picks among vision's top-k cells
    fused           rank-normalised log-space sum, weight swept
    gated           fused where vision is unsure, vision where it is confident

Heading deserves one note. The acoustic score is one number per cell, flat over
the 36 heading bins, so adding it cannot move the per-cell heading argmax. Every
fused and reranked heading is therefore vision's own heading at the chosen cell,
exactly as upstream reads it, not an approximation. Audio alone has no heading
to give, and its 1 m / 30 deg entry is left empty rather than filled with the
first bin.

    python scripts/replica_official_eval.py --visual f3loc_mono disco_rrp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DISCO_ROOT = REPO_ROOT.parent / "DisCo-FLoc"
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--visual", nargs="+", default=["f3loc_mono", "disco_rrp"])
    p.add_argument("--f3loc-run", default="echoloc_mono_fg")
    p.add_argument("--rrp-ckpt", default=str(DISCO_ROOT / "checkpoints" / "RRP_gibson_f_best.ckpt"))
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--weights", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--fov", type=float, default=106.2602)
    p.add_argument("--V", type=int, default=11)
    p.add_argument("--per-scene", action="store_true", help="also break the table down by scene")
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


def held_out(rows: list[dict], weights, gate_q: float, fit_coll: str):
    """Fit the fusion weight and the gate threshold on one collection, report on the other.

    Both knobs are otherwise chosen by maximising the very number being
    reported, which makes every gain optimistic by an unknown amount. The two
    collections are different pose sets over the same rooms, so this holds out
    the protocol rather than the room, but it is the honest version of the
    knobs and it is what decides whether the effect is real.
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
    """The four upstream numbers, plus the median error the papers omit."""
    e = np.array([r[f"{key}_err"] for r in rows])
    o = np.array([r.get(f"{key}_orn", np.nan) for r in rows], dtype=np.float64)
    out = {"r0.1": 100 * float((e < 0.1).mean()),
           "r0.5": 100 * float((e < 0.5).mean()),
           "r1.0": 100 * float((e < 1.0).mean()),
           "median_m": float(np.median(e))}
    out["r1.0_30deg"] = None if np.isnan(o).all() else \
        100 * float(np.logical_and(e < 1.0, o < 30).mean())
    return out


def print_table(title: str, table: dict[str, dict], baseline: str = "vision only") -> None:
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
    import torchvision.transforms as T
    import tqdm

    sys.path.insert(0, str(DISCO_ROOT))
    sys.path.insert(0, str(DISCO_ROOT / "eval"))
    cwd = os.getcwd()
    os.chdir(DISCO_ROOT)
    from utils.localization_utils import get_ray_from_depth, localize
    rrp = None
    if "disco_rrp" in args.visual:
        from training.RRP_lightning_module import RRPLightningModule
        rrp = RRPLightningModule.load_from_checkpoint(
            args.rrp_ckpt, map_location="cuda").cuda().eval()
    os.chdir(cwd)

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, candidate_features, observation_feature, observation_rate, score,
    )
    from track1_core.provenance import stamp

    mono = None
    if "f3loc_mono" in args.visual:
        from track1_core.models import MonoDepthModule
        mono = MonoDepthModule.load_from_checkpoint(
            str(REPO_ROOT / "outputs" / args.f3loc_run / "mono.ckpt")).cuda().eval()

    F_W = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    tf = T.Compose([T.ToTensor(), T.Resize((256, 256), antialias=True),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    root = Path(args.dataset_root)

    recs = {v: [] for v in args.visual}
    inputs = []
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
            dt = torch.tensor(desdf["desdf"], device="cuda")
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
                img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
                s = score(cf, observation_feature(np.load(f), cfg, observation_rate(f)),
                          present, cfg)
                la = np.log(np.clip(rank_norm(s), 1e-9, None))

                gx, gy, gth = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m

                for vname in args.visual:
                    with torch.no_grad():
                        if vname == "disco_rrp":
                            ft = rrp("encode", obs_img=tf(img).unsqueeze(0).cuda())
                            pred = rrp("decoder_inference", depth_cond=ft).squeeze(0).cpu().numpy()
                        else:
                            x = img.astype(np.float64) / 255.0
                            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
                            x = torch.tensor(np.transpose(x, (2, 0, 1))[None],
                                             dtype=torch.float32).cuda()
                            pred = mono.encoder(x, None)[0].squeeze(0).float().cpu().numpy()
                    rays = torch.tensor(get_ray_from_depth(pred, V=args.V, F_W=F_W),
                                        device="cuda", dtype=torch.float32)
                    _, pdist, orns, _ = localize(dt, rays, return_np=False)
                    vis = np.asarray(pdist.cpu(), dtype=np.float64)[rows, cols]
                    # heading is read at whichever cell wins, as upstream does
                    yaw = pg.bin_to_yaw(np.asarray(orns.cpu())[rows, cols])
                    lv = np.log(np.clip(rank_norm(vis), 1e-9, None))

                    best = int(vis.argmax())
                    far = np.hypot(cols - cols[best], rows - rows[best]) \
                        * pg.grid_resolution_m > args.mode_sep_m
                    margin = float(np.log((vis[best] + 1e-300)
                                          / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))
                    order = np.argsort(-vis)[: args.topk]
                    rr = int(order[int(s[order].argmax())])
                    r = {"scene": scene, "coll": coll, "margin": margin,
                         "vision_err": float(dist[best]),
                         "vision_orn": orn_err_deg(yaw[best], gth),
                         "audio_err": float(dist[int(s.argmax())]),
                         "rerank_err": float(dist[rr]),
                         "rerank_orn": orn_err_deg(yaw[rr], gth),
                         "covered": bool((dist[order] < 1.0).any())}
                    for w in args.weights:
                        j = int((lv + w * la).argmax())
                        r[f"w{w}_err"] = float(dist[j])
                        r[f"w{w}_orn"] = orn_err_deg(yaw[j], gth)
                    recs[vname].append(r)

    if not any(recs.values()):
        print("nothing scored")
        return 1

    out = {}
    for vname, rs in recs.items():
        if not rs:
            continue
        # the fusion weight is chosen on 1 m recall, the row every paper leads
        # with, so the tighter thresholds stay an honest read of that choice
        best_w = max(args.weights, key=lambda w: np.mean([r[f"w{w}_err"] < 1 for r in rs]))
        tau = float(np.quantile([r["margin"] for r in rs], args.gate_quantile))
        for r in rs:
            gated = r["margin"] < tau
            r["gated_err"] = r[f"w{best_w}_err"] if gated else r["vision_err"]
            r["gated_orn"] = r[f"w{best_w}_orn"] if gated else r["vision_orn"]

        table = {"vision only": recalls(rs, "vision"),
                 "audio only": recalls(rs, "audio"),
                 f"rerank top-{args.topk}": recalls(rs, "rerank"),
                 f"fused (w={best_w:g})": recalls(rs, f"w{best_w}"),
                 "gated fusion": recalls(rs, "gated")}
        cov = 100 * float(np.mean([r["covered"] for r in rs]))
        print_table(f"{vname}   {len(rs)} poses, {len(args.scenes)} Replica scenes, "
                    f"envelope {args.window_ms:g} ms", table)
        print(f"  truth inside vision's top-{args.topk} {cov:.0f}% of the time, "
              f"which caps what reranking can reach")
        print(f"  {'weight':10s}" + "".join(f"{w:>9g}" for w in args.weights))
        print(f"  {'1 m':10s}" + "".join(
            f"{100*np.mean([r[f'w{w}_err'] < 1 for r in rs]):8.1f}%" for w in args.weights))
        out[vname] = {"n": len(rs), "best_weight": best_w, "gate_tau": tau,
                      "shortlist_coverage": cov, "overall": table,
                      "by_weight": {str(w): 100 * float(np.mean([r[f"w{w}_err"] < 1 for r in rs]))
                                    for w in args.weights}}

        ho = held_out(rs, args.weights, args.gate_quantile, args.collections[0])
        if ho is not None:
            rep, w_ho, tau_ho = ho
            t = {"vision only": recalls(rep, "vision"),
                 f"fused (w={w_ho:g})": recalls(rep, "hof"),
                 "gated fusion": recalls(rep, "ho")}
            print_table(f"{vname} HELD OUT: weight and gate fitted on "
                        f"{args.collections[0]}, reported on the rest "
                        f"({len(rep)} poses, w={w_ho:g}, tau={tau_ho:.3f})", t)
            out[vname]["held_out"] = {"fit_on": args.collections[0], "weight": w_ho,
                                      "tau": tau_ho, "n": len(rep), "table": t}

        if args.per_scene:
            out[vname]["per_scene"] = {}
            for scene in args.scenes:
                sub = [r for r in rs if r["scene"] == scene]
                if not sub:
                    continue
                t = {"vision only": recalls(sub, "vision"),
                     "audio only": recalls(sub, "audio"),
                     f"rerank top-{args.topk}": recalls(sub, "rerank"),
                     f"fused (w={best_w:g})": recalls(sub, f"w{best_w}"),
                     "gated fusion": recalls(sub, "gated")}
                print_table(f"{vname} / {scene}   {len(sub)} poses", t)
                out[vname]["per_scene"][scene] = t

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "replica_official_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(scenes=args.scenes, collections=args.collections,
                                 condition=args.condition, window_ms=args.window_ms,
                                 topk=args.topk, results=out,
                                 provenance=stamp(inputs=inputs)), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
