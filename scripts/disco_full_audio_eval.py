#!/usr/bin/env python3
"""Does sound still add anything on top of DisCo's own reranking stage?

Every DisCo number this project has reported on Replica used only the ray
regression predictor, which is half the method. DisCo's contribution is the
second stage: it takes the RRP posterior's top candidates, consolidates them
into SE(2) modes, crops a local floorplan patch around each one, and reweights
by the similarity between the image embedding and the map-patch embedding. On
Gibson that stage is worth +6.9 points, and it attacks exactly the failure the
acoustic score attacks, namely a candidate that looks right to a depth-based
matcher but sits in the wrong part of the floorplan.

So comparing acoustics against the ray predictor alone is comparing against the
weaker half, and any gain measured that way may simply be a gain DisCo already
collects by other means. This runs the full published pipeline on Replica and
then multiplies the acoustic term into the same score, which is the natural
place for it: DisCo already combines geometry and similarity as

    final = geo_prob * exp(alpha * similarity)

and the acoustic term enters as one more factor, exp(w * log-rank). Four rows:

    RRP only            the ray predictor, the weaker half
    RRP + DisCo         the published method
    RRP + audio         acoustics instead of DisCo's reranker
    RRP + DisCo + audio both, which is the row that decides the contribution

If the last row does not beat the second, the acoustic gain measured against RRP
alone was redundant with DisCo's own stage and should not be claimed.

    python scripts/disco_full_audio_eval.py --per-scene
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
    p.add_argument("--config", default=str(DISCO_ROOT / "configs" / "paper" / "disco_gibson.yaml"))
    p.add_argument("--rrp-ckpt", default=str(DISCO_ROOT / "checkpoints" / "RRP_gibson_f_best.ckpt"))
    p.add_argument("--disco-ckpt", default=str(DISCO_ROOT / "checkpoints" / "DisCo_gibson_f_best.ckpt"))
    # DisCo's own published defaults, unchanged
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mode-source-top-k", type=int, default=1000)
    p.add_argument("--se2-sigma-t-m", type=float, default=0.6)
    p.add_argument("--se2-sigma-theta-deg", type=float, default=30.0)
    p.add_argument("--se2-angle-weight", type=float, default=1.0)
    p.add_argument("--se2-mode-radius", type=float, default=1.0)
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--weights", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--fov", type=float, default=106.2602)
    p.add_argument("--V", type=int, default=11)
    p.add_argument("--per-scene", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def orn_err_deg(pred_rad: float, gt_rad: float) -> float:
    d = (float(pred_rad) - float(gt_rad)) % (2 * np.pi)
    return min(d, 2 * np.pi - d) / np.pi * 180.0


def recalls(rows: list[dict], key: str) -> dict:
    e = np.array([r[f"{key}_err"] for r in rows])
    o = np.array([r[f"{key}_orn"] for r in rows], dtype=np.float64)
    return {"r0.1": 100 * float((e < 0.1).mean()),
            "r0.5": 100 * float((e < 0.5).mean()),
            "r1.0": 100 * float((e < 1.0).mean()),
            "r1.0_30deg": 100 * float(np.logical_and(e < 1.0, o < 30).mean()),
            "median_m": float(np.median(e))}


def print_table(title: str, table: dict[str, dict], baseline: str) -> None:
    print(f"\n=== {title}")
    print(f"{'method':26s} {'0.1 m':>7s} {'0.5 m':>7s} {'1 m':>7s} "
          f"{'1m/30deg':>9s} {'median':>8s}   vs DisCo")
    print("-" * 82)
    base = table[baseline]
    for name, m in table.items():
        d = "" if name == baseline else \
            f"  {m['r1.0'] - base['r1.0']:+5.1f} @1m {m['r1.0_30deg'] - base['r1.0_30deg']:+5.1f} @30deg"
        print(f"{name:26s} {m['r0.1']:6.1f}% {m['r0.5']:6.1f}% {m['r1.0']:6.1f}% "
              f"{m['r1.0_30deg']:8.1f}% {m['median_m']:7.2f}m{d}")


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import torchvision.transforms as T
    import tqdm
    import yaml
    from attrdict import AttrDict

    sys.path.insert(0, str(DISCO_ROOT))
    sys.path.insert(0, str(DISCO_ROOT / "eval"))
    cwd = os.getcwd()
    os.chdir(DISCO_ROOT)
    from utils.localization_utils import get_ray_from_depth, localize
    from training.RRP_lightning_module import RRPLightningModule
    from training.DisCo_lightning_module import DisCoLocModel
    # reuse DisCo's own crop and mode consolidation rather than reimplementing.
    # That module parses argv at import time, so hide ours and let it read its
    # own defaults; the evaluation body is behind a __main__ guard and does not
    # run on import.
    argv = sys.argv
    sys.argv = ["eval_disco_model_gibson"]
    try:
        from eval_disco_model_gibson import consolidate_se2_modes, crop_local_map
    finally:
        sys.argv = argv

    config = AttrDict(yaml.safe_load(open(args.config)))
    rrp = RRPLightningModule.load_from_checkpoint(args.rrp_ckpt, map_location="cuda").cuda().eval()
    cl = DisCoLocModel.load_from_checkpoint(args.disco_ckpt, config=config,
                                            map_location="cuda").cuda().eval()
    os.chdir(cwd)
    data_config = config.get("data", {}) or {}
    crop_m = data_config.get("local_map_crop_size_meters", 5.0)
    print(f"[disco] alpha {args.alpha}, top-k {args.mode_source_top_k}, "
          f"local map {crop_m} m")

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, candidate_features, observation_feature, observation_rate, score,
    )
    from track1_core.provenance import stamp

    F_W = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    tf = T.Compose([T.ToTensor(), T.Resize((256, 256), antialias=True),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    root = Path(args.dataset_root)
    MAP_RES = 0.01
    STRIDE = 10

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
            scene_map = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, scene_map.shape)
            mask = valid_pose_mask(scene_map, pg)
            rows, cols = np.nonzero(mask)
            dt = torch.tensor(desdf["desdf"], device="cuda")
            cf, present = candidate_features(g, rows, cols, mask.shape, cfg)
            # acoustic score lives on (rows, cols); scatter it back onto the grid
            # so it can multiply DisCo's candidates wherever they land
            cell_of = -np.ones(mask.shape, dtype=np.int64)
            cell_of[rows, cols] = np.arange(len(rows))
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
                obs = tf(img).unsqueeze(0).cuda()

                s = score(cf, observation_feature(np.load(f), cfg, observation_rate(f)),
                          present, cfg)
                # rank-normalise over the valid cells, then place on the grid
                r = np.empty(len(s)); r[np.argsort(s)] = np.arange(len(s))
                la_cells = np.log((r + 0.5) / len(s))

                with torch.no_grad():
                    ft = rrp("encode", obs_img=obs)
                    pred = rrp("decoder_inference", depth_cond=ft).squeeze(0).cpu().numpy()
                rays = torch.tensor(get_ray_from_depth(pred, V=args.V, F_W=F_W),
                                    device="cuda", dtype=torch.float32)
                # DisCo's mode consolidation allocates its working tensors on
                # the CPU, so the posterior stays there through that stage
                _, prob_dist, orientations, _ = localize(dt, rays, return_np=False)

                gx, gy, gth = pg.pose_metric_to_grid(poses[i, :3])

                def err(px: int, py: int, orn_idx: int) -> tuple[float, float]:
                    return (float(np.hypot(px - gx, py - gy) * pg.grid_resolution_m),
                            orn_err_deg(orn_idx / 36 * 2 * np.pi, gth))

                H_d, W_d = prob_dist.shape
                flat = prob_dist.flatten()
                # RRP alone: plain argmax of the posterior, as upstream reads it
                gi = int(flat.argmax())
                rec = {"scene": scene, "coll": coll}
                rec["rrp_err"], rec["rrp_orn"] = err(gi % W_d, gi // W_d,
                                                     int(orientations.flatten()[gi]))

                tv, ti = torch.topk(flat, k=min(args.mode_source_top_k, flat.numel()))
                tv, ti, _, _ = consolidate_se2_modes(
                    tv, ti, orientations, width=W_d,
                    meters_per_cell=pg.grid_resolution_m,
                    sigma_t_m=args.se2_sigma_t_m,
                    sigma_theta_deg=args.se2_sigma_theta_deg,
                    angle_weight=args.se2_angle_weight,
                    mode_radius=args.se2_mode_radius)
                ty, tx = (ti // W_d).cpu().numpy(), (ti % W_d).cpu().numpy()
                orn = orientations[ty, tx].cpu().numpy().astype(int)

                lmaps = []
                for j in range(len(ti)):
                    lm = crop_local_map(scene_map,
                                        tx[j] * STRIDE + desdf["l"],
                                        ty[j] * STRIDE + desdf["t"],
                                        orn[j] / 36 * 2 * np.pi,
                                        crop_size_meters=crop_m, res=MAP_RES)
                    lmaps.append(torch.from_numpy(lm).float().unsqueeze(0) / 255.0)
                with torch.no_grad():
                    ie = cl.encode_image(obs)
                    sim = cl.score_candidates(ie, torch.stack(lmaps).cuda())
                geo = tv.cuda()
                disco = geo * torch.exp(sim * args.alpha)

                # the acoustic factor for each surviving candidate
                ac = la_cells[cell_of[ty, tx]]
                ac[cell_of[ty, tx] < 0] = la_cells.min()
                ac_t = torch.tensor(ac, dtype=geo.dtype, device=geo.device)

                def take(sc, tag):
                    j = int(sc.argmax())
                    rec[f"{tag}_err"], rec[f"{tag}_orn"] = err(tx[j], ty[j], orn[j])

                take(disco, "disco")
                for w in args.weights:
                    take(geo * torch.exp(w * ac_t), f"rrpaud{w}")
                    take(disco * torch.exp(w * ac_t), f"both{w}")
                recs.append(rec)

    if not recs:
        print("nothing scored")
        return 1

    R = lambda key: np.mean([r[f"{key}_err"] < 1 for r in recs])
    bw_r = max(args.weights, key=lambda w: R(f"rrpaud{w}"))
    bw_b = max(args.weights, key=lambda w: R(f"both{w}"))
    table = {"RRP only": recalls(recs, "rrp"),
             "RRP + DisCo (published)": recalls(recs, "disco"),
             f"RRP + audio (w={bw_r:g})": recalls(recs, f"rrpaud{bw_r}"),
             f"RRP + DisCo + audio (w={bw_b:g})": recalls(recs, f"both{bw_b}")}
    print_table(f"{len(recs)} poses, {len(args.scenes)} Replica scenes, "
                f"envelope {args.window_ms:g} ms", table, "RRP + DisCo (published)")
    print(f"  {'weight':22s}" + "".join(f"{w:>9g}" for w in args.weights))
    print(f"  {'RRP + audio, 1 m':22s}" + "".join(f"{100*R(f'rrpaud{w}'):8.1f}%" for w in args.weights))
    print(f"  {'both, 1 m':22s}" + "".join(f"{100*R(f'both{w}'):8.1f}%" for w in args.weights))

    out = {"n": len(recs), "overall": table,
           "best_weight_rrp_audio": bw_r, "best_weight_both": bw_b,
           "by_weight": {"rrp_audio": {str(w): 100 * float(R(f"rrpaud{w}")) for w in args.weights},
                         "both": {str(w): 100 * float(R(f"both{w}")) for w in args.weights}}}
    if args.per_scene:
        out["per_scene"] = {}
        for scene in args.scenes:
            sub = [r for r in recs if r["scene"] == scene]
            if not sub:
                continue
            t = {"RRP only": recalls(sub, "rrp"),
                 "RRP + DisCo (published)": recalls(sub, "disco"),
                 f"RRP + audio (w={bw_r:g})": recalls(sub, f"rrpaud{bw_r}"),
                 f"RRP + DisCo + audio (w={bw_b:g})": recalls(sub, f"both{bw_b}")}
            print_table(f"{scene}   {len(sub)} poses", t, "RRP + DisCo (published)")
            out["per_scene"][scene] = t

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "disco_full_audio_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(scenes=args.scenes, condition=args.condition,
                                 alpha=args.alpha, window_ms=args.window_ms,
                                 results=out, provenance=stamp(inputs=inputs)), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
