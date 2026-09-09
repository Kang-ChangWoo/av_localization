#!/usr/bin/env python3
"""Does sound still add anything once the visual method is a recent one?

F3Loc is the baseline this project was built on, and on the Replica scenes it
reaches about 20% at 1 m in office_4. DisCo-FLoc is stronger on its own data,
and its own contribution is a *reranking* stage worth +6.9 points over its ray
predictor, which is structurally the same operation the acoustic score performs.
So the question is not whether sound beats a weak visual model, it is whether it
still contributes on top of a method that already attacks the same failure.

Both visual sources are run through one pipeline: image to 40 rays, rays to a
posterior over the shared pose grid, then a rank-normalised log-space fusion
with the acoustic score. The acoustic weight is swept rather than fixed, because
a weight tuned for a weak visual model is the wrong weight for a strong one and
a single value cannot answer the question honestly.

    python scripts/sota_audio_fusion.py --scenes office_4
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
    p.add_argument("--scenes", nargs="+", default=["office_4"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--visual", nargs="+", default=["f3loc_mono", "disco_rrp"])
    p.add_argument("--f3loc-run", default="echoloc_mono_fg")
    p.add_argument("--rrp-ckpt", default=str(DISCO_ROOT / "checkpoints" / "RRP_gibson_f_best.ckpt"))
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--weights", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--n-poses", type=int, default=150)
    p.add_argument("--fov", type=float, default=106.2602)
    p.add_argument("--V", type=int, default=11)
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
        rrp = RRPLightningModule.load_from_checkpoint(args.rrp_ckpt, map_location="cuda").cuda().eval()
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
                gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
                la = np.log(np.clip(rank_norm(s), 1e-9, None))

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
                    _, pdist, _, _ = localize(dt, rays, return_np=False)
                    vis = np.asarray(pdist.cpu(), dtype=np.float64)[rows, cols]
                    lv = np.log(np.clip(rank_norm(vis), 1e-9, None))
                    order = np.argsort(-vis)[: args.topk]
                    r = dict(vision=float(dist[int(vis.argmax())]),
                             acoustic=float(dist[int(s.argmax())]),
                             rerank=float(dist[order][int(s[order].argmax())]),
                             covered=bool((dist[order] < 1.0).any()))
                    for w in args.weights:
                        r[f"w{w}"] = float(dist[int((lv + w * la).argmax())])
                    recs[vname].append(r)

    if not any(recs.values()):
        print("nothing scored")
        return 1

    out = {}
    for vname, rs in recs.items():
        if not rs:
            continue
        R = lambda k: 100 * float(np.mean([x[k] < 1.0 for x in rs]))
        best_w = max(args.weights, key=lambda w: R(f"w{w}"))
        out[vname] = dict(n=len(rs), vision=R("vision"), acoustic=R("acoustic"),
                          rerank=R("rerank"), coverage=100 * float(np.mean([x["covered"] for x in rs])),
                          by_weight={str(w): R(f"w{w}") for w in args.weights},
                          best_weight=best_w, best=R(f"w{best_w}"))
        m = out[vname]
        print(f"\n=== {vname}   {m['n']} poses, {args.scenes}")
        print(f"  vision alone                {m['vision']:5.1f}%")
        print(f"  acoustic alone              {m['acoustic']:5.1f}%")
        print(f"  acoustic reranks vision top-{args.topk}  {m['rerank']:5.1f}%"
              f"   (truth in that shortlist {m['coverage']:.0f}% of the time)")
        print(f"  {'fusion weight':22s} {'recall@1m':>10s}")
        for w in args.weights:
            mark = "  <-- best" if w == best_w else ("  (vision only)" if w == 0 else "")
            print(f"    w = {w:<18g} {R(f'w{w}'):9.1f}%{mark}")
        print(f"  best fusion beats vision by {m['best'] - m['vision']:+.1f} and "
              f"acoustic alone by {m['best'] - m['acoustic']:+.1f}")

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "sota_audio_fusion.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(scenes=args.scenes, condition=args.condition,
                                 window_ms=args.window_ms, results=out,
                                 provenance=stamp(inputs=inputs)), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
