#!/usr/bin/env python3
"""Add the acoustic score to DisCo-FLoc's candidate re-ranking, and measure it.

DisCo-FLoc attacks the same failure this project's acoustic work attacks. Its
abstract names it directly: "physically distant poses share highly similar
visual-geometric features". DisCo resolves that with visual-geometric
contrastive disambiguation; sound resolves it with a measurement the camera
cannot make. The two are not competing, and this script tests whether they add.

The integration point is exact rather than approximate. DisCo's stage 1 is a Ray
Regression Predictor whose output is 40 ray distances -- the same quantity as
F3Loc's ``depth40`` -- which it pushes through the same ``get_ray_from_depth``
and ``localize`` against the same DESDF cache this project already uses. So the
visual posterior DisCo re-ranks and the one the acoustic score re-ranks are the
same object, on the same pose grid, and the acoustic term slots in beside
DisCo's own re-ranking rather than replacing anything.

Nothing in the DisCo checkout is modified. Its modules are imported from the
sibling clone, and the acoustic side is this project's existing training-free
scorer.

    python scripts/disco_audio_rerank.py --collections replica_f replica_g
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

STFT_BANDS = [(0, 500), (500, 1500), (1500, 4000)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--rrp-ckpt", default=str(DISCO_ROOT / "checkpoints" / "RRP_gibson_f_best.ckpt"))
    p.add_argument("--disco-config", default=str(DISCO_ROOT / "configs" / "paper" / "rrp_gibson.yaml"))
    p.add_argument("--n-poses", type=int, default=150)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--fov", type=float, default=106.2602, help="DisCo's Gibson default")
    p.add_argument("--V", type=int, default=11)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0],
                   help="acoustic weight in the log-space fusion; w = 0 is vision")
    p.add_argument("--visual", default="disco_rrp", choices=["disco_rrp", "f3loc_mono"],
                   help="which network produces the visual posterior; everything "
                        "downstream is identical, so switching this attributes any "
                        "difference in the acoustic gain to the visual model alone")
    p.add_argument("--f3loc-run", default="echoloc_mono_fg")
    p.add_argument("--desdf-clip", type=float, default=20.0,
                   help="DisCo truncates the range field at 20 m, F3Loc at 10")
    p.add_argument("--fusion-scale", default="rank",
                   choices=["rank", "logpost", "zscore"],
                   help="how the two scores are put on a common scale before "
                        "they are added; this changes the sign of the result")
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
    import yaml

    # DisCo's own modules, imported from the untouched sibling clone
    # DisCo's eval helpers live under eval/utils, and its eval script runs with
    # that directory importable; mirror that rather than editing the checkout
    sys.path.insert(0, str(DISCO_ROOT))
    sys.path.insert(0, str(DISCO_ROOT / "eval"))
    cwd = os.getcwd()
    os.chdir(DISCO_ROOT)          # its config references relative checkpoint paths
    from training.RRP_lightning_module import RRPLightningModule
    from utils.localization_utils import get_ray_from_depth, localize

    from scripts.margin_gated_fusion import band_envelope, featurize_band
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.events import EventMatchConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rrp = mono = None
    if args.visual == "disco_rrp":
        rrp = RRPLightningModule.load_from_checkpoint(args.rrp_ckpt, map_location=device)
        rrp = rrp.to(device).eval()
    os.chdir(cwd)
    if args.visual == "f3loc_mono":
        from track1_core.models import MonoDepthModule
        mono = MonoDepthModule.load_from_checkpoint(
            str(REPO_ROOT / "outputs" / args.f3loc_run / "mono.ckpt")).to(device).eval()
    # DisCo derives the focal ratio from the field of view; Replica's intrinsics
    # give the same 3/8 that Gibson has, so no per-dataset change is needed
    F_W = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    print(f"[disco] RRP loaded, F_W = {F_W:.4f}, V = {args.V}")

    ecfg = EventMatchConfig()
    root = Path(args.dataset_root)

    # ---- per (collection, scene): pose grid, desdf, rendered acoustic grid ----
    geo = {}
    for coll in args.collections:
        for scene in args.scenes:
            if not (root / coll / scene / "map.png").exists():
                continue
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            # DisCo truncates the range field at 20 m, F3Loc at 10; keep DisCo's
            desdf["desdf"][desdf["desdf"] > args.desdf_clip] = args.desdf_clip
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows_g, cols_g = np.nonzero(mask)

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
                pg=pg, mask=mask, rows=rows_g, cols=cols_g, rend=rend,
                desdf_t=torch.tensor(desdf["desdf"], dtype=torch.float32),
                poses=np.array([[float(v) for v in l.split()]
                                for l in open(root / coll / scene / "poses.txt") if l.strip()]))
            print(f"[geo] {coll}/{scene}: {len(rows_g)} cells, "
                  f"acoustic grid {'yes' if rend else 'no'}")

    # DisCo's own image preprocessing: 256x256, ImageNet normalisation
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Resize((256, 256), antialias=True),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    recs = []
    for (coll, scene), G in geo.items():
        rir_dir = root / "rir" / coll / args.condition / scene
        if not rir_dir.is_dir() or G["rend"] is None:
            continue
        rows_g, cols_g, pg = G["rows"], G["cols"], G["pg"]
        dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))
        dirs = [dirs[i] for i in np.linspace(0, len(dirs) - 1,
                                             min(args.n_poses, len(dirs))).astype(int)]

        for d in tqdm.tqdm(dirs, desc=f"{coll}/{scene}", leave=False):
            i = int(d.split("_")[1])
            f = rir_dir / d / "rir.npy"
            img_p = root / coll / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"
            if not f.exists() or not img_p.exists() or i >= len(G["poses"]):
                continue
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
            with torch.no_grad():
                if rrp is not None:
                    feat = rrp("encode", obs_img=tf(img).unsqueeze(0).to(device))
                    pred = rrp("decoder_inference", depth_cond=feat).squeeze(0).cpu().numpy()
                else:
                    # F3Loc's preprocessing, from utils/data_utils.py: keep the
                    # native 640x480, scale to [0, 1], then ImageNet-normalise.
                    # Dropping the normalisation costs almost everything: the
                    # monocular net scored 9.6% instead of 38% without it.
                    x = img.astype(np.float64) / 255.0
                    x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
                    x = torch.tensor(np.transpose(x, (2, 0, 1))[None],
                                     dtype=torch.float32, device=device)
                    pred = mono.encoder(x, None)[0].squeeze(0).float().cpu().numpy()
            rays = torch.tensor(get_ray_from_depth(pred, V=args.V, F_W=F_W))
            _, prob_dist, _, _ = localize(G["desdf_t"], rays, return_np=False)
            vis = np.asarray(prob_dist, dtype=np.float64)[rows_g, cols_g]

            obs_f = featurize_band(band_envelope(np.load(f), ecfg.direct_guard_samples, 1024,
                                                 64, 16, ecfg.sample_rate_hz, STFT_BANDS))
            nw = min(G["rend"]["feat"].shape[2], obs_f.shape[1])
            ac = -np.abs(G["rend"]["feat"][:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))
            ac = np.where(G["rend"]["present"], ac, ac.min() - 1.0)

            gx, gy, _ = pg.pose_metric_to_grid(G["poses"][i, :3])
            dist = np.hypot(cols_g - gx, rows_g - gy) * pg.grid_resolution_m

            best = int(vis.argmax())
            far = np.hypot(cols_g - cols_g[best], rows_g - rows_g[best]) \
                * pg.grid_resolution_m > args.mode_sep_m
            margin = float(np.log((vis[best] + 1e-300)
                                  / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))

            # Two combination rules, because they are not interchangeable.
            #
            # Replacement takes the acoustic argmax inside the visual shortlist,
            # which discards the visual ranking there. Vision's own top-1 is
            # already right about a third of the time, so replacement only wins
            # if the acoustic ordering beats that, which is a high bar.
            #
            # Weighted log fusion cannot lose in the same way: at w = 0 it is
            # exactly vision and it is continuous in w, so a small weight is
            # guaranteed not to be much worse. It is the rule this project
            # validated earlier, and the one to compare against.
            order = np.argsort(-vis)[: args.topk]
            j = int(order[int(ac[order].argmax())])
            # Which scale to fuse on is not a detail; it decides the sign of
            # the result. Neither score is a calibrated posterior: localize()
            # returns exp(-L1/40) with an arbitrary 40, and the acoustic score
            # is a negative L1 between normalised envelopes. Fusing their raw
            # logs therefore compares two arbitrary dynamic ranges. Ranking both
            # first removes exactly that arbitrariness, which is why it is the
            # better-founded choice here rather than the cruder one.
            if args.fusion_scale == "rank":
                lv = np.log(np.clip(rank_norm(vis), 1e-9, None))
                la = np.log(np.clip(rank_norm(ac), 1e-9, None))
            elif args.fusion_scale == "zscore":
                def z(u):
                    return (u - u.mean()) / max(u.std(), 1e-12)
                lv, la = z(np.log(np.clip(vis, 1e-300, None))), z(ac)
            else:                                     # logpost
                lv = np.log(np.clip(vis / max(vis.sum(), 1e-300), 1e-300, None))
                a_pos = ac - ac.min()
                la = np.log(np.clip(a_pos / max(a_pos.sum(), 1e-30), 1e-300, None))
            r = dict(coll=coll, scene=scene, margin=margin,
                     vision=float(dist[best]), rerank=float(dist[j]),
                     oracle=float(dist[order].min()), n_cand=int(len(rows_g)))
            for w in args.weights:
                r[f"fused_w{w}"] = float(dist[int((lv + w * la).argmax())])
            recs.append(r)

    if not recs:
        print("nothing scored")
        return 1

    fit = [r for r in recs if r["coll"] != args.collections[-1]] or recs
    tau = float(np.quantile([r["margin"] for r in fit], args.gate_quantile))
    for r in recs:
        r["gated"] = r["rerank"] if r["margin"] < tau else r["vision"]
        for w in args.weights:
            r[f"gated_w{w}"] = r[f"fused_w{w}"] if r["margin"] < tau else r["vision"]

    def row(label, key, subset=None):
        g = subset if subset is not None else recs
        e = np.array([x[key] for x in g])
        return dict(n=len(e), r01=float((e < .1).mean()), r05=float((e < .5).mean()),
                    r1=float((e < 1).mean()), median_m=float(np.median(e)))

    print(f"\nDisCo RRP visual posterior + training-free acoustic re-rank")
    print(f"{len(recs)} poses, {args.condition}, K={args.topk}, "
          f"gate at margin < {tau:.3f}\n")
    print(f"{'':34s} {'0.1m':>7s} {'0.5m':>7s} {'1m':>7s} {'median':>8s} {'vs vision':>10s}")
    print("-" * 80)
    base = row("v", "vision")["r1"]
    out = {}
    rows = [(f"{args.visual}, vision only", "vision"),
            ("  replacement: acoustic argmax in top-K", "rerank"),
            ("  replacement + margin gate", "gated")]
    for w in args.weights:
        rows.append((f"  fusion w={w}", f"fused_w{w}"))
    for w in args.weights:
        rows.append((f"  fusion w={w} + margin gate", f"gated_w{w}"))
    rows.append(("  oracle over the shortlist", "oracle"))
    for label, key in rows:
        m = row(label, key)
        out[key] = m
        d = "" if key == "vision" else f"{100*(m['r1']-base):+9.1f}"
        print(f"{label:34s} {100*m['r01']:6.1f}% {100*m['r05']:6.1f}% "
              f"{100*m['r1']:6.1f}% {m['median_m']:7.2f}m {d:>10s}")

    margins = np.array([r["margin"] for r in recs])
    q = np.quantile(margins, [1 / 3, 2 / 3])
    print(f"\n{'visual margin stratum':24s} {'n':>4s} {'vision':>8s} {'gated':>8s} {'gain':>7s}")
    strata = {}
    wbest = f"gated_w{args.weights[0]}"
    for lab, m in (("ambiguous", margins <= q[0]),
                   ("middle", (margins > q[0]) & (margins <= q[1])),
                   ("confident", margins > q[1])):
        g = [recs[i] for i in np.flatnonzero(m)]
        v, f_ = row("", "vision", g)["r1"], row("", wbest, g)["r1"]
        print(f"{lab:24s} {len(g):4d} {100*v:7.1f}% {100*f_:7.1f}% {100*(f_-v):+6.1f}")
        strata[lab] = dict(n=len(g), vision=v, gated=f_, gain=f_ - v)

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "disco_audio_rerank.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(n=len(recs), tau=tau, topk=args.topk,
                                 condition=args.condition, overall=out,
                                 by_margin=strata), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
