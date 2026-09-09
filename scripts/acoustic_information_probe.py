#!/usr/bin/env python3
"""Does the acoustic score carry information inside the visual shortlist?

Every fusion result so far confounds two questions: whether sound knows anything
useful about which visual candidate is right, and whether the particular way the
two scores were combined preserved it. This measures the first alone, with no
fusion and no weight to tune.

Take vision's top-K candidates and ask where the true pose sits when they are
ordered by acoustic score. If sound is uninformative there, the truth lands at a
uniformly random rank, mean (K+1)/2. Below that, sound carries information the
visual posterior does not already have; above it, sound is actively misleading.
The same statistic under vision's own ordering is reported next to it, so the
two can be read against each other rather than against intuition.

Two scale diagnostics come with it, because they decide what a fusion weight can
even mean: the spread of each log score over the shortlist. Rank normalisation
flattens the visual spread to about 0.01 while leaving the acoustic one near 1,
which is why a nominal weight of 0.1 behaved as full replacement.

    python scripts/acoustic_information_probe.py --visual f3loc_mono
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
    p.add_argument("--visual", default="f3loc_mono", choices=["f3loc_mono", "disco_rrp"])
    p.add_argument("--f3loc-run", default="echoloc_mono_fg")
    p.add_argument("--rrp-ckpt", default=str(DISCO_ROOT / "checkpoints" / "RRP_gibson_f_best.ckpt"))
    p.add_argument("--desdf-clip", type=float, default=10.0)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--n-poses", type=int, default=150)
    p.add_argument("--fov", type=float, default=106.2602)
    p.add_argument("--V", type=int, default=11)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import tqdm

    sys.path.insert(0, str(DISCO_ROOT))
    sys.path.insert(0, str(DISCO_ROOT / "eval"))
    cwd = os.getcwd()
    os.chdir(DISCO_ROOT)
    from utils.localization_utils import get_ray_from_depth, localize
    rrp = None
    if args.visual == "disco_rrp":
        from training.RRP_lightning_module import RRPLightningModule
        rrp = RRPLightningModule.load_from_checkpoint(args.rrp_ckpt, map_location="cuda").cuda().eval()
    os.chdir(cwd)

    from scripts.margin_gated_fusion import band_envelope, featurize_band
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.events import EventMatchConfig

    mono = None
    if args.visual == "f3loc_mono":
        from track1_core.models import MonoDepthModule
        mono = MonoDepthModule.load_from_checkpoint(
            str(REPO_ROOT / "outputs" / args.f3loc_run / "mono.ckpt")).cuda().eval()

    F_W = 1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    ecfg = EventMatchConfig()
    root = Path(args.dataset_root)
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Resize((256, 256), antialias=True),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    recs = []
    for coll in args.collections:
        for scene in args.scenes:
            gpath = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
            if not (root / coll / scene / "map.png").exists() or not gpath.exists():
                continue
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > args.desdf_clip] = args.desdf_clip
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows_g, cols_g = np.nonzero(mask)
            dt = torch.tensor(desdf["desdf"], dtype=torch.float32)

            blob = np.load(gpath)
            env = np.stack([band_envelope(c, ecfg.direct_guard_samples, 1024, 64, 16,
                                          ecfg.sample_rate_hz, STFT_BANDS) for c in blob["rir"]])
            env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
            feat = featurize_band(env)
            lut = -np.ones(mask.shape, dtype=int)
            idx = blob["index"]
            lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
            pick = lut[rows_g, cols_g]
            aligned = np.zeros((len(rows_g), feat.shape[1], feat.shape[2]), dtype=np.float32)
            aligned[pick >= 0] = feat[pick[pick >= 0]]

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rir_dir = root / "rir" / coll / args.condition / scene
            dirs = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))
            dirs = [dirs[i] for i in np.linspace(0, len(dirs) - 1,
                                                 min(args.n_poses, len(dirs))).astype(int)]

            for d in tqdm.tqdm(dirs, desc=f"{coll}/{scene}", leave=False):
                i = int(d.split("_")[1])
                f = rir_dir / d / "rir.npy"
                img_p = root / coll / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"
                if not f.exists() or not img_p.exists() or i >= len(poses):
                    continue
                img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
                with torch.no_grad():
                    if rrp is not None:
                        ft = rrp("encode", obs_img=tf(img).unsqueeze(0).cuda())
                        pred = rrp("decoder_inference", depth_cond=ft).squeeze(0).cpu().numpy()
                    else:
                        x = img.astype(np.float64) / 255.0
                        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
                        x = torch.tensor(np.transpose(x, (2, 0, 1))[None],
                                         dtype=torch.float32).cuda()
                        pred = mono.encoder(x, None)[0].squeeze(0).float().cpu().numpy()
                rays = torch.tensor(get_ray_from_depth(pred, V=args.V, F_W=F_W))
                _, prob_dist, _, _ = localize(dt, rays, return_np=False)
                vis = np.asarray(prob_dist, dtype=np.float64)[rows_g, cols_g]

                obs_f = featurize_band(band_envelope(np.load(f), ecfg.direct_guard_samples,
                                                     1024, 64, 16, ecfg.sample_rate_hz, STFT_BANDS))
                nw = min(aligned.shape[2], obs_f.shape[1])
                ac = -np.abs(aligned[:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))
                ac = np.where(pick >= 0, ac, ac.min() - 1.0)

                gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols_g - gx, rows_g - gy) * pg.grid_resolution_m

                order = np.argsort(-vis)[: args.topk]
                d_short = dist[order]
                # The quantity the recall metric actually depends on is whether a
                # pick lands anywhere inside the set of shortlist entries within a
                # metre, not whether it finds the single closest one. Those differ
                # whenever that set has more than one member, which is usual.
                hit = d_short < 1.0
                if not hit.any():
                    recs.append(dict(reachable=False))
                    continue
                p_chance = float(hit.mean())
                a_hit = bool(hit[int(np.argmax(ac[order]))])
                v_hit = bool(hit[0])

                lv = np.log(np.clip(vis[order] / max(vis[order].sum(), 1e-300), 1e-300, None))
                # shift by the minimum and a floor: subtracting the minimum alone
                # sends one entry to exactly zero and its log to -inf
                ap = ac[order] - ac[order].min()
                ap = ap + max(ap.max(), 1e-30) * 1e-6
                la = np.log(ap / ap.sum())
                recs.append(dict(reachable=True, p_chance=p_chance,
                                 a_hit=a_hit, v_hit=v_hit, n_hit=int(hit.sum()),
                                 lv_std=float(lv.std()), la_std=float(la.std())))

    ok = [r for r in recs if r["reachable"]]
    if not ok:
        print("no reachable poses")
        return 1
    K = args.topk
    from scipy import stats
    pc = np.array([r["p_chance"] for r in ok])
    ah = np.array([r["a_hit"] for r in ok], dtype=float)
    vh = np.array([r["v_hit"] for r in ok], dtype=float)

    print(f"\nacoustic information inside vision's top-{K}   [{args.visual}]")
    print(f"{len(ok)} of {len(recs)} poses have at least one shortlist entry within 1 m")
    print(f"on average {np.mean([r['n_hit'] for r in ok]):.1f} of {K} entries are within 1 m\n")
    print(f"{'how the shortlist entry is picked':38s} {'lands within 1 m':>17s}")
    print("-" * 60)
    print(f"{'uniformly at random (chance)':38s} {100*pc.mean():16.1f}%")
    print(f"{'vision top-1':38s} {100*vh.mean():16.1f}%")
    print(f"{'acoustic argmax':38s} {100*ah.mean():16.1f}%")

    t, p = stats.ttest_1samp(ah - pc, 0.0)
    verdict = ("informative" if (p < 0.05 and ah.mean() > pc.mean())
               else "misleading" if p < 0.05 else "no evidence of information")
    print(f"\nacoustic {100*ah.mean():.1f}% vs chance {100*pc.mean():.1f}%: "
          f"t = {t:+.2f}, p = {p:.2e}  ->  {verdict}")
    print(f"vision    {100*vh.mean():.1f}% vs chance {100*pc.mean():.1f}%: "
          f"t = {stats.ttest_1samp(vh - pc, 0.0).statistic:+.2f}")

    lvs = np.mean([r["lv_std"] for r in ok]); las = np.mean([r["la_std"] for r in ok])
    print(f"\nlog-score spread over the shortlist (this is what a fusion weight acts on)")
    print(f"  visual   std {lvs:.4f}")
    print(f"  acoustic std {las:.4f}")
    print(f"  a weight of w makes the acoustic term {las/max(lvs,1e-12):.1f}*w times "
          f"as influential as vision")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / f"acoustic_information_{args.visual}.json"
    out.write_text(json.dumps(dict(
        visual=args.visual, topk=K, n=len(ok), n_total=len(recs),
        chance=float(pc.mean()), acoustic_hit=float(ah.mean()),
        vision_hit=float(vh.mean()), t_stat=float(t), p_value=float(p),
        lv_std=float(lvs), la_std=float(las)), indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
