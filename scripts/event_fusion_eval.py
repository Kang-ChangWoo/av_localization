#!/usr/bin/env python3
"""Training-free fusion: vision proposes, the floorplan acoustic score rejects.

The acoustic term is not asked to localise. It is asked whether it can move the
true pose up a ranking that vision produced, and specifically whether it does so
where vision is uncertain, since that is the only place the research claim says
it should help.

Both likelihoods are rank-normalised over the shared valid support before being
combined in log space with a fixed coefficient, so neither side's arbitrary
scale decides the result. The coefficient is swept and every value reported;
none of them is selected against ground truth.

The ambiguity split uses only the visual posterior, never the truth: the gap in
log likelihood between vision's best pose and its strongest competitor at least
``mode_sep_m`` away.

    python scripts/event_fusion_eval.py --scenes office_4 apartment_2 frl_apartment_5
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
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--run-name", default="echoloc_mono_fg")
    p.add_argument("--net", default="mono")
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--n-rays", type=int, default=72)
    p.add_argument("--max-order", type=int, default=1)
    p.add_argument("--top-fraction", type=float, default=1.0)
    p.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--mode-sep-m", type=float, default=1.5)
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

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.acoustic import AcousticProxyConfig, trace_echoes
    from track1_core.likelihood.events import (
        EventMatchConfig, event_match_score, observed_events, pad_event_sets,
        predicted_events_from_echoes,
    )
    from track1_core.models import MonoDepthModule, MVDepthModule, CompDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CLS = {"mono": MonoDepthModule, "mv": MVDepthModule, "comp": CompDepthModule}
    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    kw = dict(mono_ckpt=None, mv_ckpt=None) if args.net == "comp" else {}
    model = CLS[args.net].load_from_checkpoint(str(ckpt), **kw).to(device).eval()
    ecfg = EventMatchConfig(top_fraction=args.top_fraction)
    L = 3

    recs = []
    for scene in args.scenes:
        if not (root / args.collection / scene / "map.png").exists():
            continue
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        rows_g, cols_g = np.nonzero(mask)
        free = torch.tensor(occ == 255, device=device)
        origins_px = pg.grid_to_map(np.stack([cols_g, rows_g], axis=1))
        desdf_t = torch.tensor(desdf["desdf"], device=device)

        acfg = AcousticProxyConfig(n_rays=args.n_rays, max_order=args.max_order)
        echoes = trace_echoes(free, torch.tensor(origins_px, dtype=torch.float32, device=device),
                              acfg, pg.map_resolution_m)
        ev = predicted_events_from_echoes({k: v.cpu().numpy() for k, v in echoes.items()},
                                          ecfg, origins_px, pg.map_resolution_m, args.max_order)
        pd_, pw_, _ = pad_event_sets(ev)
        print(f"[{scene}] {len(rows_g)} cells, {(pw_ > 0).sum(1).mean():.1f} predicted events each")

        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene
        dsdir = str(root / args.collection)
        ds = GridSeqDataset(dsdir, [scene], L=L, depth_dir=dsdir,
                            depth_suffix="depth40" if args.net == "mono" else "depth160")
        picks = np.linspace(0, len(ds) - 1, min(args.n_poses, len(ds))).astype(int)

        for chunk in tqdm.tqdm(picks, desc=scene, leave=False):
            pose_idx = int(chunk) * (L + 1) + L
            f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
            if not f.exists() or pose_idx >= len(poses):
                continue
            data = ds[int(chunk)]
            with torch.no_grad():
                if args.net == "mono":
                    pred, _, _ = model.encoder(
                        torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
                else:
                    b = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                         for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                    b["ref_mask"] = b["src_mask"] = None
                    pred = (model.net(b)["d"] if args.net == "mv"
                            else model.comp_d_net(b)["d_comp"])
            rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                device=device, dtype=torch.float32)
            _, prob_dist, _, _ = localize(desdf_t, rays)
            vis = np.array(prob_dist, dtype=np.float64)[rows_g, cols_g]

            ac = event_match_score(pd_, pw_, observed_events(np.load(f), ecfg), ecfg)

            # rank-normalise both sides over the shared valid support, so the
            # combination cannot be decided by either one's arbitrary scale
            def rank_norm(x):
                r = np.empty_like(x, dtype=np.float64)
                r[np.argsort(x)] = np.arange(x.shape[0])
                return (r + 0.5) / x.shape[0]
            lv = np.log(np.clip(rank_norm(vis), 1e-9, None))
            la = np.log(np.clip(rank_norm(ac), 1e-9, None))

            gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
            dist = np.hypot(cols_g - gx, rows_g - gy) * pg.grid_resolution_m

            best = int(vis.argmax())
            far = np.hypot(cols_g - cols_g[best], rows_g - rows_g[best]) \
                * pg.grid_resolution_m > args.mode_sep_m
            second = vis[far].max() if far.any() else 0.0
            margin = float(np.log((vis[best] + 1e-300) / (second + 1e-300)))

            rec = dict(scene=scene, margin=margin,
                       vis_err=float(dist[best]),
                       
                       vis_gt_rank=int((vis > vis[int(dist.argmin())]).sum()) + 1,
                       ac_gt_rank=int((ac > ac[int(dist.argmin())]).sum()) + 1)
            for w in args.weights:
                fused = lv + w * la
                b = int(fused.argmax())
                rec[f"w{w}_err"] = float(dist[b])
                rec[f"w{w}_gt_rank"] = int((fused > fused[int(dist.argmin())]).sum()) + 1
            recs.append(rec)

    if not recs:
        print("nothing scored")
        return 1

    margins = np.array([r["margin"] for r in recs])
    lo, hi = np.quantile(margins, [1 / 3, 2 / 3])
    strata = [("all", np.ones(len(recs), dtype=bool)),
              ("vision ambiguous (low margin)", margins <= lo),
              ("vision middle", (margins > lo) & (margins <= hi)),
              ("vision confident (high margin)", margins > hi)]

    print(f"\n{len(recs)} poses, {args.run_name} ({args.net}), {args.condition}, "
          f"order<={args.max_order}, top_fraction={args.top_fraction}")
    print(f"acoustic-alone GT rank, median: {np.median([r['ac_gt_rank'] for r in recs]):.0f}")
    print(f"vision-alone   GT rank, median: {np.median([r['vis_gt_rank'] for r in recs]):.0f}")

    out = {"n": len(recs), "weights": args.weights, "strata": {}}
    for name, m in strata:
        g = [recs[i] for i in np.flatnonzero(m)]
        print(f"\n--- {name}  (n={len(g)})")
        print(f"{'acoustic weight':18s} {'<1m':>7s} {'<2m':>7s} {'GT rank median':>16s}")
        base = None
        rowsd = []
        for w in args.weights:
            e = np.array([r[f"w{w}_err"] for r in g])
            rk = np.median([r[f"w{w}_gt_rank"] for r in g])
            r1 = float((e < 1).mean())
            if w == 0.0:
                base = r1
            tag = "   (vision only)" if w == 0.0 else f"   {100*(r1-base):+.1f}"
            print(f"w = {w:<14.2f} {100*r1:6.1f}% {100*float((e<2).mean()):6.1f}% "
                  f"{rk:15.0f}{tag}")
            rowsd.append(dict(weight=w, recall_1m=r1, recall_2m=float((e < 2).mean()),
                              gt_rank_median=float(rk)))
        out["strata"][name] = rowsd

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "event_fusion.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
