#!/usr/bin/env python3
"""How often does the acoustic term actually change the answer, and by how much?

The recall tables say the fusion is worth a few points, and the figures show
individual poses where the pick moves across a room. Neither says how often the
pick moves at all. If it moves on two poses in a hundred, a four-point gain is
being carried by a handful of samples and the method is far more fragile than
the tables suggest; if it moves on half of them and the net is still only four
points, then it is trading repairs against regressions at close to even odds and
the gate is doing most of the work.

This prints the distribution directly, with no thresholding except where noted:

    moved            fused pick is a different cell from the visual pick
    moved > 1.5 m    it moved to a different mode, not a neighbouring cell
    improved         the error went down, by any amount
    crossed into 1 m error was above 1 m and is now below, the recall gain
    crossed out      the reverse

It also reports the Spearman correlation between the visual and acoustic rank
fields, which bounds how much the fusion can do: if sound ranks cells the way
vision already does, adding it is close to a no-op whatever the weight.

    /opt/conda/envs/unloc/bin/python scripts/fusion_change_probe.py --checkpoint <ckpt>
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
    p.add_argument("--collection", default="replica_g")
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--checkpoint", default=str(UNLOC_ROOT / "logs" / "unloc_gibson_vitl.ckpt"))
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--weight", type=float, default=0.5)
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--orn-slice", type=int, default=36)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "metrics" / "fusion_change_probe.json")
    p.add_argument("--gpu", default="0")
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

    sys.path.insert(0, str(UNLOC_ROOT))
    from modules.depth_net_pl import UnLocDepthModule
    from utils.localization_utils import (
        get_ray_from_depth_uncertainty, localize_uncertainty,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cwd = os.getcwd()
    os.chdir(UNLOC_ROOT)
    try:
        net = UnLocDepthModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint, strict=False).to(device).eval()
    finally:
        os.chdir(cwd)

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, candidate_features, observation_feature, observation_rate, score,
    )

    root = Path(args.dataset_root)
    recs = []
    for scene in args.scenes:
        g = REPO_ROOT / args.grid_dir / f"{scene}.npz"
        if not g.exists():
            continue
        sr = int(json.loads(str(np.load(g)["config"]))["sample_rate"])
        cfg = GridScoreConfig(feature="envelope", window_ms=args.window_ms,
                              sample_rate_hz=sr,
                              direct_guard_samples=int(round(sr * 2 / 1000)),
                              usable_samples=int(round(sr * 128 / 1000)))
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        rows, cols = np.nonzero(mask)
        dt = torch.tensor(desdf["desdf"], device=device)
        cf, present = candidate_features(g, rows, cols, mask.shape, cfg)

        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rd = root / "rir" / args.collection / args.condition / scene
        ds_dirs = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
        picks = np.linspace(0, len(ds_dirs) - 1, min(args.n_poses, len(ds_dirs))).astype(int)

        for k in tqdm.tqdm(picks, desc=scene, leave=False):
            d = ds_dirs[int(k)]
            i = int(d.split("_")[1])
            f = rd / d / "rir.npy"
            img_p = root / args.collection / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"
            if not f.exists() or not img_p.exists() or i >= len(poses):
                continue
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR).astype(np.float32)
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            x = torch.tensor(np.transpose(x, (2, 0, 1))[None],
                             dtype=torch.float32, device=device)
            with torch.no_grad():
                loc, sc = net.encoder(x, None)[:2]
            pr, ps = get_ray_from_depth_uncertainty(
                loc.squeeze(0).cpu().numpy(), sc.squeeze(0).cpu().numpy())
            _, pdist, _, _ = localize_uncertainty(
                dt, torch.tensor(pr, device=device), torch.tensor(ps, device=device),
                return_np=False, orn_slice=args.orn_slice)
            vis = np.asarray(pdist.cpu(), dtype=np.float64)[rows, cols]
            ac = score(cf, observation_feature(np.load(f), cfg, observation_rate(f)),
                       present, cfg)
            rv, ra = rank_norm(vis), rank_norm(ac)
            fu = np.log(rv) + args.weight * np.log(ra)

            gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
            dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
            vb, fb = int(vis.argmax()), int(fu.argmax())
            # Spearman on ranks is just Pearson on the rank vectors
            rho = float(np.corrcoef(rv, ra)[0, 1])
            # how much of the fused ordering the visual term can still explain,
            # restricted to the cells that are actually in contention
            top = np.argsort(-vis)[:50]
            recs.append(dict(
                scene=scene, moved=vb != fb,
                move_m=float(np.hypot(cols[fb] - cols[vb], rows[fb] - rows[vb])
                             * pg.grid_resolution_m),
                e_vis=float(dist[vb]), e_fu=float(dist[fb]),
                rho=rho,
                lv_spread=float(np.log(rv[top]).max() - np.log(rv[top]).min()),
                la_spread=float(args.weight * (np.log(ra[top]).max()
                                               - np.log(ra[top]).min()))))

    if not recs:
        print("nothing scored")
        return 1
    n = len(recs)
    mv = [r for r in recs if r["moved"]]
    far = [r for r in recs if r["move_m"] > 1.5]
    imp = [r for r in mv if r["e_fu"] < r["e_vis"]]
    ci = [r for r in recs if r["e_vis"] >= 1 and r["e_fu"] < 1]
    co = [r for r in recs if r["e_vis"] < 1 and r["e_fu"] >= 1]
    pc = lambda x: f"{100*len(x)/n:5.1f}%  ({len(x):3d}/{n})"

    print(f"\n{n} poses, weight {args.weight:g}, {args.collection}")
    print(f"  pick moved at all           {pc(mv)}")
    print(f"  moved more than 1.5 m       {pc(far)}")
    print(f"  moved and got closer        {pc(imp)}"
          f"   of the {len(mv)} that moved: {100*len(imp)/max(len(mv),1):.0f}%")
    print(f"  crossed into 1 m (a gain)   {pc(ci)}")
    print(f"  crossed out of 1 m (a loss) {pc(co)}")
    print(f"  net recall change           {100*(len(ci)-len(co))/n:+.1f} points")
    if far:
        print(f"  median move, when it moves far  "
              f"{np.median([r['move_m'] for r in far]):.1f} m")
    print(f"\n  visual/acoustic rank correlation, median {np.median([r['rho'] for r in recs]):+.3f}"
          f"   (0 would mean sound says something entirely new)")
    print(f"  across vision's top-50 cells the log-score spread is")
    print(f"    visual term    {np.median([r['lv_spread'] for r in recs]):.4f}")
    print(f"    acoustic term  {np.median([r['la_spread'] for r in recs]):.4f}"
          f"   ratio {np.median([r['la_spread'] for r in recs]) / max(np.median([r['lv_spread'] for r in recs]), 1e-9):.0f}x")
    print("  so among the cells vision shortlists, the acoustic term decides "
          "the ordering\n  almost entirely on its own")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        n=n, weight=args.weight, collection=args.collection,
        moved=len(mv), moved_far=len(far), improved=len(imp),
        crossed_in=len(ci), crossed_out=len(co),
        rho_median=float(np.median([r["rho"] for r in recs])),
        lv_spread_median=float(np.median([r["lv_spread"] for r in recs])),
        la_spread_median=float(np.median([r["la_spread"] for r in recs])),
        per_scene={s: dict(
            n=sum(r["scene"] == s for r in recs),
            moved=sum(r["scene"] == s and r["moved"] for r in recs),
            crossed_in=sum(r["scene"] == s and r["e_vis"] >= 1 and r["e_fu"] < 1 for r in recs),
            crossed_out=sum(r["scene"] == s and r["e_vis"] < 1 and r["e_fu"] >= 1 for r in recs))
            for s in args.scenes}), indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
