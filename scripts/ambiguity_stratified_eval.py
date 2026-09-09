#!/usr/bin/env python3
"""Measure the acoustic gain where the claim says it should appear.

The hypothesis is not that sound localizes better than sight. It is that sound
rejects visually plausible but geometrically inconsistent hypotheses. An average
over all poses cannot show that: most poses are not ambiguous, and averaging
buries the cases the claim is about.

So poses are stratified by how ambiguous vision actually is, using signals
computable at inference time without the ground truth:

  entropy   of the visual posterior over the valid support, normalised
  margin    log ratio between the best and the strongest competing mode, where
            a competing mode is a peak at least `mode_sep_m` away from the best
  modes     how many spatially separated peaks carry meaningful probability

The prediction under the hypothesis is a gradient: little or no gain in the
unambiguous stratum, a large gain in the ambiguous one. A flat profile would
mean the acoustic term is acting as generic noise rather than as verification.

    python scripts/ambiguity_stratified_eval.py --run-name echoloc_mono_fg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--run-name", default="echoloc_mono_fg")
    p.add_argument("--net", default="mono", choices=["mono", "mv", "comp"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--mode-sep-m", type=float, default=1.5,
                   help="how far apart two peaks must be to count as competing hypotheses")
    p.add_argument("--n-poses", type=int, default=300)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def featurize(env: np.ndarray) -> np.ndarray:
    per = env / env.sum(axis=-1, keepdims=True).clip(1e-12)
    total = env.sum(axis=(-1, -2))
    split = env.sum(axis=-1) / np.clip(total[..., None], 1e-12, None)
    return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import tqdm

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.models import CompDepthModule, MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    scenes = args.scenes or sorted(os.listdir(root / "desdf"))
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
    CLS = {"mono": MonoDepthModule, "mv": MVDepthModule, "comp": CompDepthModule}

    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    kw = dict(mono_ckpt=None, mv_ckpt=None) if args.net == "comp" else {}
    model = CLS[args.net].load_from_checkpoint(str(ckpt), **kw).to(device).eval()

    recs = []
    for coll in args.collections:
        for scene in scenes:
            g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
            if not g.exists():
                continue
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows, cols = np.nonzero(mask)
            desdf_t = torch.tensor(desdf["desdf"], device=device)

            blob = np.load(g)
            cand, index = blob["rir"], blob["index"]
            env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
            env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
            lut = -np.ones(mask.shape, dtype=int)
            lut[index[:, 0], index[:, 1]] = np.arange(len(index))
            pick = lut[rows, cols]
            aligned = np.zeros((len(rows), env.shape[1], env.shape[2]))
            aligned[pick >= 0] = env[pick[pick >= 0]]
            cand_f = featurize(aligned)

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rir_dir = root / "rir" / coll / args.condition / scene
            dsdir = str(root / coll)
            dataset = GridSeqDataset(dsdir, [scene], L=L, depth_dir=dsdir,
                                     depth_suffix="depth40" if args.net == "mono" else "depth160")
            picks = np.linspace(0, len(dataset) - 1,
                                min(args.n_poses, len(dataset))).astype(int)

            for chunk in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                pose_idx = int(chunk) * (L + 1) + L
                f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
                if not f.exists():
                    continue
                data = dataset[int(chunk)]
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
                v = np.array(prob_dist, dtype=np.float64)[mask]
                p = v / max(v.sum(), 1e-300)
                order = np.argsort(-v)

                # --- how ambiguous is vision here? ---------------------------
                ent = float(-(p * np.log(np.clip(p, 1e-300, None))).sum() / np.log(len(p)))
                bx, by = cols[order[0]], rows[order[0]]
                far = np.hypot(cols - bx, rows - by) * pg.grid_resolution_m > args.mode_sep_m
                second = v[far].max() if far.any() else 0.0
                margin = float(np.log((v[order[0]] + 1e-300) / (second + 1e-300)))
                # count spatially separated peaks above half the best score
                strong = np.where(v >= 0.5 * v[order[0]])[0]
                modes, taken = 0, []
                for i in strong[np.argsort(-v[strong])]:
                    if all(np.hypot(cols[i] - cols[j], rows[i] - rows[j])
                           * pg.grid_resolution_m > args.mode_sep_m for j in taken):
                        taken.append(i); modes += 1
                    if modes >= 10:
                        break

                gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
                dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m

                obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
                obs_f = featurize(obs / max(obs.sum(), 1e-20))
                nw = min(cand_f.shape[2], obs_f.shape[1])
                ac = -np.abs(cand_f[:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))

                sel = order[: args.topk]
                recs.append(dict(
                    scene=scene, coll=coll,
                    vis=float(dist[order[0]]),
                    ac_only=float(dist[int(ac.argmax())]),
                    fused=float(dist[sel][int(ac[sel].argmax())]),
                    oracle=float(dist[sel].min()),
                    entropy=ent, margin=margin, modes=modes))

    if not recs:
        print("nothing scored")
        return 1

    def report(name, key, bins, labels):
        vals = np.array([r[key] for r in recs])
        edges = np.quantile(vals, bins)
        print(f"\n=== stratified by {name} ===")
        print(f"{'stratum':22s} {'n':>4s} {'vision':>8s} {'acoustic':>9s} "
              f"{'fused':>7s} {'gain':>7s} {'oracle':>8s}")
        out = []
        for i, lab in enumerate(labels):
            lo = -np.inf if i == 0 else edges[i]
            hi = np.inf if i == len(labels) - 1 else edges[i + 1]
            m = (vals > lo) & (vals <= hi) if i else (vals <= hi)
            if not m.any():
                continue
            g = [recs[j] for j in np.where(m)[0]]
            v = np.mean([x["vis"] < 1 for x in g])
            a = np.mean([x["ac_only"] < 1 for x in g])
            fu = np.mean([x["fused"] < 1 for x in g])
            orc = np.mean([x["oracle"] < 1 for x in g])
            print(f"{lab:22s} {len(g):4d} {100*v:7.1f}% {100*a:8.1f}% "
                  f"{100*fu:6.1f}% {100*(fu-v):+6.1f} {100*orc:7.1f}%")
            out.append(dict(stratum=lab, n=len(g), vision=float(v), acoustic=float(a),
                            fused=float(fu), gain=float(fu - v), oracle=float(orc)))
        return out

    summary = {"run": args.run_name, "net": args.net, "condition": args.condition,
               "topk": args.topk, "poses": len(recs)}
    print(f"\n{len(recs)} poses, {args.run_name} ({args.net}), K={args.topk}")
    summary["by_entropy"] = report("visual entropy (higher = more ambiguous)", "entropy",
                                   [0, 1 / 3, 2 / 3], ["low (confident)", "medium", "high (ambiguous)"])
    summary["by_margin"] = report("top-1 vs competing-mode margin (lower = more ambiguous)",
                                  "margin", [0, 1 / 3, 2 / 3],
                                  ["low margin (ambiguous)", "medium", "high margin (confident)"])
    mv = np.array([r["modes"] for r in recs])
    print(f"\n=== stratified by number of separated visual modes ===")
    print(f"{'stratum':22s} {'n':>4s} {'vision':>8s} {'acoustic':>9s} {'fused':>7s} {'gain':>7s} {'oracle':>8s}")
    modes_out = []
    for lab, m in (("1 mode", mv <= 1), ("2-3 modes", (mv >= 2) & (mv <= 3)), ("4+ modes", mv >= 4)):
        if not m.any():
            continue
        g = [recs[j] for j in np.where(m)[0]]
        v = np.mean([x["vis"] < 1 for x in g]); a = np.mean([x["ac_only"] < 1 for x in g])
        fu = np.mean([x["fused"] < 1 for x in g]); orc = np.mean([x["oracle"] < 1 for x in g])
        print(f"{lab:22s} {len(g):4d} {100*v:7.1f}% {100*a:8.1f}% {100*fu:6.1f}% "
              f"{100*(fu-v):+6.1f} {100*orc:7.1f}%")
        modes_out.append(dict(stratum=lab, n=len(g), vision=float(v), acoustic=float(a),
                              fused=float(fu), gain=float(fu - v), oracle=float(orc)))
    summary["by_modes"] = modes_out

    out = args.out or REPO_ROOT / "outputs" / "metrics" / f"ambiguity_stratified_{args.run_name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
