#!/usr/bin/env python3
"""Choose the shortlist size per observation instead of fixing it per scene.

A single K cannot serve every scene: where vision is confident a long shortlist
only invites acoustics to overrule a correct answer, and where vision is
ambiguous a short one throws the truth away before acoustics ever sees it. The
optimal K measured per scene tracked how well vision ranked the truth, which is
something vision can report about itself at inference time.

Two signals are tried, both computable without the ground truth:

  entropy   of vision's posterior on the valid support, normalised to [0, 1]
  mass      the number of cells needed to accumulate a fixed share of the
            posterior -- a direct estimate of how many hypotheses vision holds

Each maps to K by a monotone rule whose two parameters are fitted on other
scenes, never on the scene being scored.

    python scripts/adaptive_k_check.py
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
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--run-name", default="echoloc_mono")
    p.add_argument("--net", default="mono", choices=["mono", "mv"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--mass", type=float, default=0.5,
                   help="posterior share used by the 'mass' signal")
    p.add_argument("--fixed-K", type=int, nargs="+", default=[25, 50, 100, 250])
    p.add_argument("--n-poses", type=int, default=300)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def featurize(env: np.ndarray) -> np.ndarray:
    """chan_shape: per-channel envelope plus the ring's energy split."""
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
    from track1_core.models import MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    scenes = args.scenes or sorted(os.listdir(root / "desdf"))
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))

    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    cls = MonoDepthModule if args.net == "mono" else MVDepthModule
    model = cls.load_from_checkpoint(str(ckpt)).to(device).eval()

    per_scene = {}
    for scene in scenes:
        g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
        if not g.exists():
            continue
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
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
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rir_dir = root / "rir" / args.collection / args.condition / scene
        dataset_dir = str(root / args.collection)
        dataset = GridSeqDataset(dataset_dir, [scene], L=L, depth_dir=dataset_dir,
                                 depth_suffix="depth40" if args.net == "mono" else "depth160")
        picks = np.linspace(0, len(dataset) - 1, min(args.n_poses, len(dataset))).astype(int)

        recs = []
        for chunk in tqdm.tqdm(picks, desc=scene):
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
                    pred = model.net(b)["d"]
            rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                device=device, dtype=torch.float32)
            _, prob_dist, _, _ = localize(desdf_t, rays)
            v = np.array(prob_dist, dtype=np.float64)[mask]
            p = v / max(v.sum(), 1e-300)
            order = np.argsort(-p)

            # confidence signals, computed without the truth
            ent = float(-(p * np.log(np.clip(p, 1e-300, None))).sum() / np.log(len(p)))
            cum = np.cumsum(p[order])
            n_mass = int(np.searchsorted(cum, args.mass) + 1)

            obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
            obs_f = featurize(obs / max(obs.sum(), 1e-20))
            nw = min(cand_f.shape[2], obs_f.shape[1])
            ac = -np.abs(cand_f[:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))

            gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
            dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
            recs.append(dict(order=order, ac=ac, dist=dist, ent=ent, n_mass=n_mass,
                             vis=float(dist[order[0]])))
        if recs:
            per_scene[scene] = recs
            print(f"[adaptive] {scene}: {len(recs)} poses, vision {100*np.mean([r['vis']<1 for r in recs]):.0f}% @1m, "
                  f"entropy {np.mean([r['ent'] for r in recs]):.3f}, "
                  f"mass-K median {np.median([r['n_mass'] for r in recs]):.0f}")

    if not per_scene:
        print("nothing to evaluate")
        return 1

    def rerank_at(rec, K):
        sel = rec["order"][:max(1, int(K))]
        return float(rec["dist"][sel][int(rec["ac"][sel].argmax())])

    # ---- fixed K, for reference -------------------------------------------
    print(f"\n{'policy':28s} " + " ".join(f"{s[:12]:>13s}" for s in per_scene) + f" {'pooled':>8s}")
    table = {}
    allrec = [r for rs in per_scene.values() for r in rs]
    row = [np.mean([r["vis"] < 1 for r in rs]) for rs in per_scene.values()]
    table["vision alone"] = row + [np.mean([r["vis"] < 1 for r in allrec])]
    for K in args.fixed_K:
        row = [np.mean([rerank_at(r, K) < 1 for r in rs]) for rs in per_scene.values()]
        table[f"fixed K={K}"] = row + [np.mean([rerank_at(r, K) < 1 for r in allrec])]

    # ---- adaptive, calibrated leave-one-scene-out --------------------------
    grid_a = [4, 8, 16, 32, 64, 128, 256]
    grid_b = [0.5, 1.0, 2.0, 4.0]
    for signal in ("entropy", "mass"):
        row = []
        for held in per_scene:
            others = [r for s, rs in per_scene.items() if s != held for r in rs]
            best, best_acc = None, -1
            for a in grid_a:
                for b in grid_b:
                    acc = np.mean([rerank_at(r, a * (r["ent"] ** b) * len(r["dist"]) ** 0
                                             if signal == "entropy" else a * (r["n_mass"] ** b) / 8)
                                   < 1 for r in others])
                    if acc > best_acc:
                        best, best_acc = (a, b), acc
            a, b = best
            row.append(np.mean([rerank_at(r, a * (r["ent"] ** b) if signal == "entropy"
                                          else a * (r["n_mass"] ** b) / 8) < 1
                                for r in per_scene[held]]))
        table[f"adaptive ({signal})"] = row + [float(np.mean(row))]

    for name, vals in table.items():
        print(f"{name:28s} " + " ".join(f"{100*v:12.0f}%" for v in vals))

    out = args.out or REPO_ROOT / "outputs" / "metrics" / "adaptive_k.json"
    out.write_text(json.dumps({"scenes": list(per_scene),
                               "policies": {k: [float(x) for x in v] for k, v in table.items()}},
                              indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
