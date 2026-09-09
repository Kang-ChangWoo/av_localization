#!/usr/bin/env python3
"""Score the candidate grid with the learned encoder, on the held-out test scenes.

The encoder was trained to map a floorplan response and a real recording of the
same pose onto each other, and it reaches 13.3% top-1 over 1,800 validation
pairs against a 0.06% chance. That is retrieval inside a small pool, on scenes
the encoder never saw but which are not the evaluation scenes either. This asks
the question that matters instead: on the three test scenes, against the full
candidate grid of several thousand cells, does the learned score rank the true
pose better than the training-free L1 it replaces?

The comparison is exact. Both scores read the same rendered grid, the same
recordings, the same valid cells, and differ only in how a candidate and an
observation are compared: an L1 between normalised band-energy features, or a
dot product between their embeddings. Everything downstream, including the
visual fusion and the margin gate, is untouched.

    python scripts/learned_grid_eval.py --encoder outputs/acoustic_embed_full/encoder.pt
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
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--encoder", default="outputs/acoustic_embed_full/encoder.pt")
    p.add_argument("--run-name", default="echoloc_mono_fg")
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--fuse-weight", type=float, default=0.25)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
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
    import torch.nn.functional as F
    import tqdm

    from scripts.train_acoustic_embedding import build_encoder
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, candidate_features, observation_feature, score,
    )
    from track1_core.provenance import stamp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.encoder, map_location="cpu")
    net = build_encoder(ck["rows"], ck["frames"], ck["width"], ck["embed_dim"])
    net.load_state_dict(ck["state"])
    net = net.to(device).eval()
    bands = tuple(tuple(b) for b in ck["bands"])
    first = ck["first_frame"]
    print(f"[enc] {args.encoder}: {ck['rows']} rows, bands {bands}, "
          f"first frame {first}, dim {ck['embed_dim']}")

    # the training-free baseline reads exactly the same window and bands
    cfg = GridScoreConfig(bands=bands, first_frame=first,
                          nfft=ck["nfft"], hop=ck["hop"])

    def embed_input(e: np.ndarray) -> np.ndarray:
        """Same normalisation the encoder was trained on: per-row shape."""
        return (e / np.clip(e.sum(axis=-1, keepdims=True), 1e-20, None)).astype(np.float32)

    root = Path(args.dataset_root)
    from track1_core.models import MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize
    mono = MonoDepthModule.load_from_checkpoint(
        str(REPO_ROOT / "outputs" / args.run_name / "mono.ckpt")).to(device).eval()

    recs, inputs = [], [Path(args.encoder)]
    for coll in args.collections:
        for scene in args.scenes:
            g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
            if not g.exists() or not (root / coll / scene / "map.png").exists():
                continue
            inputs.append(g)
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows, cols = np.nonzero(mask)
            dt = torch.tensor(desdf["desdf"], device=device)

            l1_feat, present = candidate_features(g, rows, cols, mask.shape, cfg)

            blob = np.load(g)
            raw = np.stack([band_energy(c, cfg) for c in blob["rir"]])
            with torch.no_grad():
                emb = torch.cat([
                    F.normalize(net(torch.tensor(embed_input(raw[i:i + 256])).to(device)), dim=1).cpu()
                    for i in range(0, len(raw), 256)]).numpy()
            idx = blob["index"]
            lut = -np.ones(mask.shape, dtype=int)
            lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
            pick = lut[rows, cols]
            cand_emb = np.zeros((len(rows), emb.shape[1]), dtype=np.float32)
            cand_emb[pick >= 0] = emb[pick[pick >= 0]]
            print(f"[geo] {coll}/{scene}: {len(rows)} cells embedded")

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rd = root / "rir" / coll / args.condition / scene
            dsdir = str(root / coll)
            ds = GridSeqDataset(dsdir, [scene], L=3, depth_dir=dsdir, depth_suffix="depth40")
            picks = np.linspace(0, len(ds) - 1, min(args.n_poses, len(ds))).astype(int)

            for chunk in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                pi = int(chunk) * 4 + 3
                f = rd / f"pose_{pi:05d}" / "rir.npy"
                if not f.exists() or pi >= len(poses):
                    continue
                d = ds[int(chunk)]
                with torch.no_grad():
                    pred, _, _ = mono.encoder(
                        torch.tensor(d["ref_img"], device=device).unsqueeze(0), None)
                rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                    device=device, dtype=torch.float32)
                _, prob_dist, _, _ = localize(dt, rays)
                vis = np.asarray(prob_dist, dtype=np.float64)[rows, cols]

                rir = np.load(f)
                s_l1 = score(l1_feat, observation_feature(rir, cfg), present, cfg)
                with torch.no_grad():
                    oe = F.normalize(net(torch.tensor(
                        embed_input(band_energy(rir, cfg))[None]).to(device)), dim=1).cpu().numpy()[0]
                s_emb = cand_emb @ oe
                s_emb = np.where(present, s_emb, s_emb.min() - 1.0)

                gx, gy, _ = pg.pose_metric_to_grid(poses[pi, :3])
                dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
                gt = int(dist.argmin())
                best = int(vis.argmax())
                far = np.hypot(cols - cols[best], rows - rows[best]) \
                    * pg.grid_resolution_m > args.mode_sep_m
                margin = float(np.log((vis[best] + 1e-300)
                                      / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))
                lv = np.log(np.clip(rank_norm(vis), 1e-9, None))
                r = dict(margin=margin, vision=float(dist[best]), n=len(rows))
                for tag, s in (("l1", s_l1), ("emb", s_emb)):
                    r[f"{tag}_alone"] = float(dist[int(s.argmax())])
                    r[f"{tag}_rank"] = int((s > s[gt]).sum()) + 1
                    la = np.log(np.clip(rank_norm(s), 1e-9, None))
                    r[f"{tag}_fused"] = float(dist[int((lv + args.fuse_weight * la).argmax())])
                recs.append(r)

    if not recs:
        print("nothing scored")
        return 1
    margins = np.array([r["margin"] for r in recs])
    tau = float(np.quantile(margins, args.gate_quantile))
    for r in recs:
        for tag in ("l1", "emb"):
            r[f"{tag}_gated"] = r[f"{tag}_fused"] if r["margin"] < tau else r["vision"]

    nc = np.mean([r["n"] for r in recs])
    v1 = np.mean([r["vision"] < 1 for r in recs])
    print(f"\n{len(recs)} poses on the three held-out test scenes, "
          f"{nc:.0f} candidates each\nvision alone {100*v1:.1f}% at 1 m, "
          f"gate at margin < {tau:.3f}\n")
    print(f"{'acoustic score':30s} {'alone <1m':>10s} {'GT rank':>9s} {'fused':>7s} {'gated':>7s} {'vs vis':>8s}")
    print("-" * 78)
    out = {"vision": float(v1), "candidates": float(nc), "n": len(recs)}
    for tag, name in (("l1", "training-free L1"), ("emb", "learned embedding")):
        a = np.mean([r[f"{tag}_alone"] < 1 for r in recs])
        rk = np.median([r[f"{tag}_rank"] for r in recs])
        fu = np.mean([r[f"{tag}_fused"] < 1 for r in recs])
        ga = np.mean([r[f"{tag}_gated"] < 1 for r in recs])
        out[tag] = dict(alone=float(a), gt_rank_median=float(rk),
                        fused=float(fu), gated=float(ga), gain=float(ga - v1))
        print(f"{name:30s} {100*a:9.1f}% {rk:9.0f} {100*fu:6.1f}% {100*ga:6.1f}% "
              f"{100*(ga-v1):+7.1f}")
    print(f"\nchance GT rank would be {nc/2:.0f}")

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "learned_grid_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(encoder=args.encoder, condition=args.condition,
                                 results=out, provenance=stamp(inputs=inputs)), indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
