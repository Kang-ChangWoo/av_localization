#!/usr/bin/env python3
"""Isolate the part of the acoustic response that carries non-line-of-sight geometry.

The rendered candidate grid already contains NLOS paths: SoundSpaces traces
multiple reflections, so energy that has been around a corner is in there. What
has never been done is to *separate* it from the line-of-sight part, and the
separation is available for free in the time axis.

An arrival at delay tau has travelled c*tau. The first wall return is 2d/c for a
wall at distance d, so for rooms of a few metres everything before roughly 30 ms
is direct-plus-first-bounce, which is close to what the camera can also see. A
path that has visited a region outside line of sight has to reflect at least
twice, which puts it later. Cutting the window therefore splits the response
into a part the visual branch is largely redundant with and a part it
structurally cannot have.

Two other axes are swept with it, both aimed at the same problem, which is that
furniture is in the recording and not in the floorplan:

  frequency   a 500 Hz wavelength is 69 cm and diffracts around a chair; a
              3 kHz wavelength is 11 cm and scatters off it. Measured in the
              matched domain the 0-1 kHz band reaches 98-100% recall against
              30-38% for 2-4 kHz, so restricting to low frequency is the
              cheapest clutter filter available.
  asymmetry   furniture adds arrivals and occludes wall returns, so the model
              should be checked as a subset of the observation rather than
              against it. Symmetric L1 charges for every unexplainable echo.

    python scripts/acoustic_nlos_sweep.py --collections replica_f replica_g
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

SOUND_SPEED = 343.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--run-name", default="echoloc_mono_fg")
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--fuse-weight", type=float, default=0.25)
    p.add_argument("--gate-quantile", type=float, default=0.6)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--nfft", type=int, default=64)
    p.add_argument("--hop", type=int, default=16)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# (label, band list, first frame, last frame or None, one-sided?)
def variants(frames: int, ms_per_frame: float):
    """The sweep. Frame f covers window time f * ms_per_frame."""
    def fr(ms):
        return int(round(ms / ms_per_frame))
    LOW = [(0, 1000)]
    THREE = [(0, 500), (500, 1500), (1500, 4000)]
    HIGH = [(1500, 4000)]
    # 30 ms of window time is about 5 m of extra path beyond the direct sound,
    # which is where second-order and later reflections start to dominate
    late0 = fr(30.0)
    return [
        ("three bands, whole window  [current]", THREE, 0, frames, False),
        ("low band only, whole window", LOW, 0, frames, False),
        ("high band only, whole window", HIGH, 0, frames, False),
        ("three bands, early <30 ms  [LOS-like]", THREE, 0, late0, False),
        ("three bands, late >30 ms   [NLOS]", THREE, late0, frames, False),
        ("low band, early <30 ms", LOW, 0, late0, False),
        ("low band, late >30 ms      [NLOS]", LOW, late0, frames, False),
        ("low band, late, one-sided", LOW, late0, frames, True),
        ("three bands, late, one-sided", THREE, late0, frames, True),
        ("low band, whole window, one-sided", LOW, 0, frames, True),
    ]


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import tqdm

    from scripts.margin_gated_fusion import band_envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.events import EventMatchConfig
    from track1_core.models import MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    ecfg = EventMatchConfig()
    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ms_per_frame = 1000.0 * args.hop / ecfg.sample_rate_hz
    n_frames = 1 + (1024 - args.nfft) // args.hop
    VAR = variants(n_frames, ms_per_frame)
    print(f"[nlos] window {n_frames} frames x {ms_per_frame:.2f} ms = "
          f"{n_frames*ms_per_frame:.0f} ms; late starts at frame "
          f"{int(round(30.0/ms_per_frame))}")

    mono = MonoDepthModule.load_from_checkpoint(
        str(REPO_ROOT / "outputs" / args.run_name / "mono.ckpt")).to(device).eval()

    def describe(rir, bands):
        """(channel x band, frame) energy, undivided so windows can be sliced."""
        return band_envelope(rir, ecfg.direct_guard_samples, 1024, args.nfft,
                             args.hop, ecfg.sample_rate_hz, bands)

    def shape(env):
        """Per-row temporal shape plus the energy split across rows."""
        per = env / env.sum(axis=-1, keepdims=True).clip(1e-12)
        total = env.sum(axis=(-1, -2))
        split = env.sum(axis=-1) / np.clip(total[..., None], 1e-12, None)
        return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)

    band_sets = {tuple(b) for _, b, _, _, _ in VAR}
    recs = {lab: [] for lab, *_ in VAR}
    vis_rec = []

    for coll in args.collections:
        for scene in args.scenes:
            gpath = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
            if not gpath.exists() or not (root / coll / scene / "map.png").exists():
                continue
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            mask = valid_pose_mask(occ, pg)
            rows_g, cols_g = np.nonzero(mask)
            dt = torch.tensor(desdf["desdf"], device=device)

            blob = np.load(gpath)
            idx = blob["index"]
            lut = -np.ones(mask.shape, dtype=int)
            lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
            pick = lut[rows_g, cols_g]
            cand = {}
            for bs in band_sets:
                e = np.stack([describe(c, list(bs)) for c in blob["rir"]])
                a = np.zeros((len(rows_g), e.shape[1], e.shape[2]), dtype=np.float32)
                a[pick >= 0] = e[pick[pick >= 0]]
                cand[bs] = a
            print(f"[geo] {coll}/{scene}: {len(rows_g)} cells, "
                  f"{len(band_sets)} band sets cached")

            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            rir_dir = root / "rir" / coll / args.condition / scene
            dsdir = str(root / coll)
            ds = GridSeqDataset(dsdir, [scene], L=3, depth_dir=dsdir, depth_suffix="depth40")
            picks = np.linspace(0, len(ds) - 1, min(args.n_poses, len(ds))).astype(int)

            for chunk in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                pi = int(chunk) * 4 + 3
                f = rir_dir / f"pose_{pi:05d}" / "rir.npy"
                if not f.exists() or pi >= len(poses):
                    continue
                d = ds[int(chunk)]
                with torch.no_grad():
                    pred, _, _ = mono.encoder(
                        torch.tensor(d["ref_img"], device=device).unsqueeze(0), None)
                rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                    device=device, dtype=torch.float32)
                _, prob_dist, _, _ = localize(dt, rays)
                vis = np.asarray(prob_dist, dtype=np.float64)[rows_g, cols_g]

                gx, gy, _ = pg.pose_metric_to_grid(poses[pi, :3])
                dist = np.hypot(cols_g - gx, rows_g - gy) * pg.grid_resolution_m
                best = int(vis.argmax())
                far = np.hypot(cols_g - cols_g[best], rows_g - rows_g[best]) \
                    * pg.grid_resolution_m > args.mode_sep_m
                margin = float(np.log((vis[best] + 1e-300)
                                      / ((vis[far].max() + 1e-300) if far.any() else 1e-300)))
                vis_rec.append(dict(err=float(dist[best]), margin=margin))
                lv = np.log(np.clip(rank_norm(vis), 1e-9, None))
                rir = np.load(f)
                obs_raw = {bs: describe(rir, list(bs)) for bs in band_sets}

                for lab, bands, f0, f1, one_sided in VAR:
                    bs = tuple(bands)
                    c = cand[bs][:, :, f0:f1]
                    o = obs_raw[bs][:, f0:f1]
                    cn = c / np.clip(c.sum(axis=(1, 2), keepdims=True), 1e-20, None)
                    on = o / max(o.sum(), 1e-20)
                    cf, of = shape(cn), shape(on)
                    nw = min(cf.shape[2], of.shape[1])
                    diff = cf[:, :, :nw] - of[None, :, :nw]
                    # one-sided: only charge for model energy the recording lacks,
                    # since furniture adds arrivals but cannot invent wall returns
                    ac = -(np.clip(diff, 0, None) if one_sided else np.abs(diff)).sum(axis=(1, 2))
                    ac = np.where(pick >= 0, ac, ac.min() - 1.0)
                    la = np.log(np.clip(rank_norm(ac), 1e-9, None))
                    fused = int((lv + args.fuse_weight * la).argmax())
                    recs[lab].append(dict(
                        alone=float(dist[int(ac.argmax())]),
                        fused=float(dist[fused]), margin=margin,
                        vis=float(dist[best])))

    if not vis_rec:
        print("nothing scored")
        return 1
    margins = np.array([r["margin"] for r in vis_rec])
    tau = float(np.quantile(margins, args.gate_quantile))
    v1 = np.mean([r["err"] < 1 for r in vis_rec])

    print(f"\n{len(vis_rec)} poses, vision alone {100*v1:.1f}% at 1 m, "
          f"gate at margin < {tau:.3f}\n")
    print(f"{'acoustic variant':40s} {'alone':>7s} {'fused':>7s} {'gated':>7s} {'vs vision':>10s}")
    print("-" * 78)
    out = {}
    for lab, *_ in VAR:
        g = recs[lab]
        if not g:
            continue
        alone = np.mean([x["alone"] < 1 for x in g])
        fused = np.mean([x["fused"] < 1 for x in g])
        gated = np.mean([(x["fused"] if x["margin"] < tau else x["vis"]) < 1 for x in g])
        out[lab] = dict(alone=float(alone), fused=float(fused), gated=float(gated),
                        gain=float(gated - v1))
        print(f"{lab:40s} {100*alone:6.1f}% {100*fused:6.1f}% {100*gated:6.1f}% "
              f"{100*(gated-v1):+9.1f}")

    p = args.out or REPO_ROOT / "outputs" / "metrics" / "acoustic_nlos_sweep.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(n=len(vis_rec), vision=float(v1), tau=tau,
                                 condition=args.condition, variants=out), indent=2))
    print(f"\nwrote {p}")
    return 0


def rank_norm(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.shape[0], dtype=np.float64)
    r[np.argsort(x)] = np.arange(x.shape[0])
    return (r + 0.5) / x.shape[0]


if __name__ == "__main__":
    raise SystemExit(main())
