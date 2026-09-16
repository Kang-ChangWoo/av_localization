#!/usr/bin/env python3
"""Stage 1b: measure what the method costs, on the machine that produced the results.

Three things a reviewer asks for and a recall number does not answer: how many
parameters, how long a query takes and where that time goes, and what has to
be computed once per building before any query can be answered. Everything is
timed here rather than estimated, with warm-up and medians, and written to
``data/metrics/cost.json`` for ``make_cost_table.py`` to typeset.

Needs the GPU, the UnLoc checkpoint, the dataset and the candidate grids, like
``export_cases.py``; run it in the same environment.

    /opt/conda/envs/unloc/bin/python src/measure_cost.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]          # analysis/exp1_computational_resource
REPO_ROOT = HERE.parents[1]                          # av_localization
UNLOC_ROOT = REPO_ROOT.parent / "UnLoc"
DISCO_ROOT = REPO_ROOT.parent / "DisCo-FLoc"
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINTS = {
    "UnLoc": UNLOC_ROOT / "tb_logs/my_model/version_1/checkpoints/epoch=19-step=1040.ckpt",
    "F3Loc mono": REPO_ROOT / "outputs/echoloc_mono_fg/mono.ckpt",
    "DisCo-FLoc RRP": DISCO_ROOT / "checkpoints/RRP_gibson_f_best.ckpt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset/replica")
    p.add_argument("--grid-dir", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_v2")
    p.add_argument("--collection", default="replica_g")
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--n-queries", type=int, default=30, help="timed per scene")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=Path, default=HERE / "data" / "cost.json")
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def count_params(path: Path) -> dict:
    """Parameters in a Lightning checkpoint, split by the top two name levels."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    total, by = 0, {}
    for k, v in sd.items():
        if not hasattr(v, "numel"):
            continue
        total += v.numel()
        head = ".".join(k.split(".")[:2])
        by[head] = by.get(head, 0) + v.numel()
    return {"total": int(total), "by_module": {k: int(v) for k, v in by.items()},
            "file_bytes": int(path.stat().st_size)}


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch

    from track1_core.modes import ModeConfig, aggregate, extract_modes, local_discs
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, candidate_features, observation_feature, observation_rate, score,
    )
    from track1_core.floorplan import PoseGrid
    from track1_core.floorplan import valid_pose_mask

    out: dict = {"hardware": {}, "params": {}, "offline": {}, "online": {}}

    # ---- hardware -----------------------------------------------------
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True).strip()
        except Exception:
            return ""
    out["hardware"] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"),
        "python": platform.python_version(), "torch": torch.__version__,
    }

    # ---- parameters --------------------------------------------------
    for name, path in CHECKPOINTS.items():
        if path.exists():
            out["params"][name] = count_params(path)
            print(f"[params] {name}: {out['params'][name]['total']/1e6:.1f} M")
    # the acoustic branch is a matcher with no learned weights; the fusion rule
    # is four scalars chosen on the fit split
    out["params"]["acoustic branch"] = {"total": 0, "by_module": {}, "file_bytes": 0}
    out["params"]["fusion rule"] = {"total": 4, "by_module": {}, "file_bytes": 0}

    # ---- the visual backbone ------------------------------------------
    sys.path.insert(0, str(UNLOC_ROOT))
    from modules.depth_net_pl import UnLocDepthModule
    from utils.localization_utils import get_ray_from_depth_uncertainty, localize_uncertainty
    device = "cuda"
    cwd = os.getcwd()
    os.chdir(UNLOC_ROOT)
    try:
        t0 = time.perf_counter()
        net = UnLocDepthModule.load_from_checkpoint(
            checkpoint_path=str(CHECKPOINTS["UnLoc"]), strict=False).to(device).eval()
        torch.cuda.synchronize()
        out["online"]["model_load_s"] = time.perf_counter() - t0
    finally:
        os.chdir(cwd)
    policy = json.loads((REPO_ROOT / "outputs" / "metrics" / "unified_policy.json").read_text())["policy"]["unlocSTFT"]
    fuse_cfg = ModeFusionConfig(**{k: (-np.inf if v is None else v) for k, v in policy.items()})
    mcfg = ModeConfig(nms_radius_m=1.5, n_modes=10, local_radius_m=0.5)

    def sync():
        torch.cuda.synchronize()

    root = Path(args.dataset_root)
    per_scene_online = []
    for scene in args.scenes:
        g = args.grid_dir / f"{scene}.npz"
        sr = int(json.loads(str(np.load(g)["config"]))["sample_rate"])
        cfg = GridScoreConfig(feature="stft_band", window_ms=2.0, nfft=256, hop=64,
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

        # one-off per building: read the rendered grid and turn it into the
        # feature the scorer compares against
        t0 = time.perf_counter()
        cand, present = candidate_features(g, rows, cols, mask.shape, cfg)
        t_feat = time.perf_counter() - t0
        out["offline"][scene] = {
            "cells": int(len(rows)), "rendered": int(present.sum()),
            "grid_bytes": int(g.stat().st_size), "feature_bytes": int(cand.nbytes),
            "featurise_s": t_feat,
            "desdf_shape": list(desdf["desdf"].shape),
            "desdf_bytes": int(desdf["desdf"].nbytes),
        }
        print(f"[offline] {scene}: {len(rows)} cells, grid {g.stat().st_size/1e9:.2f} GB "
              f"-> feature {cand.nbytes/1e6:.1f} MB in {t_feat:.1f} s")

        poses = np.array([[float(v) for v in l.split()]
                          for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
        rd = root / "rir" / args.collection / args.condition / scene
        ds = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
        picks = np.linspace(0, len(ds) - 1, args.n_queries + args.warmup).astype(int)

        T = {k: [] for k in ("image_io", "vis_encoder", "vis_rays", "vis_localize",
                             "ac_io", "ac_feature", "ac_score", "modes", "fusion", "total")}
        torch.cuda.reset_peak_memory_stats()
        for n, k in enumerate(picks):
            i = int(ds[int(k)].split("_")[1])
            img_p = root / args.collection / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"
            rir_p = rd / ds[int(k)] / "rir.npy"
            if not img_p.exists() or not rir_p.exists():
                continue
            tt = {}
            t_all = time.perf_counter()

            t0 = time.perf_counter()
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            x = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            x = torch.tensor(np.transpose(x, (2, 0, 1))[None], dtype=torch.float32, device=device)
            sync(); tt["image_io"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.no_grad():
                loc, scl = net.encoder(x, None)[:2]
            sync(); tt["vis_encoder"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            pr, ps = get_ray_from_depth_uncertainty(loc.squeeze(0).cpu().numpy(),
                                                    scl.squeeze(0).cpu().numpy())
            tt["vis_rays"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            _, pdist, orns, _ = localize_uncertainty(
                dt, torch.tensor(pr, device=device), torch.tensor(ps, device=device),
                return_np=False, orn_slice=36)
            vis = np.asarray(pdist.cpu(), dtype=np.float64)[rows, cols]
            sync(); tt["vis_localize"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rir = np.load(rir_p); rate = observation_rate(rir_p)
            tt["ac_io"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            obs = observation_feature(rir, cfg, rate)
            tt["ac_feature"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ac = score(cand, obs, present, cfg)
            tt["ac_score"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            modes = extract_modes(vis, rows, cols, pg.grid_resolution_m, mcfg)
            discs = local_discs(modes, rows, cols, pg.grid_resolution_m, mcfg.local_radius_m)
            vis_ev = aggregate(np.log(np.clip(vis, 1e-300, None)), modes, discs, mcfg)[fuse_cfg.vis_evidence]
            ac_ev = aggregate(ac, modes, discs, mcfg)[fuse_cfg.ac_evidence]
            tt["modes"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            choose(vis_ev, ac_ev, fuse_cfg)
            tt["fusion"] = time.perf_counter() - t0
            tt["total"] = time.perf_counter() - t_all

            if n >= args.warmup:
                for kk, v in tt.items():
                    T[kk].append(v)
        med = {k: float(np.median(v)) * 1000 for k, v in T.items()}
        med["n"] = len(T["total"])
        med["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        per_scene_online.append(dict(scene=scene, **med))
        print(f"[online] {scene}: total {med['total']:.0f} ms  encoder {med['vis_encoder']:.0f}  "
              f"localize {med['vis_localize']:.0f}  acoustic {med['ac_feature']+med['ac_score']:.1f}  "
              f"modes+fusion {med['modes']+med['fusion']:.1f}  (n={med['n']})")
        del cand

    out["online"]["per_scene_ms"] = per_scene_online
    out["online"]["n_queries_per_scene"] = args.n_queries
    out["provenance"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "argv": sys.argv}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
