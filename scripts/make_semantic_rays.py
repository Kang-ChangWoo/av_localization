#!/usr/bin/env python3
"""Semantic ray ground truth for SemRayLoc-style training on echoloc_dataset.

For every frame of a collection, the same 40 (or 160) rays that produced
depth40.txt are cast again on the scene's semantic_map.png (0 empty, 1 wall,
2 window, 3 door; its non-zero set is exactly the obstacle set of map.png), and
the label of the first obstacle pixel each ray enters is written, one row per
frame, as semantic<N>.txt next to depth<N>.txt, in SemRayLoc's class ids:

    0 wall   1 window   2 door   3 unknown (ray left the map / saturated)

Ray convention is the dataset's own (dataset_meta.json: center_angs =
flip(arctan2(u - u.mean(), N * F_W)) + yaw, exact DDA of
echoloc_simulator/raycast.py, 20 m saturation). With --check the z-depth of
the re-cast ray is compared with depth<N>.txt, which must agree to numerical
precision because it is the same cast on the same obstacle set.

    python scripts/make_semantic_rays.py --dataset-root /root/storage/echoloc_dataset/replica \
        --collections replica_f replica_g --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

SIM = Path("/root/storage/echoloc_simulator")
sys.path.insert(0, str(SIM))
from raycast import ray_cast  # noqa: E402  exact Amanatides-Woo DDA

TO_SRL = np.array([3, 0, 1, 2], dtype=np.int64)   # semantic_map label -> SemRayLoc id
UNKNOWN = 3
MAP_RES = 0.01
DIST_MAX_M = 20.0

_sem = _occ = None


def _init(sem_path):
    global _sem, _occ
    _sem = np.array(Image.open(sem_path))
    if _sem.ndim == 3:
        _sem = _sem[:, :, 0]
    _occ = np.where(_sem == 0, 255, 0).astype(np.uint8)


def center_angs(ray_n, f_w):
    u = np.arange(ray_n)
    return np.flip(np.arctan2(u - u.mean(), ray_n * f_w))


def _frame(args):
    i, x, y, yaw, ray_n, f_w = args
    H, W = _occ.shape
    r0, c0 = y / MAP_RES + H / 2, x / MAP_RES + W / 2
    dmax = DIST_MAX_M / MAP_RES
    lab = np.full(ray_n, UNKNOWN, dtype=np.int64)
    dep = np.empty(ray_n)
    for j, a in enumerate(center_angs(ray_n, f_w)):
        ang = float(a + yaw)
        t = ray_cast(_occ, np.array([r0, c0]), ang, dist_max=dmax)
        dep[j] = t * MAP_RES * np.cos(a)
        if t < dmax:
            # the pixel the ray enters at distance t: step a hair past the boundary
            rr = int(np.floor(r0 + (t + 1e-3) * np.sin(ang)))
            cc = int(np.floor(c0 + (t + 1e-3) * np.cos(ang)))
            if 0 <= rr < H and 0 <= cc < W and _sem[rr, cc] > 0:
                lab[j] = TO_SRL[_sem[rr, cc]]
            else:                      # t == 0 (start inside an obstacle) or a rounding miss
                lab[j] = TO_SRL[_sem[int(r0), int(c0)]] if _sem[int(r0), int(c0)] > 0 else UNKNOWN
    return i, lab, dep


def run(scene_dir: Path, ray_n: int, f_w: float, workers: int, check: bool, force: bool):
    poses = np.loadtxt(scene_dir / "poses.txt", ndmin=2)
    out = scene_dir / f"semantic{ray_n}.txt"
    if out.exists() and not force and sum(1 for _ in open(out)) == len(poses):
        return None
    sem_path = scene_dir / "semantic_map.png"
    if not sem_path.exists():
        sem_path = scene_dir.parents[1] / "maps" / scene_dir.name / "semantic_map.png"
    jobs = [(i, float(x), float(y), float(yaw), ray_n, f_w) for i, (x, y, yaw) in enumerate(poses)]
    lab = np.zeros((len(poses), ray_n), dtype=np.int64)
    dep = np.zeros((len(poses), ray_n))
    with Pool(workers, initializer=_init, initargs=(str(sem_path),)) as pool:
        for i, l, d in pool.imap_unordered(_frame, jobs, chunksize=8):
            lab[i] = l
            dep[i] = d
    with open(out, "w") as f:
        for row in lab:
            f.write(" ".join(str(int(v)) for v in row) + "\n")
    info = dict(frames=len(poses), counts=np.bincount(lab.ravel(), minlength=4).tolist())
    if check:
        ref = np.loadtxt(scene_dir / f"depth{ray_n}.txt", ndmin=2)
        info["depth_max_abs_diff_m"] = float(np.abs(ref - dep).max())
    return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--collections", nargs="+", required=True)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--ray-n", type=int, default=40)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--check", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    root = Path(a.dataset_root)
    meta = json.loads((root / "dataset_meta.json").read_text())
    cam = meta["camera"]
    f_w = float(cam.get("F_W") or 1 / (2 * np.tan(np.deg2rad(cam["hfov_deg"]) / 2)))
    print(f"[semantic rays] {root.name}: F_W={f_w:.5f} hfov={cam.get('hfov_deg')}", flush=True)
    tot = np.zeros(4, dtype=np.int64)
    worst = 0.0
    for col in a.collections:
        scenes = a.scenes or sorted(d.name for d in (root / col).iterdir() if (d / "poses.txt").exists())
        for sc in scenes:
            info = run(root / col / sc, a.ray_n, f_w, a.workers, a.check, a.force)
            if info is None:
                print(f"[{col}/{sc}] cached", flush=True)
                continue
            tot += np.array(info["counts"])
            worst = max(worst, info.get("depth_max_abs_diff_m", 0.0))
            print(f"[{col}/{sc}] {info['frames']} frames  wall/window/door/unknown = {info['counts']}"
                  + (f"  depth check {info['depth_max_abs_diff_m']:.2e} m" if a.check else ""), flush=True)
    frac = tot / max(tot.sum(), 1)
    print(f"[semantic rays] total rays {tot.sum()}  wall {frac[0]:.3%} window {frac[1]:.3%} door {frac[2]:.3%} "
          f"unknown {frac[3]:.3%}" + (f"  worst depth diff {worst:.2e} m" if a.check else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
