#!/usr/bin/env python3
"""Semantic DESDF: the label of the wall each grid ray hits, in the DESDF frame.

Same grid, corner-cast rays, 36 bins and quarter-pixel march as
scripts/build_desdf.py (which reproduces the shipped caches to 0.2 px), but on
semantic_map.png: for every cell and bin the class of the first obstacle pixel
the ray enters, in SemRayLoc's ids (0 wall, 1 window, 2 door, 3 unknown = the
ray left the map or exceeded 10 m). Written as <out-dir>/<scene>/semdesdf.npy,
a dict with the same l, t as desdf.npy plus 'semdesdf' (H, W, 36) uint8, so
SemRayLoc's semantic localisation runs on exactly the pose grid the other
backbones use. With --check the frame and the depth of the same march are
compared with the reference desdf.npy (shipped, or outputs/desdf_val).

    python scripts/build_semantic_desdf.py --dataset-root /root/storage/echoloc_dataset/replica \
        --collection replica_f --scenes office_4 apartment_2 frl_apartment_5 \
        --out-dir outputs/semdesdf/replica --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CELL_PX = 10
PX_M = 0.01
MAX_M = 10.0
BINS = 36
TO_SRL = np.array([3, 0, 1, 2], dtype=np.uint8)   # semantic_map label -> SemRayLoc id
UNKNOWN = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ref-desdf-dir", type=Path, default=None,
                   help="where <scene>/desdf.npy lives for --check (default <dataset-root>/desdf)")
    p.add_argument("--check", action="store_true")
    return p.parse_args()


def build(occ: np.ndarray, sem: np.ndarray) -> dict:
    free = occ >= 128
    ys, xs = np.nonzero(free)
    l = int((xs.min() - 20) // CELL_PX * CELL_PX)
    t = int((ys.min() - 20) // CELL_PX * CELL_PX)
    W = int(np.ceil((xs.max() + CELL_PX - l) / CELL_PX))
    H = int(np.ceil((ys.max() + CELL_PX - t) / CELL_PX))
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy = t + CELL_PX * rows.ravel()
    cx = l + CELL_PX * cols.ravel()
    h, w = occ.shape
    inside = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
    centre_free = np.zeros(cy.shape, bool)
    centre_free[inside] = free[cy[inside], cx[inside]]
    theta = np.arange(BINS) * (2 * np.pi / BINS)
    dx, dy = np.cos(theta), np.sin(theta)
    steps = np.arange(0.25, MAX_M / PX_M + 0.25, 0.25, dtype=np.float32)
    depth = np.zeros((H * W, BINS), np.float32)
    lab = np.full((H * W, BINS), UNKNOWN, np.uint8)
    live = np.nonzero(centre_free)[0]
    chunk = max(1, int(4e7 // (BINS * steps.size)))
    sem_srl = TO_SRL[np.clip(sem, 0, 3)]
    for s in range(0, live.size, chunk):
        idx = live[s: s + chunk]
        px = cx[idx][:, None, None] + dx[None, :, None] * steps[None, None, :]
        py = cy[idx][:, None, None] + dy[None, :, None] * steps[None, None, :]
        ix = np.clip(np.floor(px).astype(np.int32), 0, w - 1)
        iy = np.clip(np.floor(py).astype(np.int32), 0, h - 1)
        hit = ~free[iy, ix]
        out = (px < 0) | (px > w - 1) | (py < 0) | (py > h - 1)
        hit |= out
        any_hit = hit.any(-1)
        first = np.where(any_hit, hit.argmax(-1), steps.size - 1)
        depth[idx] = np.minimum(steps[first] * PX_M, MAX_M)
        n, b = first.shape
        fi = np.take_along_axis(iy, first[:, :, None], 2)[:, :, 0]
        fx = np.take_along_axis(ix, first[:, :, None], 2)[:, :, 0]
        fo = np.take_along_axis(out, first[:, :, None], 2)[:, :, 0]
        cls = sem_srl[fi, fx]
        cls[~any_hit | fo] = UNKNOWN
        lab[idx] = cls
    return dict(l=l, t=t, desdf=depth.reshape(H, W, BINS), semdesdf=lab.reshape(H, W, BINS))


def main() -> int:
    a = parse_args()
    ref_dir = a.ref_desdf_dir or a.dataset_root / "desdf"
    for scene in a.scenes:
        sd = a.dataset_root / a.collection / scene
        occ = np.array(Image.open(sd / "map.png"))[:, :, 0]
        sp = sd / "semantic_map.png"
        if not sp.exists():
            sp = a.dataset_root / "maps" / scene / "semantic_map.png"
        sem = np.array(Image.open(sp))
        sem = sem[:, :, 0] if sem.ndim == 3 else sem
        assert ((sem > 0) == (occ < 128)).all(), f"{scene}: semantic non-zero set != obstacle set"
        d = build(occ, sem)
        msg = ""
        if a.check:
            rp = ref_dir / scene / "desdf.npy"
            if rp.exists():
                ref = np.load(rp, allow_pickle=True).item()
                same = (ref["l"], ref["t"], ref["desdf"].shape) == (d["l"], d["t"], d["desdf"].shape)
                diff = float(np.abs(np.minimum(ref["desdf"], MAX_M) - d["desdf"]).max()) if same else float("nan")
                msg = f"  frame {'ok' if same else 'DIFFERS'}  depth max|diff| {diff:.3f} m"
            else:
                msg = "  (no reference desdf)"
        counts = np.bincount(d["semdesdf"].ravel(), minlength=4)
        out = a.out_dir / scene / "semdesdf.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, dict(l=d["l"], t=d["t"], semdesdf=d["semdesdf"]), allow_pickle=True)
        print(f"[{scene}] wrote {out} shape {d['semdesdf'].shape} wall/window/door/unknown {counts.tolist()}{msg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
