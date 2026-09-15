#!/usr/bin/env python3
"""Build a DESDF for a scene from its map.png, in the dataset's own convention.

The dataset ships DESDFs for its test rooms only. Selecting anything on a
validation room needs the same cache there, and the rule it follows is fully
determined by the shipped ones, so it is rebuilt here and checked against them
before it is trusted:

* the grid is the free-space bounding box of ``map.png`` padded by 20 px on the
  left and top, aligned down to a 10 px cell, and wide enough to cover the box
  plus 10 px on the right and bottom;
* cell (r, c) is cast from map pixel (t + 10 r, l + 10 c), the cell's corner, and
  pixels are looked up by truncation, not rounding;
* 36 rays at 10 degree steps, bin 0 along +x and counter-clockwise on the image
  (y down), sampled every quarter pixel;
* a ray returns the distance in metres to the first occupied pixel, capped at
  10 m, and a cell whose centre pixel is occupied is 0 in every bin.

``--check`` rebuilds a shipped test room and reports the agreement, which is
the only argument for using the output on a room that has no reference.

    python scripts/build_desdf.py --dataset-root /root/storage/echoloc_dataset/replica \
        --collection replica_f --scenes apartment_1 frl_apartment_4 office_3 \
        --out-dir outputs/desdf_val/replica
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CELL_PX = 10
PX_M = 0.01
MAX_M = 10.0
BINS = 36


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--check", action="store_true",
                   help="compare with the shipped desdf/<scene>/desdf.npy instead of writing")
    return p.parse_args()


def build(occ: np.ndarray) -> dict:
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
    dx, dy = np.cos(theta), np.sin(theta)             # bin k is k*10 deg counter-clockwise on the image (verified against the shipped cache)
    steps = np.arange(0.25, MAX_M / PX_M + 0.25, 0.25, dtype=np.float32)   # quarter-pixel steps
    out = np.zeros((H * W, BINS), np.float32)
    live = np.nonzero(centre_free)[0]
    # chunked so that (cells x bins x steps) stays a few hundred MB
    chunk = max(1, int(4e7 // (BINS * steps.size)))
    for s in range(0, live.size, chunk):
        idx = live[s: s + chunk]
        px = cx[idx][:, None, None] + dx[None, :, None] * steps[None, None, :]
        py = cy[idx][:, None, None] + dy[None, :, None] * steps[None, None, :]
        ix = np.clip(np.floor(px).astype(np.int32), 0, w - 1)
        iy = np.clip(np.floor(py).astype(np.int32), 0, h - 1)
        hit = ~free[iy, ix]                                           # (n, bins, steps)
        # leaving the map counts as a hit at the edge
        hit |= (px < 0) | (px > w - 1) | (py < 0) | (py > h - 1)
        first = np.where(hit.any(-1), hit.argmax(-1), steps.size - 1)
        out[idx] = np.minimum(steps[first] * PX_M, MAX_M)
    return dict(l=l, t=t, desdf=out.reshape(H, W, BINS))


def main() -> int:
    args = parse_args()
    import cv2
    for scene in args.scenes:
        occ = cv2.imread(str(args.dataset_root / args.collection / scene / "map.png"))[:, :, 0]
        d = build(occ)
        if args.check:
            ref = np.load(args.dataset_root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            same = (ref["l"], ref["t"], ref["desdf"].shape) == (d["l"], d["t"], d["desdf"].shape)
            if not same:
                print(f"[{scene}] frame differs: ref l,t,shape {ref['l']},{ref['t']},{ref['desdf'].shape} "
                      f"vs {d['l']},{d['t']},{d['desdf'].shape}")
                continue
            a, b = ref["desdf"], d["desdf"]
            both = (a > 0) & (b > 0)
            diff = np.abs(a - b)[both]
            print(f"[{scene}] frame identical; zero-cells agree {np.mean((a == 0) == (b == 0)):.4f}; "
                  f"over {both.sum()} rays: mean |diff| {diff.mean()*100:.2f} px, "
                  f"95th {np.percentile(diff, 95)*100:.1f} px, >5 px {np.mean(diff > 0.05)*100:.2f}%")
        else:
            out = args.out_dir / scene / "desdf.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, d, allow_pickle=True)
            print(f"[{scene}] wrote {out}  l,t={d['l']},{d['t']} shape {d['desdf'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
