#!/usr/bin/env python3
"""Render the candidate grid with an image-source model instead of a simulator.

This is the middle rung between the hand-rolled analytic proxy and SoundSpaces:
exact image sources on a 3D room extruded from the same floorplan, with
frequency-independent absorption taken from the proxy's own material mix. It
needs no simulator and no GPU, so it is the option that scales to a new scene
without a render pass -- which is exactly what it is here to test.

It writes the same ``.npz`` layout as ``soundspaces_render.py``, so
``acoustic_grid_probe.py`` scores both through identical code and the numbers
are directly comparable.

    python scripts/pyroom_grid.py --scene office_4 --max-order 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

RING_ANGLES_RAD = np.array(
    [np.pi, 4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi, np.pi / 3, 2 * np.pi / 3]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-order", type=int, default=3)
    p.add_argument("--ring-radius-m", type=float, default=0.05)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--clearance-m", type=float, default=0.25)
    p.add_argument("--polygon-epsilon-m", type=float, default=0.10,
                   help="contour simplification; the image-source model needs a "
                        "polygon, not a staircase of 1 cm pixels")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--progress-every", type=int, default=200)
    return p.parse_args()


def room_polygon(occ: np.ndarray, grid, epsilon_m: float) -> np.ndarray:
    """Largest free-space contour, in floorplan metres, counter-clockwise."""
    import cv2

    free = (occ == 255).astype(np.uint8)
    contours, _ = cv2.findContours(free, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no free-space contour in the map")
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_m / grid.map_resolution_m
    poly = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(np.float64)   # (K, 2) map px
    centre = np.array([grid.map_shape[1] / 2.0, grid.map_shape[0] / 2.0])
    metric = (poly - centre) * grid.map_resolution_m
    # pyroomacoustics wants counter-clockwise corners
    area = 0.5 * np.sum(metric[:, 0] * np.roll(metric[:, 1], -1)
                        - np.roll(metric[:, 0], -1) * metric[:, 1])
    if area < 0:
        metric = metric[::-1]
    return metric.T                                                     # (2, K)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import pyroomacoustics as pra
    from track1_core.floorplan import PoseGrid, valid_pose_mask

    root = Path(args.dataset_root)
    scene = args.scene
    fp = json.loads((root / "floorplan_proxy" / scene / "floorplan.json").read_text())
    rir_dir = root / "rir" / args.collection / "floorplan_closed" / scene
    sample = sorted(d for d in os.listdir(rir_dir) if d.startswith("pose_"))[0]
    md = json.loads((rir_dir / sample / "rir_metadata.json").read_text())

    height = float(fp["room_height_m"])
    src_h = float(md["array_center_habitat"][1]) - float(fp["z_floor"])
    absorption = float(fp["material_mix"]["absorption_1k"])
    print(f"[pra] {scene}: room height {height:.3f} m, source height {src_h:.3f} m, "
          f"absorption {absorption:.3f}")

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    grid = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, grid, clearance_m=args.clearance_m)
    rows, cols = np.nonzero(mask)
    if args.limit:
        rows, cols = rows[: args.limit], cols[: args.limit]
    metric = grid.grid_to_metric(np.stack([cols, rows], axis=-1))
    print(f"[pra] {len(rows)} candidate cells, ISM order {args.max_order}")

    corners = room_polygon(occ, grid, args.polygon_epsilon_m)
    print(f"[pra] room polygon: {corners.shape[1]} corners")
    material = pra.Material(absorption)

    def render(xy: np.ndarray) -> np.ndarray:
        room = pra.Room.from_corners(corners, fs=args.sample_rate, materials=material,
                                     max_order=args.max_order)
        room.extrude(height, materials=material)
        mics = np.stack([
            np.array([xy[0] + args.ring_radius_m * np.cos(a),
                      xy[1] + args.ring_radius_m * np.sin(a), src_h])
            for a in RING_ANGLES_RAD
        ], axis=-1)
        room.add_microphone_array(pra.MicrophoneArray(mics, args.sample_rate))
        room.add_source([xy[0], xy[1], src_h])
        room.compute_rir()
        n = max(len(r[0]) for r in room.rir)
        return np.stack([np.pad(r[0], (0, n - len(r[0]))) for r in room.rir]).astype(np.float32)

    rirs, index, start, skipped = [], [], time.time(), 0
    for i, xy in enumerate(metric):
        try:
            rirs.append(render(xy))
            index.append((int(rows[i]), int(cols[i]), 0))
        except Exception:
            # a candidate can fall outside the simplified polygon even though it
            # is inside the rasterised map; drop it rather than fake a response
            skipped += 1
        if (i + 1) % args.progress_every == 0:
            rate = (i + 1) / (time.time() - start)
            print(f"[pra] {i+1}/{len(metric)}  {rate:.1f} cells/s  "
                  f"eta {(len(metric)-i-1)/rate/60:.0f} min  skipped {skipped}")

    if not rirs:
        print("[pra] every candidate failed; the polygon is probably wrong")
        return 1
    length = min(r.shape[1] for r in rirs)
    stack = np.stack([r[:, :length] for r in rirs])
    out = args.out or REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}_pra.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rir=stack, index=np.array(index, dtype=np.int32),
                        grid=json.dumps(grid.describe()),
                        config=json.dumps(vars(args), default=str))
    print(f"[pra] wrote {out}  rir{stack.shape}  skipped {skipped}  "
          f"in {(time.time()-start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
