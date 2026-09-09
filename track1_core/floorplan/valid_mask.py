"""The single valid-pose mask every modality must score against.

``docs/decision_log.md`` (2026-07-11, "Candidate validity") fixes why this
exists: a wall-only floorplan range field happily scores cells that are inside
walls or outside the building, and without one shared mask an apparent acoustic
gain can come from suppressing those invalid candidates rather than from any
real structural complementarity. Visual likelihood, acoustic likelihood, the
fused posterior, its normalization, its argmax, and every diagnostic rank must
use the same mask.

The default clearance matches how the dataset's poses were sampled
(``dataset_meta.json``: "clearance >= 0.25 m from nearest wall pixel"), so a
ground-truth pose is never itself masked out.
"""

from __future__ import annotations

import cv2
import numpy as np

from track1_core.floorplan.pose_grid import PoseGrid

FREE_VALUE = 255
DEFAULT_CLEARANCE_M = 0.25


def free_space(occupancy: np.ndarray) -> np.ndarray:
    """Boolean free-space mask at map resolution.

    Free space is exactly ``255``; anything else is an obstacle. This mirrors
    the ray caster, which stops at the first pixel that is not 255, so an
    anti-aliased wall edge is a wall here too.
    """
    if occupancy.ndim == 3:
        occupancy = occupancy[:, :, 0]
    return occupancy == FREE_VALUE


def clearance_map_m(occupancy: np.ndarray, map_resolution_m: float) -> np.ndarray:
    """Distance from each map pixel to the nearest obstacle, in metres."""
    free = free_space(occupancy).astype(np.uint8)
    # distanceTransform measures distance to the nearest zero pixel
    return cv2.distanceTransform(free, cv2.DIST_L2, 5) * map_resolution_m


def valid_pose_mask(
    occupancy: np.ndarray,
    grid: PoseGrid,
    clearance_m: float = DEFAULT_CLEARANCE_M,
) -> np.ndarray:
    """``(H, W)`` bool mask of candidate positions worth scoring.

    A grid cell is valid when the map pixel at its centre is free and at least
    ``clearance_m`` from the nearest obstacle. Sampling at cell centres (rather
    than any-free within the cell) keeps the mask consistent with the DESDF,
    which is also cast from cell centres.
    """
    clearance = clearance_map_m(occupancy, grid.map_resolution_m)
    rows, cols = np.mgrid[0 : grid.height, 0 : grid.width]
    map_xy = grid.grid_to_map(np.stack([cols.ravel(), rows.ravel()], axis=-1))
    col_px = np.rint(map_xy[:, 0]).astype(int)
    row_px = np.rint(map_xy[:, 1]).astype(int)

    inside = (
        (row_px >= 0)
        & (row_px < clearance.shape[0])
        & (col_px >= 0)
        & (col_px < clearance.shape[1])
    )
    mask = np.zeros(row_px.shape, dtype=bool)
    mask[inside] = clearance[row_px[inside], col_px[inside]] >= clearance_m
    return mask.reshape(grid.height, grid.width)


def apply_mask(volume: np.ndarray, mask: np.ndarray, *, log: bool = False) -> np.ndarray:
    """Zero (or ``-inf``) every invalid candidate before normalization.

    ``volume`` is ``(H, W)`` or ``(H, W, O)``; ``mask`` is ``(H, W)``.
    """
    if mask.shape != volume.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match volume {volume.shape[:2]}")
    out = np.array(volume, dtype=np.float32, copy=True)
    invalid = ~mask
    out[invalid] = -np.inf if log else 0.0
    return out


def normalize(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Normalise a non-negative score volume into a posterior over valid cells."""
    out = apply_mask(volume, mask) if mask is not None else np.array(volume, np.float32, copy=True)
    total = float(out.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("score volume has no positive mass on the valid support")
    return out / total


def mask_summary(mask: np.ndarray, grid: PoseGrid) -> dict:
    """Numbers worth recording next to any experiment that uses the mask."""
    n_valid = int(mask.sum())
    return {
        "grid_shape": [grid.height, grid.width],
        "valid_cells": n_valid,
        "total_cells": int(mask.size),
        "valid_fraction": n_valid / float(mask.size),
        "valid_area_m2": n_valid * grid.grid_resolution_m**2,
        "candidate_poses": n_valid * grid.orientation_bins,
    }
