"""Contract tests for the shared pose grid and valid-pose mask."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path(os.environ.get("AVFPLOC_ECHOLOC_DATA", "/root/storage/echoloc_dataset"))
COLLECTION = "replica_f"

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "desdf").is_dir(), reason=f"dataset not available at {DATA_ROOT}"
)


def _scene():
    import cv2

    from track1_core.floorplan import PoseGrid

    scene = sorted(os.listdir(DATA_ROOT / "desdf"))[0]
    desdf = np.load(DATA_ROOT / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(DATA_ROOT / COLLECTION / scene / "map.png"))[:, :, 0]
    return scene, desdf, occ, PoseGrid.from_desdf(desdf, occ.shape)


def test_grid_matches_the_desdf_it_came_from():
    _scene_name, desdf, occ, grid = _scene()
    assert grid.shape == desdf["desdf"].shape
    assert grid.orientation_bins == 36
    assert grid.map_shape == occ.shape


def test_grid_map_roundtrip():
    _s, _d, _o, grid = _scene()
    pts = np.array([[0.0, 0.0], [10.0, 20.0], [grid.width - 1.0, grid.height - 1.0]])
    np.testing.assert_allclose(grid.map_to_grid(grid.grid_to_map(pts)), pts, atol=1e-9)


def test_grid_metric_roundtrip():
    _s, _d, _o, grid = _scene()
    pts = np.array([[0.0, 0.0], [5.5, 12.25], [grid.width - 1.0, grid.height - 1.0]])
    np.testing.assert_allclose(grid.metric_to_grid(grid.grid_to_metric(pts)), pts, atol=1e-6)


def test_grid_conversion_matches_the_upstream_formula():
    """The evaluation code converts poses inline; the grid must agree with it."""
    _s, desdf, occ, grid = _scene()
    h, w = occ.shape
    x_m, y_m, yaw = -1.234, 2.345, 0.5
    # what scripts/eval_visual.py computes
    x_map = x_m / 0.01 + w / 2
    y_map = y_m / 0.01 + h / 2
    expect = np.array([(x_map - desdf["l"]) / 10, (y_map - desdf["t"]) / 10])
    got = grid.pose_metric_to_grid(np.array([x_m, y_m, yaw]))
    np.testing.assert_allclose(got[:2], expect, atol=1e-9)
    assert got[2] == yaw


def test_yaw_bins_are_ccw_and_wrap():
    from track1_core.floorplan import PoseGrid

    _s, _d, _o, grid = _scene()
    assert grid.yaw_to_bin(0.0) == 0
    assert grid.yaw_to_bin(2 * np.pi) == 0
    assert grid.yaw_to_bin(np.pi) == 18
    np.testing.assert_allclose(grid.bin_to_yaw(9), np.pi / 2)
    assert PoseGrid.orientation_error_deg(0.1, 2 * np.pi - 0.1) == pytest.approx(11.459, abs=1e-2)


def test_valid_mask_is_a_strict_subset_of_free_space():
    from track1_core.floorplan import free_space, valid_pose_mask

    _s, _d, occ, grid = _scene()
    mask = valid_pose_mask(occ, grid)
    assert mask.shape == (grid.height, grid.width)
    assert mask.dtype == bool
    assert 0 < mask.sum() < mask.size

    free = free_space(occ)
    rows, cols = np.nonzero(mask)
    map_xy = grid.grid_to_map(np.stack([cols, rows], axis=-1))
    px = np.rint(map_xy).astype(int)
    assert free[px[:, 1], px[:, 0]].all(), "a valid cell landed on an obstacle pixel"


def test_ground_truth_poses_are_not_masked_out():
    """Poses were sampled with >= 0.25 m clearance, so the mask must keep them."""
    from track1_core.floorplan import valid_pose_mask

    scene, _d, occ, grid = _scene()
    mask = valid_pose_mask(occ, grid)
    rows = [l.split() for l in open(DATA_ROOT / COLLECTION / scene / "poses.txt") if l.strip()]
    poses = np.array([[float(v) for v in r] for r in rows])[:, :3]
    grid_poses = grid.pose_metric_to_grid(poses)
    gx = np.rint(grid_poses[:, 0]).astype(int)
    gy = np.rint(grid_poses[:, 1]).astype(int)
    inside = (gx >= 0) & (gx < grid.width) & (gy >= 0) & (gy < grid.height)
    assert inside.mean() > 0.99, "ground-truth poses fall outside the grid"
    kept = mask[gy[inside], gx[inside]].mean()
    assert kept > 0.95, f"mask rejects {100*(1-kept):.1f}% of ground-truth poses"


def test_apply_mask_and_normalize():
    from track1_core.floorplan import apply_mask, normalize, valid_pose_mask

    _s, desdf, occ, grid = _scene()
    mask = valid_pose_mask(occ, grid)
    vol = np.random.rand(*grid.shape).astype(np.float32)

    masked = apply_mask(vol, mask)
    assert (masked[~mask] == 0).all()
    log_masked = apply_mask(vol, mask, log=True)
    assert np.isneginf(log_masked[~mask]).all()

    post = normalize(vol, mask)
    assert post.sum() == pytest.approx(1.0, abs=1e-5)
    assert (post[~mask] == 0).all()
