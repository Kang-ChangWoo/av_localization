"""The pose grid every likelihood, the posterior, and evaluation share.

``docs/data_contracts.md`` fixes the contract this implements: a ``(H, W, O)``
float32 volume indexed ``(grid_y, grid_x, yaw_bin)``, 0.1 m/cell, 36
counter-clockwise yaw bins, with pose vectors written ``(x, y, yaw)``.

Coordinate conversion is explicit and carries its metadata. The crop offset,
map resolution, and yaw-zero convention are fields on the grid, never implicit
constants, because the visual DESDF cache, the acoustic likelihood, and the
ground-truth poses each arrive in a different frame and a silent mismatch
between them is indistinguishable from a bad model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAP_RESOLUTION_M = 0.01
GRID_RESOLUTION_M = 0.1
ORIENTATION_BINS = 36


@dataclass(frozen=True)
class PoseGrid:
    """Metric/grid conversion for one scene.

    height, width      grid cells (``desdf`` rows and columns)
    left, top          crop offset of the grid inside ``map.png``, in map pixels
    map_shape          ``(rows, cols)`` of ``map.png``
    map_resolution_m   metres per map pixel
    grid_resolution_m  metres per grid cell
    orientation_bins   yaw bins, counter-clockwise, bin 0 at yaw 0
    """

    height: int
    width: int
    left: int
    top: int
    map_shape: tuple[int, int]
    map_resolution_m: float = MAP_RESOLUTION_M
    grid_resolution_m: float = GRID_RESOLUTION_M
    orientation_bins: int = ORIENTATION_BINS

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_desdf(cls, desdf: dict, map_shape: tuple[int, int], **kwargs) -> "PoseGrid":
        """Build the grid a DESDF cache defines."""
        volume = desdf["desdf"]
        if volume.ndim != 3:
            raise ValueError(f"desdf must be (H, W, O), got {volume.shape}")
        return cls(
            height=int(volume.shape[0]),
            width=int(volume.shape[1]),
            left=int(desdf["l"]),
            top=int(desdf["t"]),
            map_shape=(int(map_shape[0]), int(map_shape[1])),
            orientation_bins=int(volume.shape[2]),
            **kwargs,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.orientation_bins)

    @property
    def cells_per_map_pixel(self) -> float:
        return self.grid_resolution_m / self.map_resolution_m

    # ---- grid <-> map pixels --------------------------------------------
    def grid_to_map(self, grid_xy: np.ndarray) -> np.ndarray:
        """Grid ``(x, y)`` -> map pixel ``(col, row)``."""
        grid_xy = np.asarray(grid_xy, dtype=np.float64)
        offset = np.array([self.left, self.top], dtype=np.float64)
        return grid_xy * self.cells_per_map_pixel + offset

    def map_to_grid(self, map_xy: np.ndarray) -> np.ndarray:
        """Map pixel ``(col, row)`` -> grid ``(x, y)``."""
        map_xy = np.asarray(map_xy, dtype=np.float64)
        offset = np.array([self.left, self.top], dtype=np.float64)
        return (map_xy - offset) / self.cells_per_map_pixel

    # ---- grid <-> world metres ------------------------------------------
    def grid_to_metric(self, grid_xy: np.ndarray) -> np.ndarray:
        """Grid ``(x, y)`` -> world metres ``(x, y)``, origin at the map centre."""
        map_xy = self.grid_to_map(grid_xy)
        centre = np.array([self.map_shape[1] / 2.0, self.map_shape[0] / 2.0])
        return (map_xy - centre) * self.map_resolution_m

    def metric_to_grid(self, metric_xy: np.ndarray) -> np.ndarray:
        """World metres ``(x, y)`` -> grid ``(x, y)``."""
        metric_xy = np.asarray(metric_xy, dtype=np.float64)
        centre = np.array([self.map_shape[1] / 2.0, self.map_shape[0] / 2.0])
        return self.map_to_grid(metric_xy / self.map_resolution_m + centre)

    # ---- yaw -------------------------------------------------------------
    def bin_to_yaw(self, yaw_bin: np.ndarray | int) -> np.ndarray:
        """Yaw bin -> radians, counter-clockwise, bin 0 at yaw 0."""
        return np.asarray(yaw_bin, dtype=np.float64) / self.orientation_bins * 2 * np.pi

    def yaw_to_bin(self, yaw_rad: np.ndarray | float) -> np.ndarray:
        """Radians -> nearest yaw bin, wrapped into ``[0, orientation_bins)``."""
        yaw = np.asarray(yaw_rad, dtype=np.float64) % (2 * np.pi)
        return np.round(yaw / (2 * np.pi) * self.orientation_bins).astype(int) % self.orientation_bins

    # ---- poses -----------------------------------------------------------
    def pose_metric_to_grid(self, pose: np.ndarray) -> np.ndarray:
        """World ``(x, y, yaw)`` -> grid ``(x, y, yaw)``; yaw stays in radians."""
        pose = np.asarray(pose, dtype=np.float64)
        xy = self.metric_to_grid(pose[..., :2])
        return np.concatenate([xy, pose[..., 2:3]], axis=-1)

    def pose_grid_to_metric(self, pose: np.ndarray) -> np.ndarray:
        """Grid ``(x, y, yaw)`` -> world ``(x, y, yaw)``."""
        pose = np.asarray(pose, dtype=np.float64)
        xy = self.grid_to_metric(pose[..., :2])
        return np.concatenate([xy, pose[..., 2:3]], axis=-1)

    def translation_error_m(self, pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        """Distance between two grid-frame poses, in metres."""
        a = np.asarray(pose_a, dtype=np.float64)[:2]
        b = np.asarray(pose_b, dtype=np.float64)[:2]
        return float(np.linalg.norm(a - b) * self.grid_resolution_m)

    @staticmethod
    def orientation_error_deg(yaw_a: float, yaw_b: float) -> float:
        """Wrapped minimum angular difference, in degrees."""
        d = (float(yaw_a) - float(yaw_b)) % (2 * np.pi)
        return min(d, 2 * np.pi - d) / np.pi * 180.0

    def describe(self) -> dict:
        """Metadata to record alongside any cache built on this grid."""
        return {
            "shape": list(self.shape),
            "left": self.left,
            "top": self.top,
            "map_shape": list(self.map_shape),
            "map_resolution_m": self.map_resolution_m,
            "grid_resolution_m": self.grid_resolution_m,
            "orientation_bins": self.orientation_bins,
            "index_order": "(grid_y, grid_x, yaw_bin)",
            "pose_order": "(x, y, yaw)",
            "yaw_zero": "+x axis, counter-clockwise",
        }
