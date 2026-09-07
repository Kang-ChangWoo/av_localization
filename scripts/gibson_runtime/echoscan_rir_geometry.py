"""Geometry helpers for controlled F3Loc-to-EchoScan acoustic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


F3LOC_RESOLUTION_M = 0.01
PROXY_GRID_RESOLUTION_M = 0.10
ECHOSCAN_RING_ANGLES_RAD = np.array(
    [3 * np.pi / 3, 4 * np.pi / 3, 5 * np.pi / 3, 6 * np.pi / 3, np.pi / 3, 2 * np.pi / 3],
    dtype=np.float64,
)


@dataclass(frozen=True)
class GridComponent:
    free_mask: np.ndarray
    resolution_m: float
    origin_xy_m: tuple[float, float]
    touches_border: bool


def echoscan_ring_offsets_habitat(yaw_rad: float, radius_m: float = 0.05) -> np.ndarray:
    """Map EchoScan's channel ordering into the established Habitat stage frame."""
    if radius_m <= 0:
        raise ValueError("Microphone radius must be positive.")
    angles = ECHOSCAN_RING_ANGLES_RAD + float(yaw_rad)
    return np.stack(
        [radius_m * np.cos(angles), np.zeros_like(angles), -radius_m * np.sin(angles)], axis=1
    ).astype(np.float32)


def seeded_free_component(
    map_array: np.ndarray,
    *,
    pose_xy_m: tuple[float, float],
    grid_resolution_m: float = PROXY_GRID_RESOLUTION_M,
) -> GridComponent:
    """Extract the closed map component containing one F3Loc source pose."""
    if map_array.ndim != 2 or map_array.size == 0:
        raise ValueError("Expected a non-empty grayscale F3Loc map.")
    cells_per_axis = int(round(grid_resolution_m / F3LOC_RESOLUTION_M))
    if cells_per_axis <= 0 or not np.isclose(cells_per_axis * F3LOC_RESOLUTION_M, grid_resolution_m):
        raise ValueError("Proxy grid resolution must be an integer multiple of 0.01 m.")
    height, width = map_array.shape
    trimmed_height = height - height % cells_per_axis
    trimmed_width = width - width % cells_per_axis
    if trimmed_height == 0 or trimmed_width == 0:
        raise ValueError("Map is too small for the requested proxy grid.")
    dark = map_array[:trimmed_height, :trimmed_width] < 128
    blocked = dark.reshape(
        trimmed_height // cells_per_axis,
        cells_per_axis,
        trimmed_width // cells_per_axis,
        cells_per_axis,
    ).any(axis=(1, 3))
    labels_count, labels, _, _ = cv2.connectedComponentsWithStats((~blocked).astype(np.uint8), connectivity=8)
    x_pixel = pose_xy_m[0] / F3LOC_RESOLUTION_M + width / 2
    y_pixel = pose_xy_m[1] / F3LOC_RESOLUTION_M + height / 2
    col = int(np.floor(x_pixel / cells_per_axis))
    row = int(np.floor(y_pixel / cells_per_axis))
    if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
        raise ValueError("F3Loc pose is outside the proxy map grid.")
    label = int(labels[row, col])
    if label == 0 or label >= labels_count:
        raise ValueError("F3Loc pose falls on a proxy wall cell.")
    free_mask = labels == label
    touches_border = bool(
        free_mask[0, :].any()
        or free_mask[-1, :].any()
        or free_mask[:, 0].any()
        or free_mask[:, -1].any()
    )
    if touches_border:
        raise ValueError("Selected free component reaches the map border and cannot form a closed proxy.")
    return GridComponent(
        free_mask=free_mask,
        resolution_m=grid_resolution_m,
        origin_xy_m=(-width * F3LOC_RESOLUTION_M / 2, -height * F3LOC_RESOLUTION_M / 2),
        touches_border=touches_border,
    )


def write_closed_proxy_obj(
    component: GridComponent, output_path: Path, ceiling_height_m: float = 2.5
) -> dict[str, int | float | str]:
    """Write a double-sided floor, ceiling, and boundary-wall OBJ in raw z-up axes."""
    if ceiling_height_m <= 0:
        raise ValueError("Proxy ceiling height must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def quad(points: list[tuple[float, float, float]]) -> None:
        start = len(vertices) + 1
        vertices.extend(points)
        faces.extend(
            [
                (start, start + 1, start + 2),
                (start, start + 2, start + 3),
                (start + 2, start + 1, start),
                (start + 3, start + 2, start),
                (start + 3, start, start + 1),
                (start + 1, start + 2, start + 3),
            ]
        )

    free = component.free_mask
    resolution = component.resolution_m
    origin_x, origin_y = component.origin_xy_m
    for row, col in np.argwhere(free):
        x0 = origin_x + col * resolution
        x1 = x0 + resolution
        y0 = origin_y + row * resolution
        y1 = y0 + resolution
        quad([(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)])
        quad([(x0, y0, ceiling_height_m), (x0, y1, ceiling_height_m), (x1, y1, ceiling_height_m), (x1, y0, ceiling_height_m)])
        for delta_row, delta_col, wall in (
            (0, -1, [(x0, y0, 0.0), (x0, y1, 0.0), (x0, y1, ceiling_height_m), (x0, y0, ceiling_height_m)]),
            (0, 1, [(x1, y1, 0.0), (x1, y0, 0.0), (x1, y0, ceiling_height_m), (x1, y1, ceiling_height_m)]),
            (-1, 0, [(x1, y0, 0.0), (x0, y0, 0.0), (x0, y0, ceiling_height_m), (x1, y0, ceiling_height_m)]),
            (1, 0, [(x0, y1, 0.0), (x1, y1, 0.0), (x1, y1, ceiling_height_m), (x0, y1, ceiling_height_m)]),
        ):
            neighbor_row, neighbor_col = row + delta_row, col + delta_col
            if not (0 <= neighbor_row < free.shape[0] and 0 <= neighbor_col < free.shape[1]) or not free[neighbor_row, neighbor_col]:
                quad(wall)

    with output_path.open("w", encoding="ascii") as handle:
        handle.write("# F3Loc structural control proxy; generated, not a released asset.\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            handle.write(f"f {a} {b} {c}\n")
    return {
        "path": str(output_path),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "free_cell_count": int(free.sum()),
        "grid_resolution_m": resolution,
        "ceiling_height_m": ceiling_height_m,
    }
