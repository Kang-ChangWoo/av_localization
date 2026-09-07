"""Deterministic 2D acoustic proxy utilities for F3Loc floorplans.

The proxy is intentionally limited to direct sound plus first wall returns from
a uniform, non-random radial ray fan.  It is a map-topology likelihood model,
not a material-faithful room-acoustics simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


F3LOC_MAP_RESOLUTION_M = 0.01
ECHOSCAN_RING_ANGLES_RAD = np.array(
    [np.pi, 4 * np.pi / 3, 5 * np.pi / 3, 0.0, np.pi / 3, 2 * np.pi / 3],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PlanarRirConfig:
    """Fixed, explicitly non-material parameters for a planar proxy RIR."""

    sample_rate_hz: int = 8000
    sound_speed_m_s: float = 343.0
    ray_count: int = 72
    max_time_ms: float = 100.0
    wall_reflection_gain: float = 0.7
    receiver_radius_m: float = 0.05
    direct_gain: float = 0.2
    energy_window_ms: float = 2.0
    peak_count: int = 4

    @property
    def sample_count(self) -> int:
        return int(round(self.sample_rate_hz * self.max_time_ms / 1000.0))

    @property
    def energy_window_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.energy_window_ms / 1000.0))


@dataclass(frozen=True)
class CandidateState:
    """One F3Loc-compatible spatial/orientation hypothesis."""

    row_index: int
    col_index: int
    yaw_index: int
    x_m: float
    y_m: float
    yaw_rad: float
    quantization_error_m: float = 0.0

    @property
    def identifier(self) -> str:
        return f"r{self.row_index}_c{self.col_index}_o{self.yaw_index}"


def map_xy_to_row_col(map_shape: tuple[int, int], xy_m: tuple[float, float]) -> tuple[float, float]:
    """Convert F3Loc planar metres to map row/column coordinates."""
    height, width = map_shape
    x_m, y_m = xy_m
    return (
        y_m / F3LOC_MAP_RESOLUTION_M + height / 2.0,
        x_m / F3LOC_MAP_RESOLUTION_M + width / 2.0,
    )


def row_col_to_map_xy(map_shape: tuple[int, int], row_col: tuple[float, float]) -> tuple[float, float]:
    """Convert map row/column coordinates to F3Loc planar metres."""
    height, width = map_shape
    row, col = row_col
    return (
        (col - width / 2.0) * F3LOC_MAP_RESOLUTION_M,
        (row - height / 2.0) * F3LOC_MAP_RESOLUTION_M,
    )


def microphone_offsets_xy(yaw_rad: float, radius_m: float = 0.05) -> np.ndarray:
    """Return the established six-channel microphone ring in F3Loc axes."""
    if radius_m <= 0.0:
        raise ValueError("Microphone ring radius must be positive.")
    angles = ECHOSCAN_RING_ANGLES_RAD + float(yaw_rad)
    return np.stack((radius_m * np.cos(angles), radius_m * np.sin(angles)), axis=1).astype(np.float64)


def _validate_occupancy(occupancy: np.ndarray) -> np.ndarray:
    map_array = np.asarray(occupancy)
    if map_array.ndim != 2 or map_array.size == 0:
        raise ValueError("Expected a non-empty grayscale occupancy map.")
    if not np.isfinite(map_array).all():
        raise ValueError("Occupancy map must be finite.")
    return map_array


def _candidate_row_col(occupancy: np.ndarray, candidate_xy_m: tuple[float, float]) -> tuple[float, float]:
    row, col = map_xy_to_row_col(occupancy.shape, candidate_xy_m)
    height, width = occupancy.shape
    row_index, col_index = int(np.floor(row)), int(np.floor(col))
    if not (0 <= row_index < height and 0 <= col_index < width):
        raise ValueError("Candidate is outside the F3Loc map.")
    if occupancy[row_index, col_index] < 128:
        raise ValueError("Candidate falls in a blocked F3Loc map cell.")
    return row, col


def _first_wall_distance_m(occupancy: np.ndarray, row: float, col: float, angle_rad: float) -> float | None:
    """Return first blocked-cell crossing with exact grid DDA traversal."""
    direction_col = float(np.cos(angle_rad))
    direction_row = float(np.sin(angle_rad))
    cell_row, cell_col = int(np.floor(row)), int(np.floor(col))
    height, width = occupancy.shape

    if abs(direction_col) < 1e-12:
        t_delta_col = np.inf
        t_max_col = np.inf
    elif direction_col > 0.0:
        t_delta_col = 1.0 / direction_col
        t_max_col = (cell_col + 1.0 - col) / direction_col
    else:
        t_delta_col = -1.0 / direction_col
        t_max_col = (col - cell_col) / -direction_col

    if abs(direction_row) < 1e-12:
        t_delta_row = np.inf
        t_max_row = np.inf
    elif direction_row > 0.0:
        t_delta_row = 1.0 / direction_row
        t_max_row = (cell_row + 1.0 - row) / direction_row
    else:
        t_delta_row = -1.0 / direction_row
        t_max_row = (row - cell_row) / -direction_row

    max_steps = height + width
    for _ in range(max_steps):
        if t_max_col < t_max_row:
            cell_col += 1 if direction_col > 0.0 else -1
            distance_px = t_max_col
            t_max_col += t_delta_col
        else:
            cell_row += 1 if direction_row > 0.0 else -1
            distance_px = t_max_row
            t_max_row += t_delta_row
        if not (0 <= cell_row < height and 0 <= cell_col < width):
            return None
        if occupancy[cell_row, cell_col] < 128:
            return float(distance_px * F3LOC_MAP_RESOLUTION_M)
    return None


def wall_distances_from_candidate(
    occupancy: np.ndarray,
    candidate_xy_m: tuple[float, float],
    config: PlanarRirConfig,
) -> np.ndarray:
    """Return first-wall distances for the fixed, deterministic radial ray fan."""
    map_array = _validate_occupancy(occupancy)
    if config.ray_count <= 0:
        raise ValueError("Ray count must be positive.")
    row, col = _candidate_row_col(map_array, candidate_xy_m)
    angles = np.arange(config.ray_count, dtype=np.float64) * (2.0 * np.pi / config.ray_count)
    distances = np.full(config.ray_count, np.nan, dtype=np.float64)
    for index, angle in enumerate(angles):
        distance_m = _first_wall_distance_m(map_array, row, col, float(angle))
        if distance_m is not None:
            distances[index] = distance_m
    return distances


def _add_impulse(signal: np.ndarray, sample_index: float, amplitude: float) -> None:
    lower = int(np.floor(sample_index))
    fraction = sample_index - lower
    if 0 <= lower < signal.shape[0]:
        signal[lower] += amplitude * (1.0 - fraction)
    if 0 <= lower + 1 < signal.shape[0]:
        signal[lower + 1] += amplitude * fraction


def planar_rir_from_wall_distances(
    candidate_xy_m: tuple[float, float],
    yaw_rad: float,
    wall_distances_m: np.ndarray,
    config: PlanarRirConfig,
) -> np.ndarray:
    """Synthesize a finite `(6, samples)` proxy RIR from precomputed wall hits."""
    distances = np.asarray(wall_distances_m, dtype=np.float64)
    if distances.shape != (config.ray_count,):
        raise ValueError("Wall-distance count must equal the configured ray count.")
    if not np.isfinite(np.asarray(candidate_xy_m, dtype=np.float64)).all():
        raise ValueError("Candidate must contain only finite coordinates.")
    offsets = microphone_offsets_xy(yaw_rad, config.receiver_radius_m)
    rir = np.zeros((6, config.sample_count), dtype=np.float32)

    for channel, offset in enumerate(offsets):
        direct_distance = float(np.linalg.norm(offset))
        _add_impulse(
            rir[channel],
            direct_distance / config.sound_speed_m_s * config.sample_rate_hz,
            config.direct_gain,
        )

    angles = np.arange(config.ray_count, dtype=np.float64) * (2.0 * np.pi / config.ray_count)
    for angle, wall_distance_m in zip(angles, distances):
        if not np.isfinite(wall_distance_m) or wall_distance_m <= 0.0:
            continue
        wall_xy = np.array(
            [
                candidate_xy_m[0] + wall_distance_m * np.cos(angle),
                candidate_xy_m[1] + wall_distance_m * np.sin(angle),
            ],
            dtype=np.float64,
        )
        for channel, offset in enumerate(offsets):
            microphone_xy = np.asarray(candidate_xy_m, dtype=np.float64) + offset
            path_distance_m = wall_distance_m + float(np.linalg.norm(wall_xy - microphone_xy))
            gain = config.wall_reflection_gain / np.sqrt(max(path_distance_m, 0.1)) / config.ray_count
            _add_impulse(
                rir[channel],
                path_distance_m / config.sound_speed_m_s * config.sample_rate_hz,
                gain,
            )
    if not np.isfinite(rir).all():
        raise RuntimeError("Planar proxy RIR contains non-finite values.")
    return rir


def planar_rir_from_candidate(
    occupancy: np.ndarray,
    candidate_xy_m: tuple[float, float],
    yaw_rad: float,
    config: PlanarRirConfig,
) -> np.ndarray:
    """Synthesize a finite `(6, samples)` planar proxy RIR for one candidate.

    The source sits at the candidate centre. Each radial source ray contributes
    the source-wall-microphone path to all receiver-ring channels. First-wall
    hits are obtained from the released binary map; no wall material, ceiling,
    furniture, or higher-order reflection is inferred.
    """
    if config.sample_count <= 0:
        raise ValueError("Sample count must be positive.")
    distances = wall_distances_from_candidate(occupancy, candidate_xy_m, config)
    return planar_rir_from_wall_distances(candidate_xy_m, yaw_rad, distances, config)


def early_energy_envelope(rir: np.ndarray, config: PlanarRirConfig) -> np.ndarray:
    """Return channel-wise 2 ms energy envelopes over the configured early window."""
    values = np.asarray(rir, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 6:
        raise ValueError("Expected a finite channel-first `(6, samples)` RIR.")
    if not np.isfinite(values).all():
        raise ValueError("RIR must contain only finite values.")
    window = config.energy_window_samples
    sample_count = min(values.shape[1], config.sample_count)
    sample_count -= sample_count % window
    if sample_count == 0:
        raise ValueError("RIR is shorter than one energy window.")
    early = values[:, :sample_count]
    return (early.reshape(6, -1, window) ** 2).sum(axis=2)


def _normalize_energy(envelope: np.ndarray) -> np.ndarray:
    peak = np.maximum(envelope.max(axis=1, keepdims=True), 1e-12)
    log_energy = np.log1p(envelope / peak)
    norm = np.maximum(np.linalg.norm(log_energy, axis=1, keepdims=True), 1e-12)
    return log_energy / norm


def _peak_time_distance(reference: np.ndarray, candidate: np.ndarray, peak_count: int) -> float:
    distances = []
    for ref_channel, candidate_channel in zip(reference, candidate):
        ref_indices = np.argsort(ref_channel)[-peak_count:]
        candidate_indices = np.argsort(candidate_channel)[-peak_count:]
        distances.append(np.abs(ref_indices[:, None] - candidate_indices[None, :]).min(axis=1).mean())
    return float(np.mean(distances) / max(reference.shape[1] - 1, 1))


def score_planar_rir(observed: np.ndarray, proxy: np.ndarray, config: PlanarRirConfig) -> dict[str, float]:
    """Score a proxy by normalized early energy and sparse peak-time agreement."""
    observed_energy = early_energy_envelope(observed, config)
    proxy_energy = early_energy_envelope(proxy, config)
    frame_count = min(observed_energy.shape[1], proxy_energy.shape[1])
    if frame_count == 0:
        raise ValueError("Observed and proxy RIRs do not share an early-time window.")
    observed_normalized = _normalize_energy(observed_energy[:, :frame_count])
    proxy_normalized = _normalize_energy(proxy_energy[:, :frame_count])
    envelope_mse = float(np.mean((observed_normalized - proxy_normalized) ** 2))
    peak_time_distance = _peak_time_distance(
        observed_energy[:, :frame_count], proxy_energy[:, :frame_count], config.peak_count
    )
    mismatch = 0.8 * envelope_mse + 0.2 * peak_time_distance
    return {
        "envelope_mse": envelope_mse,
        "peak_time_distance": peak_time_distance,
        "mismatch": mismatch,
        "likelihood": float(np.exp(-mismatch)),
    }


def build_local_candidate_lattice(
    occupancy: np.ndarray,
    desdf: Mapping[str, Any],
    pose_xy_yaw: tuple[float, float, float],
    radius_m: float,
) -> tuple[list[CandidateState], CandidateState]:
    """Build valid local DESDF candidates and identify the quantized GT state."""
    map_array = _validate_occupancy(occupancy)
    if radius_m < 0.0:
        raise ValueError("Candidate radius must be non-negative.")
    desdf_values = np.asarray(desdf.get("desdf"))
    if desdf_values.ndim != 3 or desdf_values.shape[2] <= 0:
        raise ValueError("DESDF must have shape `(rows, cols, orientations)`.")
    if "l" not in desdf or "t" not in desdf:
        raise ValueError("DESDF must provide map crop offsets `l` and `t`.")
    top, left = int(desdf["t"]), int(desdf["l"])
    rows, cols, orientation_count = desdf_values.shape
    gt_x, gt_y, gt_yaw = pose_xy_yaw
    gt_row_map, gt_col_map = map_xy_to_row_col(map_array.shape, (gt_x, gt_y))
    gt_row_index = int(np.rint((gt_row_map - top) / 10.0))
    gt_col_index = int(np.rint((gt_col_map - left) / 10.0))
    if not (0 <= gt_row_index < rows and 0 <= gt_col_index < cols):
        raise ValueError("Ground-truth pose lies outside the DESDF crop.")
    orientation_step = 2.0 * np.pi / orientation_count
    gt_yaw_index = int(np.rint((gt_yaw % (2.0 * np.pi)) / orientation_step)) % orientation_count
    radius_cells = int(np.ceil(radius_m / 0.1))
    candidates: list[CandidateState] = []
    ground_truth: CandidateState | None = None
    for row_index in range(max(0, gt_row_index - radius_cells), min(rows, gt_row_index + radius_cells + 1)):
        for col_index in range(max(0, gt_col_index - radius_cells), min(cols, gt_col_index + radius_cells + 1)):
            map_row, map_col = top + row_index * 10, left + col_index * 10
            if not (0 <= map_row < map_array.shape[0] and 0 <= map_col < map_array.shape[1]):
                continue
            if map_array[map_row, map_col] < 128:
                continue
            x_m, y_m = row_col_to_map_xy(map_array.shape, (map_row, map_col))
            spatial_distance = float(np.hypot(x_m - gt_x, y_m - gt_y))
            is_gt_cell = row_index == gt_row_index and col_index == gt_col_index
            if not is_gt_cell and spatial_distance > radius_m + 1e-9:
                continue
            for yaw_index in range(orientation_count):
                candidate = CandidateState(
                    row_index=row_index,
                    col_index=col_index,
                    yaw_index=yaw_index,
                    x_m=x_m,
                    y_m=y_m,
                    yaw_rad=yaw_index * orientation_step,
                    quantization_error_m=spatial_distance if is_gt_cell else 0.0,
                )
                candidates.append(candidate)
                if is_gt_cell and yaw_index == gt_yaw_index:
                    ground_truth = candidate
    if ground_truth is None:
        raise ValueError("The quantized ground-truth DESDF state is not a valid free candidate.")
    return candidates, ground_truth


def summarize_candidate_scores(
    scores: Sequence[float], true_index: int, positions_m: np.ndarray
) -> dict[str, float | int | bool]:
    """Summarize ranking and location metrics for a candidate score vector."""
    values = np.asarray(scores, dtype=np.float64)
    positions = np.asarray(positions_m, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be a non-empty finite vector.")
    if positions.shape != (values.size, 2):
        raise ValueError("Positions must have shape `(candidate_count, 2)`.")
    if not 0 <= true_index < values.size:
        raise IndexError("True candidate index is outside the score vector.")
    order = np.argsort(-values, kind="stable")
    best_index = int(order[0])
    ground_truth_rank = int(np.where(order == true_index)[0][0]) + 1
    best_distance = float(np.linalg.norm(positions[best_index] - positions[true_index]))
    return {
        "ground_truth_rank": ground_truth_rank,
        "best_candidate_index": best_index,
        "best_candidate_distance_m": best_distance,
        "success_within_0_25m": best_distance <= 0.25,
        "success_within_0_5m": best_distance <= 0.5,
    }


def config_to_dict(config: PlanarRirConfig) -> Mapping[str, float | int]:
    """Return JSON-compatible fixed configuration provenance."""
    return {
        "sample_rate_hz": config.sample_rate_hz,
        "sound_speed_m_s": config.sound_speed_m_s,
        "ray_count": config.ray_count,
        "max_time_ms": config.max_time_ms,
        "wall_reflection_gain": config.wall_reflection_gain,
        "receiver_radius_m": config.receiver_radius_m,
        "direct_gain": config.direct_gain,
        "energy_window_ms": config.energy_window_ms,
        "peak_count": config.peak_count,
    }
