#!/usr/bin/env python3
"""Pure coordinate and decision helpers for Gibson raw meshes in Habitat."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple


def source_to_habitat(source_xyz: Sequence[float]) -> Tuple[float, float, float]:
    """Map Gibson z-up source coordinates to the declared Habitat stage frame."""
    if len(source_xyz) != 3:
        raise ValueError("Expected exactly three source coordinates.")
    x, y, z = (float(value) for value in source_xyz)
    return x, z, -y


def direct_floor_index(contract_scene: Mapping[str, Any]) -> int:
    """Read the selected original Gibson floor from a passed coordinate contract."""
    try:
        return int(contract_scene["best_by_mapping"]["identity"]["floor_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Contract scene has no accepted direct floor index.") from error


def choose_floor_elevation(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the floor candidate with the strongest horizontal navmesh evidence.

    `floor_trav_<n>.png` establishes a planar convention but does not prove that
    its number is the same position in `floors.txt`.  F3Loc poses do not encode a
    height, so horizontal navmesh coverage is the decision signal here.
    """
    if not candidates:
        raise ValueError("At least one floor candidate is required.")
    required = {"floor_index", "floor_z_m", "planar_within_threshold_count", "mean_planar_snap_distance_m"}
    if any(not required.issubset(candidate) for candidate in candidates):
        raise ValueError("Floor candidate lacks required planar navmesh metrics.")
    return min(
        candidates,
        key=lambda candidate: (
            -int(candidate["planar_within_threshold_count"]),
            float(candidate["mean_planar_snap_distance_m"]),
            int(candidate["floor_index"]),
        ),
    )


def stage_status(snap_distances_m: Sequence[float], *, threshold_m: float) -> str:
    """Accept only finite pose snap distances at or below the configured threshold."""
    if not snap_distances_m or threshold_m <= 0:
        return "BLOCKED"
    if all(0.0 <= float(distance) <= threshold_m for distance in snap_distances_m):
        return "ACCEPTED_FOR_HABITAT_STAGE_NAVIGATION"
    return "BLOCKED"


def stage_readiness(
    *,
    pose_count: int,
    within_threshold_count: int,
    max_planar_distance_m: float,
    threshold_m: float,
    partial_fraction: float = 0.99,
    partial_max_distance_m: float = 0.30,
) -> str:
    """Classify an all-pass stage, a bounded local exception, or a block."""
    if pose_count <= 0 or threshold_m <= 0:
        return "BLOCKED"
    if within_threshold_count == pose_count and max_planar_distance_m <= threshold_m:
        return "ACCEPTED_FOR_HABITAT_STAGE_NAVIGATION"
    if (
        within_threshold_count / pose_count >= partial_fraction
        and max_planar_distance_m <= partial_max_distance_m
    ):
        return "PARTIALLY_READY"
    return "BLOCKED"
