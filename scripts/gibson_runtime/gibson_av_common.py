#!/usr/bin/env python3
"""Shared, read-only helpers for the F3Loc Gibson mesh audit tools."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:
    import yaml
except ImportError as error:  # pragma: no cover - present in the project images
    raise RuntimeError("PyYAML is required. Run these utilities in the f3loc container.") from error


GRID_SPLITS = {"gibson_f", "gibson_g"}
TRAJECTORY_SPLITS = {"gibson_t"}
MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".dae", ".stl", ".fbx"}
METADATA_KEYWORDS = (
    "pose", "traj", "camera", "extrinsic", "transform", "height", "quaternion",
    "rotation", "roll", "pitch",
)
METADATA_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".npy", ".npz", ".pkl", ".csv"}
SCENE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<number>\d+)$")
GRID_RGB_RE = re.compile(r"^(?P<index>\d+)-(?P<view>\d+)\.png$", re.IGNORECASE)
TRAJECTORY_RGB_RE = re.compile(r"^(?P<index>\d+)\.png$", re.IGNORECASE)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "track1_core/configs/datasets/gibson_av.yaml"


def load_phase_one_scenes(config_path: Path = DEFAULT_CONFIG_PATH) -> List[str]:
    scenes = load_yaml(config_path).get("phase_one_scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"phase_one_scenes is missing or empty in {config_path}")
    return [str(scene) for scene in scenes]


def load_phase_one_asset_names(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, str]:
    mapping = load_yaml(config_path).get("phase_one_asset_names", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"phase_one_asset_names must be a mapping in {config_path}")
    return {str(scene): str(asset_name) for scene, asset_name in mapping.items()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def binary_floorplan_mask(path: Path, threshold: int = 127) -> np.ndarray:
    """Return the light occupancy mask from a grayscale floorplan image.

    F3Loc maps use dark wall strokes on a light background. The caller decides
    whether the returned light mask is interpreted as free space or as a
    rendering background; this helper deliberately does not invent a room
    closure policy.
    """
    from PIL import Image

    with Image.open(path) as image:
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    return grayscale > threshold


def project_planar_poses(
    poses: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    resolution_m_per_pixel: float,
    flip_x: bool,
    flip_y: bool,
) -> List[Dict[str, Any]]:
    """Project planar poses into a centered source-floor raster without fitting.

    Gibson source-floor rasters use a metric image-centre convention. The
    optional sign flips exist solely to make the coordinate convention testable;
    callers must not treat them as an inferred world transform.
    """
    if width <= 0 or height <= 0 or resolution_m_per_pixel <= 0:
        raise ValueError("Raster dimensions and resolution must be positive.")
    projected = []
    for index, pose in enumerate(poses):
        if len(pose) < 2:
            raise ValueError(f"Pose {index} has fewer than two planar coordinates.")
        x_meter = float(pose[0])
        y_meter = float(pose[1])
        if not math.isfinite(x_meter) or not math.isfinite(y_meter):
            raise ValueError(f"Pose {index} has non-finite planar coordinates.")
        if flip_x:
            x_meter = -x_meter
        if flip_y:
            y_meter = -y_meter
        projected.append({
            "pose_index": index,
            "x_pixel": round(x_meter / resolution_m_per_pixel + width / 2),
            "y_pixel": round(y_meter / resolution_m_per_pixel + height / 2),
        })
    return projected


def score_traversability_projection(
    raster_path: Path,
    projected_pixels: Sequence[Mapping[str, Any]],
    *,
    threshold: int = 127,
) -> Dict[str, Any]:
    """Classify projected pixels against one binary Gibson traversal raster."""
    from PIL import Image

    with Image.open(raster_path) as image:
        raster = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = raster.shape
    traversable_count = 0
    blocked_count = 0
    out_of_bounds_count = 0
    classifications = []
    for item in projected_pixels:
        x_pixel = int(item["x_pixel"])
        y_pixel = int(item["y_pixel"])
        if not (0 <= x_pixel < width and 0 <= y_pixel < height):
            label = "out_of_bounds"
            out_of_bounds_count += 1
        elif raster[y_pixel, x_pixel] > threshold:
            label = "traversable"
            traversable_count += 1
        else:
            label = "blocked"
            blocked_count += 1
        classifications.append({**dict(item), "classification": label})
    total_count = len(projected_pixels)
    return {
        "raster_width": width,
        "raster_height": height,
        "threshold": threshold,
        "pose_count": total_count,
        "traversable_count": traversable_count,
        "blocked_count": blocked_count,
        "out_of_bounds_count": out_of_bounds_count,
        "hit_fraction": traversable_count / total_count if total_count else 0.0,
        "classifications": classifications,
    }


def similarity_candidates(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> List[Dict[str, Any]]:
    """Enumerate dihedral image orientations with dimension-normalizing scale."""
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    if source_height <= 0 or source_width <= 0 or target_height <= 0 or target_width <= 0:
        raise ValueError("Similarity candidate shapes must be positive.")
    candidates: List[Dict[str, Any]] = []
    for rotation_deg in (0, 90, 180, 270):
        rotated_height, rotated_width = (
            (source_width, source_height) if rotation_deg in {90, 270} else (source_height, source_width)
        )
        for reflected in (False, True):
            candidates.append({
                "rotation_deg": rotation_deg,
                "reflected": reflected,
                "scale_x": target_width / rotated_width,
                "scale_y": target_height / rotated_height,
                "translation_px": [0.0, 0.0],
            })
    return candidates


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Compute binary intersection-over-union for same-shape masks."""
    if left.shape != right.shape:
        raise ValueError(f"Mask shapes differ: {left.shape} != {right.shape}")
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def symmetric_chamfer_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return symmetric pixel Chamfer distance between two binary masks."""
    if left.shape != right.shape:
        raise ValueError(f"Mask shapes differ: {left.shape} != {right.shape}")
    if not left.any() or not right.any():
        return math.inf
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - available in f3loc image
        raise RuntimeError("OpenCV is required for mask distance transforms.") from error
    to_right = cv2.distanceTransform((~right).astype(np.uint8), cv2.DIST_L2, 3)
    to_left = cv2.distanceTransform((~left).astype(np.uint8), cv2.DIST_L2, 3)
    return float((to_right[left].mean() + to_left[right].mean()) / 2.0)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def read_split_scenes(split_file: Path) -> Dict[str, List[str]]:
    data = load_yaml(split_file)
    return {
        split_name: [str(scene) for scene in scenes or []]
        for split_name, scenes in data.items()
        if isinstance(scenes, list)
    }


def build_scene_candidates(scene: str) -> Dict[str, Any]:
    match = SCENE_SUFFIX_RE.match(scene)
    stripped = match.group("base") if match else None
    return {
        "exact": scene,
        "suffix_stripped_candidate": stripped,
        "suffix_stripped_is_ambiguous": stripped is not None,
    }


def parse_poses(path: Path) -> Dict[str, Any]:
    rows: List[List[float]] = []
    nonempty_row_count = 0
    invalid_rows: List[int] = []
    nonfinite_rows: List[int] = []
    dimensions: Counter[int] = Counter()
    if not path.is_file():
        return {
            "pose_file_present": False,
            "pose_count": 0,
            "pose_valid_count": 0,
            "pose_dimensions": {},
            "pose_invalid_rows": [],
            "pose_nonfinite_rows": [],
            "pose_ranges": [],
            "pose_duplicate_count": 0,
            "pose_largest_planar_steps": [],
        }

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        values = line.split()
        if not values:
            continue
        nonempty_row_count += 1
        try:
            row = [float(value) for value in values]
        except ValueError:
            invalid_rows.append(line_number)
            continue
        dimensions[len(row)] += 1
        if not all(math.isfinite(value) for value in row):
            nonfinite_rows.append(line_number)
        rows.append(row)

    finite_rows = [row for row in rows if all(math.isfinite(value) for value in row)]
    max_dimension = max((len(row) for row in finite_rows), default=0)
    ranges = []
    for dimension in range(max_dimension):
        values = [row[dimension] for row in finite_rows if len(row) > dimension]
        if values:
            ranges.append({"dimension": dimension, "min": min(values), "max": max(values)})
    planar_steps = []
    for index, (before, after) in enumerate(zip(finite_rows, finite_rows[1:])):
        if len(before) >= 2 and len(after) >= 2:
            planar_steps.append((math.hypot(after[0] - before[0], after[1] - before[1]), index, index + 1))
    largest_steps = [
        {"from_row": before, "to_row": after, "distance_m": distance}
        for distance, before, after in sorted(planar_steps, reverse=True)[:5]
    ]
    duplicate_count = len(finite_rows) - len({tuple(row) for row in finite_rows})
    return {
        "pose_file_present": True,
        "pose_count": nonempty_row_count,
        "pose_valid_count": len(rows),
        "pose_dimensions": dict(sorted(dimensions.items())),
        "pose_invalid_rows": invalid_rows,
        "pose_nonfinite_rows": nonfinite_rows,
        "pose_ranges": ranges,
        "pose_duplicate_count": duplicate_count,
        "pose_largest_planar_steps": largest_steps,
    }


def _audit_rgb(rgb_dir: Path, split_name: str, pose_count: int) -> Dict[str, Any]:
    expected_views = 4 if split_name in GRID_SPLITS else 1
    pose_rows_per_group = 4 if split_name in GRID_SPLITS else 1
    names = sorted(path.name for path in rgb_dir.glob("*.png")) if rgb_dir.is_dir() else []
    groups: Dict[int, set[int]] = {}
    unrecognized: List[str] = []
    matcher = GRID_RGB_RE if split_name in GRID_SPLITS else TRAJECTORY_RGB_RE
    for name in names:
        match = matcher.match(name)
        if not match:
            unrecognized.append(name)
            continue
        index = int(match.group("index"))
        view = int(match.group("view")) if split_name in GRID_SPLITS else 0
        groups.setdefault(index, set()).add(view)
    expected_group_count = pose_count // pose_rows_per_group
    expected_indexes = set(range(expected_group_count))
    missing_groups = sorted(
        index for index in expected_indexes
        if groups.get(index, set()) != set(range(expected_views))
    )
    extra_groups = sorted(set(groups) - expected_indexes)
    logical_count = sum(1 for views in groups.values() if views == set(range(expected_views)))
    return {
        "rgb_dir_present": rgb_dir.is_dir(),
        "rgb_physical_count": len(names),
        "rgb_logical_count": logical_count,
        "rgb_views_per_logical_group": expected_views,
        "pose_rows_per_logical_group": pose_rows_per_group,
        "rgb_missing_groups": missing_groups,
        "rgb_extra_groups": extra_groups,
        "rgb_unrecognized_files": unrecognized,
        "rgb_count_mismatch": bool(missing_groups or extra_groups or unrecognized or pose_count % pose_rows_per_group),
    }


def _map_info(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"map_present": path.is_file(), "map_width": None, "map_height": None, "map_error": None}
    if not path.is_file():
        return result
    try:
        from PIL import Image
        with Image.open(path) as image:
            result["map_width"], result["map_height"] = image.size
    except Exception as error:  # A corrupted map must be reported, not abort the audit.
        result["map_error"] = str(error)
    return result


def _metadata_files(scene_dir: Path) -> List[str]:
    matches = []
    for path in scene_dir.rglob("*"):
        if not path.is_file() or path.name == "poses.txt" or path.suffix.lower() not in METADATA_SUFFIXES:
            continue
        lower_name = path.name.lower()
        if any(keyword in lower_name for keyword in METADATA_KEYWORDS):
            matches.append(str(path.relative_to(scene_dir)))
    return sorted(matches)


def _line_count(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def audit_scene(dataset_root: Path, split_name: str, scene: str, desdf_root: Optional[Path] = None) -> Dict[str, Any]:
    scene_dir = dataset_root / split_name / scene
    poses = parse_poses(scene_dir / "poses.txt")
    audit: Dict[str, Any] = {
        "split": split_name,
        "scene": scene,
        "scene_dir": str(scene_dir),
        "scene_dir_present": scene_dir.is_dir(),
        **poses,
        **_audit_rgb(scene_dir / "rgb", split_name, poses["pose_count"]),
        **_map_info(scene_dir / "map.png"),
        "depth40_present": (scene_dir / "depth40.txt").is_file(),
        "depth40_count": _line_count(scene_dir / "depth40.txt"),
        "depth160_present": (scene_dir / "depth160.txt").is_file(),
        "depth160_count": _line_count(scene_dir / "depth160.txt"),
        "metadata_files": _metadata_files(scene_dir) if scene_dir.is_dir() else [],
    }
    if desdf_root is None:
        desdf_root = dataset_root / "desdf"
    audit["desdf_present"] = (desdf_root / scene).exists()
    audit["desdf_path"] = str(desdf_root / scene)
    audit["depth40_count_mismatch"] = None if audit["depth40_count"] is None else audit["depth40_count"] != audit["pose_count"]
    audit["depth160_count_mismatch"] = None if audit["depth160_count"] is None else audit["depth160_count"] != audit["pose_count"]
    return audit


def build_mesh_index(mesh_root: Path) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = {}
    if not mesh_root.is_dir():
        return index
    for path in mesh_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MESH_SUFFIXES:
            continue
        identifiers = {path.stem, path.parent.name}
        for identifier in identifiers:
            index.setdefault(identifier, []).append({"path": str(path), "format": path.suffix.lower().lstrip(".")})
    return index


def mesh_matches(scene: str, mesh_index: Mapping[str, List[Dict[str, str]]], expected_asset_name: Optional[str] = None) -> Dict[str, Any]:
    candidates = build_scene_candidates(scene)
    exact = list(mesh_index.get(scene, []))
    if expected_asset_name:
        exact.extend(mesh_index.get(expected_asset_name, []))
    stripped_name = candidates["suffix_stripped_candidate"]
    stripped = list(mesh_index.get(stripped_name, [])) if stripped_name else []
    if exact:
        status = "exact"
        matches = exact
    elif stripped:
        status = "candidate"
        matches = stripped
    else:
        status = "missing"
        matches = []
    return {**candidates, "expected_asset_name": expected_asset_name or scene, "asset_match_status": status, "matched_meshes": matches}


def build_manifest_records(audits: Sequence[Mapping[str, Any]], mesh_index: Mapping[str, List[Dict[str, str]]], expected_asset_names: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    by_scene: Dict[str, List[Mapping[str, Any]]] = {}
    for audit in audits:
        scene = str(audit.get("scene", ""))
        if scene:
            by_scene.setdefault(scene, []).append(audit)
    records = []
    for scene, scene_audits in sorted(by_scene.items()):
        observations: Dict[str, int] = {}
        for audit in scene_audits:
            membership = f"{audit.get('split', '')}:{audit.get('split_group', '')}"
            observations[membership] = int(audit.get("rgb_logical_count", audit.get("pose_count", 0)) or 0)
        match = mesh_matches(scene, mesh_index, (expected_asset_names or {}).get(scene))
        records.append({
            "scene": scene,
            "included_splits": sorted(observations),
            "observations_by_split": observations,
            "pose_rows_by_split": {
                f"{audit.get('split', '')}:{audit.get('split_group', '')}": int(audit.get("pose_count", 0) or 0)
                for audit in scene_audits
            },
            **match,
            "notes": "Suffix-stripped names are candidates only; never auto-accepted.",
        })
    return records


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inspect_archive(path: Path, compute_sha256: bool = False) -> Dict[str, Any]:
    digest = sha256_file(path) if compute_sha256 and path.is_file() else None
    result: Dict[str, Any] = {"archive": str(path), "sha256": digest, "status": "unsupported_or_corrupt", "members": []}
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                result.update(status="listed_unverified", members=archive.namelist())
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                result.update(status="listed_unverified", members=archive.getnames())
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        pass
    return result


def selected_archive_members(members: Iterable[str], scenes: Iterable[str], asset_name_aliases: Optional[Mapping[str, str]] = None) -> List[str]:
    scene_set = set(scenes)
    asset_name_set = set((asset_name_aliases or {}).values())
    selected = []
    for member in members:
        parts = Path(member).parts
        stem = Path(member).stem
        if any(part in scene_set or part in asset_name_set for part in parts) or stem in scene_set or stem in asset_name_set:
            selected.append(member)
    return selected
