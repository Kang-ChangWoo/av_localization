#!/usr/bin/env python3
"""Render controlled six-channel SoundSpaces RIRs at released F3Loc poses."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from echoscan_rir_geometry import echoscan_ring_offsets_habitat, seeded_free_component, write_closed_proxy_obj
from soundspaces_gibson_common import source_to_habitat
from validate_f3loc_soundspaces_stage import _stage_simulator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path("/ssd1/track1/f3loc/data")
DEFAULT_MESH_ROOT = Path("/ssd1/track1/f3loc/raw/gibson_meshes/gibson_v2_selected")
DEFAULT_STAGE_ROOT = WORKSPACE_ROOT / "outputs/f3loc_runs/gibson_soundspaces_stage"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs/f3loc_runs/echoscan_rir_controlled"
ECHOSCAN_CHANNELS = 6
ECHOSCAN_SAMPLES = 1024
DIRECT_SOUND_GUARD_SAMPLES = 16


@dataclass(frozen=True)
class AcousticRenderConfig:
    indirect_rays: int
    indirect_ray_depth: int
    source_rays: int
    source_ray_depth: int
    diffraction: bool
    transmission: bool


def _configure_acoustics(config: Any, settings: AcousticRenderConfig, *, sample_rate: int) -> None:
    """Apply explicit SoundSpaces ray-tracing settings to one audio sensor."""
    config.sampleRate = sample_rate
    config.direct = True
    config.indirect = True
    config.indirectRayCount = settings.indirect_rays
    config.indirectRayDepth = settings.indirect_ray_depth
    config.sourceRayCount = settings.source_rays
    config.sourceRayDepth = settings.source_ray_depth
    config.diffraction = settings.diffraction
    config.transmission = settings.transmission
    config.maxIRLength = 1.0


def stack_mono_observations(observations: Sequence[np.ndarray]) -> np.ndarray:
    """Stack six finite mono SoundSpaces observations as channel-first RIR data."""
    if len(observations) != 6:
        raise ValueError("Expected exactly six mono observations.")
    channels = []
    for observation in observations:
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim == 2 and 1 in values.shape:
            values = values.reshape(-1)
        if values.ndim != 1:
            raise ValueError("Audio observation must be mono with one sample dimension.")
        if not np.isfinite(values).all():
            raise ValueError("Audio observation must contain only finite values.")
        channels.append(values)
    lengths = {channel.shape[0] for channel in channels}
    if len(lengths) != 1:
        raise ValueError("All audio observations must have the same sample count.")
    return np.stack(channels, axis=0)


def prepare_rir_for_echoscan(rir: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Match EchoScan's direct-arrival alignment and fixed 6 x 1024 input shape."""
    values = np.asarray(rir, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != ECHOSCAN_CHANNELS:
        raise ValueError("Expected a finite channel-first RIR with exactly six channels.")
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("RIR must contain at least one finite sample per channel.")

    normalized = values - values.mean(axis=1, keepdims=True)
    normalized /= np.max(np.abs(normalized), axis=1, keepdims=True) + 1e-8
    direct_peak_index = int(np.argmax(np.abs(normalized[0])))
    start_index = direct_peak_index + DIRECT_SOUND_GUARD_SAMPLES
    prepared = normalized[:, start_index : start_index + ECHOSCAN_SAMPLES]
    if prepared.shape[1] < ECHOSCAN_SAMPLES:
        prepared = np.pad(prepared, ((0, 0), (0, ECHOSCAN_SAMPLES - prepared.shape[1])))
    return prepared.astype(np.float32, copy=False), {
        "direct_peak_index": direct_peak_index,
        "direct_sound_guard_samples": DIRECT_SOUND_GUARD_SAMPLES,
        "echoscan_start_index": start_index,
    }


def _read_pose(path: Path, pose_index: int) -> tuple[float, float, float]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= pose_index < len(lines):
        raise IndexError(f"Pose index {pose_index} is outside 0..{len(lines) - 1}.")
    values = lines[pose_index]
    if len(values) < 3:
        raise ValueError(f"Pose {pose_index} does not contain x, y, yaw.")
    pose = tuple(float(value) for value in values[:3])
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"Pose {pose_index} is non-finite.")
    return pose


def _selected_floor(stage_root: Path, scene: str) -> tuple[int, float]:
    metrics_path = stage_root / scene / f"{scene}_stage_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return int(metrics["floor_index"]), float(metrics["source_floor_z_m"])


def _render_one_channel(
    mesh_path: Path,
    *,
    pose: tuple[float, float, float],
    floor_z_m: float,
    offset_habitat: np.ndarray,
    sample_rate: int,
    acoustic_settings: AcousticRenderConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one mono channel in a fresh simulator for this legacy binding."""
    simulator = _stage_simulator(mesh_path, f"echoscan_rir_{mesh_path.stem}", 64, 0.0, 0.0)
    try:
        import habitat_sim

        navmesh = habitat_sim.NavMeshSettings()
        navmesh.set_defaults()
        navmesh.agent_height = 0.0
        navmesh.agent_radius = 0.0
        navmesh.cell_size = 0.01
        if not simulator.recompute_navmesh(simulator.pathfinder, navmesh):
            raise RuntimeError("Habitat could not build a navmesh for the acoustic stage.")
        floor_point = np.asarray(source_to_habitat((pose[0], pose[1], floor_z_m)), dtype=np.float32)
        snapped_floor = np.asarray(simulator.pathfinder.snap_point(floor_point), dtype=np.float32)
        if not np.isfinite(snapped_floor).all():
            raise RuntimeError("Habitat could not snap the F3Loc planar pose to this stage.")
        center = snapped_floor + np.asarray([0.0, 1.25, 0.0], dtype=np.float32)

        spec = habitat_sim.AudioSensorSpec()
        # The installed legacy binding reads only this UUID in _get_audio_observation.
        spec.uuid = "audio_sensor"
        spec.enableMaterials = False
        spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
        spec.channelLayout.channelCount = 1
        spec.position = offset_habitat.tolist()
        _configure_acoustics(spec.acousticsConfig, acoustic_settings, sample_rate=sample_rate)
        simulator.add_sensor(spec)
        agent = simulator.get_agent(0)
        state = agent.get_state()
        state.position = center
        state.sensor_states = {}
        agent.set_state(state, True)
        agent._sensors[spec.uuid].setAudioSourceTransform(center)
        observation = np.asarray(simulator.get_sensor_observations()[spec.uuid])
        return observation, {
            "floor_point_habitat": floor_point.tolist(),
            "snapped_floor_habitat": snapped_floor.tolist(),
            "array_source_center_habitat": center.tolist(),
            "horizontal_snap_displacement_m": float(np.linalg.norm((snapped_floor - floor_point)[[0, 2]])),
        }
    finally:
        simulator.close()


def _render_stage_rir(
    mesh_path: Path,
    *,
    pose: tuple[float, float, float],
    floor_z_m: float,
    sample_rate: int,
    acoustic_settings: AcousticRenderConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    offsets = echoscan_ring_offsets_habitat(pose[2])
    observations, channel_metadata = [], []
    for offset in offsets:
        observation, metadata = _render_one_channel(
            mesh_path,
            pose=pose,
            floor_z_m=floor_z_m,
            offset_habitat=offset,
            sample_rate=sample_rate,
            acoustic_settings=acoustic_settings,
        )
        observations.append(observation)
        channel_metadata.append(metadata)
    rir = stack_mono_observations(observations)
    if not np.any(np.abs(rir) > 0):
        raise RuntimeError("SoundSpaces produced a zero-energy RIR.")
    return rir, {
        **channel_metadata[0],
        "vertical_source_height_m": 1.25,
        "materials_enabled": False,
        "sample_rate_hz": sample_rate,
        "soundspaces_acoustics": asdict(acoustic_settings),
        "channel_offsets_habitat_m": offsets.tolist(),
        "channel_snap_displacements_m": [item["horizontal_snap_displacement_m"] for item in channel_metadata],
        "audio_binding_scope": "Legacy Habitat binding renders one audio_sensor UUID per fresh simulator; channels are rendered sequentially.",
    }


def _write_waveform(rir: np.ndarray, output_path: Path, sample_rate: int, title: str) -> None:
    times = np.arange(rir.shape[1]) / sample_rate
    figure, axis = plt.subplots(figsize=(11, 4))
    for channel, values in enumerate(rir):
        axis.plot(times, values, linewidth=0.7, label=f"ch{channel}")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.set_title(title)
    axis.legend(ncol=6, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_condition(
    output_dir: Path,
    rir: np.ndarray,
    metadata: Mapping[str, Any],
    sample_rate: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rir_path = output_dir / "rir_raw.npy"
    np.save(rir_path, rir)
    echoscan_rir, preprocessing = prepare_rir_for_echoscan(rir)
    echoscan_rir_path = output_dir / "rir_echoscan.npy"
    np.save(echoscan_rir_path, echoscan_rir)
    waveform_path = output_dir / "rir_waveform.png"
    _write_waveform(rir, waveform_path, sample_rate, "SoundSpaces full RIR (direct peak included)")
    echoscan_waveform_path = output_dir / "rir_echoscan_waveform.png"
    _write_waveform(
        echoscan_rir,
        echoscan_waveform_path,
        sample_rate,
        "EchoScan input RIR (direct peak plus 2 ms removed)",
    )
    metadata_path = output_dir / "rir_metadata.json"
    raw_peak_indices = np.argmax(np.abs(rir), axis=1)
    crop_start = preprocessing["echoscan_start_index"]
    raw_peak = float(np.abs(rir).max())
    post_guard_peak = float(np.abs(rir[:, crop_start:]).max())
    metadata_payload = {
        **dict(metadata),
        "soundspaces_full_shape": list(rir.shape),
        "echoscan_input_shape": list(echoscan_rir.shape),
        "echoscan_preprocessing": {
            **preprocessing,
            "all_channel_peak_indices": raw_peak_indices.tolist(),
            "all_channel_peaks_before_crop": bool(np.all(raw_peak_indices < crop_start)),
            "post_guard_peak_to_raw_peak": post_guard_peak / raw_peak if raw_peak else 0.0,
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "rir": str(rir_path),
        "rir_full": str(rir_path),
        "rir_echoscan": str(echoscan_rir_path),
        "waveform": str(waveform_path),
        "echoscan_waveform": str(echoscan_waveform_path),
        "metadata": str(metadata_path),
    }


def render_pose(
    *,
    scene: str,
    pose_index: int,
    dataset_root: Path,
    mesh_root: Path,
    stage_root: Path,
    output_root: Path,
    sample_rate: int,
    acoustic_settings: AcousticRenderConfig,
) -> list[dict[str, Any]]:
    pose = _read_pose(dataset_root / "gibson_t" / scene / "poses.txt", pose_index)
    floor_index, raw_floor_z = _selected_floor(stage_root, scene)
    pose_root = output_root / scene / f"pose_{pose_index:05d}"
    raw_meshes = sorted((mesh_root / scene).glob("*_mesh.obj"))
    if len(raw_meshes) != 1:
        raise FileNotFoundError(f"Expected one raw Gibson OBJ for {scene}.")
    records = []
    raw_rir, raw_metadata = _render_stage_rir(
        raw_meshes[0], pose=pose, floor_z_m=raw_floor_z, sample_rate=sample_rate, acoustic_settings=acoustic_settings
    )
    raw_metadata.update(
        {
            "condition": "raw_scan_open",
            "scene": scene,
            "pose_index": pose_index,
            "f3loc_pose_m_rad": list(pose),
            "mesh": str(raw_meshes[0]),
            "selected_floor_index": floor_index,
            "selected_floor_z_m": raw_floor_z,
            "scope": "Original scan geometry with fused objects and known open boundaries; audio materials are disabled.",
        }
    )
    records.append({"condition": "raw_scan_open", **_write_condition(pose_root / "raw_scan_open", raw_rir, raw_metadata, sample_rate)})

    map_array = np.asarray(Image.open(dataset_root / "gibson_t" / scene / "map.png").convert("L"))
    component = seeded_free_component(map_array, pose_xy_m=(pose[0], pose[1]))
    proxy_dir = pose_root / "floorplan_closed"
    proxy_path = proxy_dir / "floorplan_closed_proxy.obj"
    proxy_info = write_closed_proxy_obj(component, proxy_path)
    proxy_rir, proxy_metadata = _render_stage_rir(
        proxy_path, pose=pose, floor_z_m=0.0, sample_rate=sample_rate, acoustic_settings=acoustic_settings
    )
    proxy_metadata.update(
        {
            "condition": "floorplan_closed",
            "scene": scene,
            "pose_index": pose_index,
            "f3loc_pose_m_rad": list(pose),
            "mesh": str(proxy_path),
            "proxy": proxy_info,
            "scope": "Generated structural control from the F3Loc map component; it is not the original scene with furniture removed.",
        }
    )
    records.append({"condition": "floorplan_closed", **_write_condition(proxy_dir, proxy_rir, proxy_metadata, sample_rate)})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Springhill")
    parser.add_argument("--pose-index", type=int, action="append", dest="pose_indices")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--indirect-rays", type=int, default=512)
    parser.add_argument("--indirect-ray-depth", type=int, default=16)
    parser.add_argument("--source-rays", type=int)
    parser.add_argument("--source-ray-depth", type=int, default=8)
    parser.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transmission", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.indirect_rays <= 0 or args.indirect_ray_depth <= 0 or args.source_ray_depth <= 0:
        parser.error("Ray counts and depths must be positive.")
    acoustic_settings = AcousticRenderConfig(
        indirect_rays=args.indirect_rays,
        indirect_ray_depth=args.indirect_ray_depth,
        source_rays=args.source_rays or max(64, args.indirect_rays // 8),
        source_ray_depth=args.source_ray_depth,
        diffraction=args.diffraction,
        transmission=args.transmission,
    )
    pose_indices = args.pose_indices or [0, 141, 283]
    manifest = []
    for pose_index in pose_indices:
        manifest.extend(
            render_pose(
                scene=args.scene,
                pose_index=pose_index,
                dataset_root=args.dataset_root,
                mesh_root=args.mesh_root,
                stage_root=args.stage_root,
                output_root=args.output_root,
                sample_rate=args.sample_rate,
                acoustic_settings=acoustic_settings,
            )
        )
    manifest_path = args.output_root / args.scene / "render_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
