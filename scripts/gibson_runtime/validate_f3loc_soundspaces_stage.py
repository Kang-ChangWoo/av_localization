#!/usr/bin/env python3
"""Validate raw z-up Gibson OBJ stages in Habitat-Sim before SoundSpaces RIR work."""

from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from gibson_av_common import load_phase_one_scenes, write_json
from soundspaces_gibson_common import choose_floor_elevation, direct_floor_index, source_to_habitat, stage_readiness


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_TO_HABITAT = "(x, y, z) -> (x, z, -y)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene")
    selection.add_argument("--all-phase-one-scenes", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("/ssd1/track1/f3loc/data"))
    parser.add_argument("--mesh-root", type=Path, default=Path("/ssd1/track1/f3loc/raw/gibson_meshes/gibson_v2_selected"))
    parser.add_argument("--contract-summary", type=Path, default=WORKSPACE_ROOT / "outputs/f3loc_runs/gibson_v2_contract/gibson_v2_contract_summary.json")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "outputs/f3loc_runs/gibson_soundspaces_stage")
    parser.add_argument("--snap-threshold-m", type=float, default=0.25)
    parser.add_argument("--agent-height-m", type=float, default=1.7)
    parser.add_argument("--agent-radius-m", type=float, default=0.0, help="Use zero radius for point-source/listener geometry alignment.")
    parser.add_argument("--navmesh-cell-size-m", type=float, default=0.01, help="Fine Recast grid used to compare against 0.01 m Gibson traversability rasters.")
    parser.add_argument("--preview-size", type=int, default=256)
    return parser.parse_args()


def read_finite_poses(path: Path) -> List[Tuple[float, float, float]]:
    poses = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values[:2]):
            poses.append((values[0], values[1], values[2] if len(values) >= 3 else 0.0))
    return poses


def _backend_configuration(scene_id: str) -> Any:
    import habitat_sim

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = scene_id
    config.enable_physics = False
    config.gpu_device_id = 0
    return config


def _agent_configuration(preview_size: int, agent_height_m: float, agent_radius_m: float) -> Any:
    import habitat_sim

    color = habitat_sim.CameraSensorSpec()
    color.uuid = "stage_rgb"
    color.sensor_type = habitat_sim.SensorType.COLOR
    color.resolution = [preview_size, preview_size]
    color.position = [0.0, agent_height_m, 0.0]
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "stage_depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [preview_size, preview_size]
    depth.position = [0.0, agent_height_m, 0.0]
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = agent_height_m
    agent.radius = agent_radius_m
    agent.sensor_specifications = [color, depth]
    return agent


def _stage_simulator(mesh_path: Path, handle: str, preview_size: int, agent_height_m: float, agent_radius_m: float) -> Any:
    # This import order is required by the installed Habitat-Sim binding.
    import quaternion  # noqa: F401
    import habitat_sim

    initial_agent = _agent_configuration(preview_size, agent_height_m, agent_radius_m)
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(_backend_configuration("NONE"), [initial_agent]))
    manager = simulator.get_stage_template_manager()
    stage = manager.create_new_template(handle, register_template=False)
    stage.render_asset_handle = str(mesh_path)
    stage.collision_asset_handle = str(mesh_path)
    stage.is_collidable = True
    stage.orient_up = [0.0, 0.0, 1.0]
    stage.orient_front = [0.0, 1.0, 0.0]
    manager.register_template(stage)
    simulator.reconfigure(
        habitat_sim.Configuration(
            _backend_configuration(handle),
            [_agent_configuration(preview_size, agent_height_m, agent_radius_m)],
        )
    )
    return simulator


def _save_preview(observations: Mapping[str, Any], output_dir: Path) -> Dict[str, str]:
    rgb = np.asarray(observations["stage_rgb"])
    if rgb.ndim == 3 and rgb.shape[-1] >= 3:
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(output_dir / "stage_rgb_preview.png")
    else:
        raise ValueError("Habitat RGB preview did not have an RGB channel layout.")
    depth = np.asarray(observations["stage_depth"], dtype=np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if not finite.size:
        max_depth = None
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        max_depth = float(np.percentile(finite, 95))
        normalized = np.clip(depth / max_depth, 0.0, 1.0)
    Image.fromarray((normalized * 255).astype(np.uint8)).save(output_dir / "stage_depth_preview.png")
    return {"rgb_preview": "stage_rgb_preview.png", "depth_preview": "stage_depth_preview.png", "depth_p95_m": max_depth, "depth_preview_has_positive_values": bool(finite.size)}


def _floor_candidate_metrics(
    simulator: Any,
    poses: Sequence[Tuple[float, float, float]],
    floor_index: int,
    floor_z_m: float,
    threshold_m: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    planar_distances, three_dimensional_distances, vertical_offsets = [], [], []
    first_snapped = None
    for x, y, _ in poses:
        point = np.asarray(source_to_habitat((x, y, floor_z_m)), dtype=np.float32)
        snapped = simulator.pathfinder.snap_point(point)
        if first_snapped is None:
            first_snapped = snapped
        delta = snapped - point
        planar_distances.append(float(np.linalg.norm(delta[[0, 2]])))
        three_dimensional_distances.append(float(np.linalg.norm(delta)))
        vertical_offsets.append(float(delta[1]))
    return {
        "floor_index": floor_index,
        "floor_z_m": floor_z_m,
        "planar_within_threshold_count": sum(distance <= threshold_m for distance in planar_distances),
        "max_planar_snap_distance_m": max(planar_distances),
        "mean_planar_snap_distance_m": sum(planar_distances) / len(planar_distances),
        "max_3d_snap_distance_m": max(three_dimensional_distances),
        "mean_3d_snap_distance_m": sum(three_dimensional_distances) / len(three_dimensional_distances),
        "mean_vertical_snap_offset_m": sum(vertical_offsets) / len(vertical_offsets),
        "min_vertical_snap_offset_m": min(vertical_offsets),
        "max_vertical_snap_offset_m": max(vertical_offsets),
    }, first_snapped


def validate_scene(
    scene: str,
    contract_scene: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    source_root = args.mesh_root / scene
    mesh_matches = sorted(source_root.glob("*_mesh.obj"))
    if len(mesh_matches) != 1:
        return {"scene": scene, "status": "BLOCKED", "reason": "Expected exactly one raw OBJ mesh."}
    direct_traversability_floor_index = direct_floor_index(contract_scene)
    floors = [float(value) for value in (source_root / "floors.txt").read_text().splitlines() if value.strip()]
    poses = read_finite_poses(args.dataset_root / "gibson_t" / scene / "poses.txt")
    if not poses:
        return {"scene": scene, "status": "BLOCKED", "reason": "No finite F3Loc poses."}
    output_dir = args.output_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator = _stage_simulator(mesh_matches[0], f"gibson_zup_{scene}", args.preview_size, args.agent_height_m, args.agent_radius_m)
    try:
        import habitat_sim

        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = args.agent_height_m
        settings.agent_radius = args.agent_radius_m
        settings.cell_size = args.navmesh_cell_size_m
        navmesh_ok = bool(simulator.recompute_navmesh(simulator.pathfinder, settings))
        candidates_with_snapped = [
            _floor_candidate_metrics(simulator, poses, floor_index, floor_z, args.snap_threshold_m)
            for floor_index, floor_z in enumerate(floors)
        ]
        candidates = [candidate for candidate, _ in candidates_with_snapped]
        selected_floor = choose_floor_elevation(candidates)
        selected_index = int(selected_floor["floor_index"])
        first_snapped = candidates_with_snapped[selected_index][1]
        agent = simulator.get_agent(0)
        state = agent.get_state()
        state.position = first_snapped
        agent.set_state(state, True)
        previews = _save_preview(simulator.get_sensor_observations(), output_dir)
        status = stage_readiness(
            pose_count=len(poses),
            within_threshold_count=selected_floor["planar_within_threshold_count"],
            max_planar_distance_m=selected_floor["max_planar_snap_distance_m"],
            threshold_m=args.snap_threshold_m,
        )
        if status == "ACCEPTED_FOR_HABITAT_STAGE_NAVIGATION":
            reason = "All finite F3Loc poses satisfy the horizontal Habitat navmesh snap threshold."
        elif status == "PARTIALLY_READY":
            reason = "A bounded local navmesh exception remains; use Habitat snap_point before rendering that pose."
        else:
            reason = "One or more F3Loc poses exceed the horizontal Habitat navmesh snap threshold."
        result: Dict[str, Any] = {
            "scene": scene,
            "status": status,
            "reason": reason,
            "scope": "Raw z-up OBJ Habitat stage and navmesh only; no RIR was generated.",
            "mesh": str(mesh_matches[0]),
            "direct_traversability_floor_index": direct_traversability_floor_index,
            "floor_index": selected_index,
            "source_floor_z_m": selected_floor["floor_z_m"],
            "floor_selection_scope": "Selected by horizontal navmesh evidence; floor_trav file index is not assumed to equal floors.txt row index.",
            "floor_candidates": candidates,
            "source_to_habitat": SOURCE_TO_HABITAT,
            "stage_frame": {"orient_up": [0, 0, 1], "orient_front": [0, 1, 0]},
            "finite_pose_count": len(poses),
            "snap_threshold_m": args.snap_threshold_m,
            "navmesh_recomputed": navmesh_ok,
            "navmesh_loaded": bool(simulator.pathfinder.is_loaded),
            "navmesh_cell_size_m": args.navmesh_cell_size_m,
            "snap_within_threshold_count": selected_floor["planar_within_threshold_count"],
            "max_snap_distance_m": selected_floor["max_planar_snap_distance_m"],
            "mean_snap_distance_m": selected_floor["mean_planar_snap_distance_m"],
            "max_3d_snap_distance_m": selected_floor["max_3d_snap_distance_m"],
            "mean_3d_snap_distance_m": selected_floor["mean_3d_snap_distance_m"],
            "mean_vertical_snap_offset_m": selected_floor["mean_vertical_snap_offset_m"],
            "representative_camera_height_m": args.agent_height_m,
            "camera_height_scope": "Representative renderer mount only; exact released F3Loc camera height remains unresolved.",
            **previews,
        }
    finally:
        simulator.close()
    write_json(output_dir / f"{scene}_stage_metrics.json", result)
    return result


def _write_aggregate(output_root: Path, results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    statuses = {item.get("status") for item in results}
    if results and statuses == {"ACCEPTED_FOR_HABITAT_STAGE_NAVIGATION"}:
        status = "ACCEPTED_FOR_HABITAT_STAGE_NAVIGATION"
    elif "BLOCKED" not in statuses:
        status = "PARTIALLY_READY"
    else:
        status = "BLOCKED"
    aggregate = {"status": status, "scope": "No audio sensor or RIR was generated.", "scenes": list(results)}
    write_json(output_root / "gibson_soundspaces_stage_summary.json", aggregate)
    fields = ["scene", "status", "finite_pose_count", "floor_index", "source_floor_z_m", "snap_within_threshold_count", "max_snap_distance_m", "mean_snap_distance_m"]
    with (output_root / "gibson_soundspaces_stage_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: scene.get(field) for field in fields} for scene in results])
    lines = ["# F3Loc SoundSpaces Raw OBJ Stage Validation", "", f"- Status: **{status}**", "- Coordinate mapping: `(x, y, z) -> (x, z, -y)`", "- Stage frame: `orient_up=(0,0,1)`, `orient_front=(0,1,0)`", "- Criterion: horizontal navmesh snap at 0.01 m Recast cell size; F3Loc has no released camera-height value.", "- Scope: raw mesh stage/navmesh only. No audio sensor or RIR was generated.", "", "| Scene | Status | Poses | Floor | Horizontal hit | Max horizontal snap (m) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for scene in results:
        lines.append("| {scene} | {status} | {finite_pose_count} | {floor_index} | {snap_within_threshold_count} | {max_snap_distance_m:.4f} |".format(**scene))
    (output_root / "gibson_soundspaces_stage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tabs, panels = [], []
    for index, scene in enumerate(results):
        name = escape(str(scene["scene"]))
        selected = "true" if index == 0 else "false"
        hidden = "" if index == 0 else " hidden"
        tabs.append(f'<button class="tab" data-panel="panel-{index}" aria-selected="{selected}">{name}</button>')
        panels.append(f'<section id="panel-{index}"{hidden}><div class="metrics"><b>{escape(scene["status"])}</b><b>{scene["finite_pose_count"]:,} poses</b><b>{scene["snap_within_threshold_count"]:,} horizontally within 0.25 m</b><b>max {scene["max_snap_distance_m"]:.3f} m</b></div><p>{escape(scene["floor_selection_scope"])} Selected <code>floors.txt[{scene["floor_index"]}] = {scene["source_floor_z_m"]:.4f} m</code>; planar-raster index is <code>{scene["direct_traversability_floor_index"]}</code>.</p><div class="images"><figure><img src="{name}/{scene["rgb_preview"]}" alt="{name} raw OBJ RGB preview"><figcaption>Representative RGB preview. This is not a F3Loc RGB pose-equivalence check.</figcaption></figure><figure><img src="{name}/{scene["depth_preview"]}" alt="{name} raw OBJ depth preview"><figcaption>Normalized depth preview from the same representative camera.</figcaption></figure></div></section>')
    accepted_count = sum(item.get("status") == "ACCEPTED_FOR_HABITAT_STAGE_NAVIGATION" for item in results)
    partial_count = sum(item.get("status") == "PARTIALLY_READY" for item in results)
    total_poses = sum(int(item.get("finite_pose_count", 0)) for item in results)
    status_class = "partial" if status == "PARTIALLY_READY" else "blocked" if status == "BLOCKED" else "accepted"
    html = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F3Loc SoundSpaces Stage Validation</title><style>body{{margin:0;background:#f5f7fa;color:#172033;font-family:system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:26px}}h1{{margin:0}}p{{color:#596779;line-height:1.5}}.status{{display:inline-block;margin-top:12px;padding:7px 10px;font-weight:700;border-radius:6px}}.status.accepted{{border:1px solid #167349;color:#167349;background:#e7f4eb}}.status.partial{{border:1px solid #a76500;color:#7a4c00;background:#fff5dc}}.status.blocked{{border:1px solid #a73535;color:#8e2929;background:#fceaea}}.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}.summary div,.scope div,section{{background:#fff;border:1px solid #d9e0ea;border-radius:8px;padding:16px}}.summary b{{display:block;font-size:24px;margin-top:4px}}.summary span{{font-size:13px;color:#596779}}.scope{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}}.scope div:last-child{{border-left:4px solid #a73535}}.tabs{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:14px}}button{{border:1px solid #d9e0ea;background:#fff;padding:9px;text-align:left;border-radius:5px;cursor:pointer}}button[aria-selected=true]{{background:#edf3f7;border-color:#0f5f8c;color:#0f5f8c;font-weight:700}}.metrics{{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:12px}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0;border:1px solid #d9e0ea}}img{{display:block;width:100%;background:#000;max-height:520px;object-fit:contain}}figcaption{{padding:9px;font-size:13px;color:#596779}}@media(max-width:700px){{.summary,.scope,.images,.tabs{{grid-template-columns:1fr}}}}</style></head><body><main><h1>F3Loc SoundSpaces Raw OBJ Stage Validation</h1><div class="status {status_class}">{escape(status)}</div><p>Raw Gibson z-up OBJ meshes loaded in Habitat with explicit source-to-Habitat coordinate conversion.</p><div class="summary"><div><span>Accepted scenes</span><b>{accepted_count}/9</b></div><div><span>Partially ready scenes</span><b>{partial_count}</b></div><div><span>Finite F3Loc poses checked</span><b>{total_poses:,}</b></div></div><div class="scope"><div><b>Established</b><p>Stage orientation, floor elevation, raw mesh collision navmesh, and F3Loc planar pose placement.</p></div><div><b>Still excluded</b><p>Acoustic materials, source/listener policy, camera-yaw equivalence, SoundSpaces RIR, and EchoScan input evaluation.</p></div></div><section><div class="tabs">{"".join(tabs)}</div>{"".join(panels)}</section><script>document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{{document.querySelectorAll('.tab').forEach(x=>x.setAttribute('aria-selected','false'));document.querySelectorAll('section[id]').forEach(x=>x.hidden=true);t.setAttribute('aria-selected','true');document.getElementById(t.dataset.panel).hidden=false}})</script></main></body></html>'''
    (output_root / "gibson_soundspaces_stage_report.html").write_text(html, encoding="utf-8")
    return aggregate


def main() -> int:
    args = parse_args()
    contract = json.loads(args.contract_summary.read_text(encoding="utf-8"))
    contract_by_scene = {item["scene"]: item for item in contract["scenes"]}
    allowed = load_phase_one_scenes()
    scenes = allowed if args.all_phase_one_scenes else [args.scene]
    if any(scene not in allowed for scene in scenes):
        raise SystemExit("Requested scene is outside the phase-one allowlist.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [validate_scene(scene, contract_by_scene[scene], args) for scene in scenes]
    aggregate = _write_aggregate(args.output_root, results)
    print(f"{aggregate['status']}: {args.output_root}")
    return 0 if aggregate["status"] != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
