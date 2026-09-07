#!/usr/bin/env python3
"""Read-only collaborator preflight for the AV-FPLoc Gibson RIR workflow."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track1_core.configs.data_paths import DataRoots

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Springhill"); parser.add_argument("--data-root")
    args = parser.parse_args(); scene = DataRoots.from_environment(args.data_root).scene(args.scene)
    required = {"map.png": scene.map_png, "poses.txt": scene.poses_txt, "desdf.npy": scene.desdf,
                "mesh OBJ": scene.mesh_dir / f"{args.scene}_mesh.obj", "stage alignment": scene.alignment_dir / f"{args.scene}_stage_metrics.json"}
    missing = [name for name, path in required.items() if not path.is_file()]
    probe = subprocess.run([sys.executable, "-c", "import habitat_sim; assert hasattr(habitat_sim, 'AudioSensorSpec')"], capture_output=True, text=True)
    habitat = probe.returncode == 0; audio_sensor = habitat
    report = {"scene": args.scene, "data_root": str(scene.root), "assets": {name: str(path) for name, path in required.items()}, "missing": missing, "habitat_sim": habitat, "audio_sensor_spec": audio_sensor, "ready_for_rir": not missing and audio_sensor}
    if probe.returncode:
        report["habitat_probe_error"] = probe.stderr.strip()[-500:]
    print(json.dumps(report, indent=2)); return 0 if report["ready_for_rir"] else 1
if __name__ == "__main__": raise SystemExit(main())
