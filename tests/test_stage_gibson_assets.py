from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage_gibson_assets.py"
SPEC = importlib.util.spec_from_file_location("stage_gibson_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
staging = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = staging
SPEC.loader.exec_module(staging)


def test_stage_plan_contains_only_scene_assets(tmp_path: Path) -> None:
    data = tmp_path / "source-data"
    mesh = tmp_path / "source-mesh" / "Springhill"
    alignment = tmp_path / "source-alignment" / "Springhill"
    (data / "gibson_t" / "Springhill").mkdir(parents=True)
    (data / "desdf" / "Springhill").mkdir(parents=True)
    mesh.mkdir(parents=True)
    alignment.mkdir(parents=True)
    for path in [
        data / "gibson_t" / "Springhill" / "map.png",
        data / "gibson_t" / "Springhill" / "poses.txt",
        data / "desdf" / "Springhill" / "desdf.npy",
        mesh / "Springhill_mesh.obj",
        mesh / "Springhill_mesh.mtl",
        alignment / "Springhill_stage_metrics.json",
    ]:
        path.write_bytes(b"asset")

    plan = staging.build_plan(data, mesh.parent, alignment.parent, tmp_path / "shared", "Springhill")

    assert len(plan) == 6
    assert all("Springhill" in str(item.destination) for item in plan)
