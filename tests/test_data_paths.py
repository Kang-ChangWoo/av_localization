from __future__ import annotations

from pathlib import Path

from track1_core.configs.data_paths import DataRoots


def test_data_roots_uses_environment_variable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AVFPLOC_DATA_ROOT", str(tmp_path))

    roots = DataRoots.from_environment()

    assert roots.root == tmp_path
    assert roots.f3loc_data == tmp_path / "datasets" / "f3loc"
    assert roots.meshes == tmp_path / "meshes" / "gibson_v2"
    assert roots.alignment == tmp_path / "alignment"


def test_scene_paths_reports_the_expected_shared_layout(tmp_path: Path) -> None:
    paths = DataRoots.from_environment(str(tmp_path)).scene("Springhill")

    assert paths.map_png == tmp_path / "datasets" / "f3loc" / "gibson_t" / "Springhill" / "map.png"
    assert paths.mesh_dir == tmp_path / "meshes" / "gibson_v2" / "Springhill"
