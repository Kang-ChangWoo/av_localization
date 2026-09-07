"""Machine-local roots for non-versioned AV-FPLoc experiment assets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenePaths:
    root: Path
    scene: str

    @property
    def map_png(self) -> Path:
        return self.root / "datasets" / "f3loc" / "gibson_t" / self.scene / "map.png"

    @property
    def poses_txt(self) -> Path:
        return self.map_png.with_name("poses.txt")

    @property
    def desdf(self) -> Path:
        return self.root / "datasets" / "f3loc" / "desdf" / self.scene / "desdf.npy"

    @property
    def mesh_dir(self) -> Path:
        return self.root / "meshes" / "gibson_v2" / self.scene

    @property
    def alignment_dir(self) -> Path:
        return self.root / "alignment" / self.scene


@dataclass(frozen=True)
class DataRoots:
    root: Path

    @classmethod
    def from_environment(cls, value: str | None = None) -> "DataRoots":
        return cls(Path(value or os.environ.get("AVFPLOC_DATA_ROOT", "/file2/jeongeon/AV-FPLoc")))

    @property
    def f3loc_data(self) -> Path:
        return self.root / "datasets" / "f3loc"

    @property
    def meshes(self) -> Path:
        return self.root / "meshes" / "gibson_v2"

    @property
    def alignment(self) -> Path:
        return self.root / "alignment"

    @property
    def renders(self) -> Path:
        return self.root / "rendered_rirs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def scene(self, scene: str) -> ScenePaths:
        return ScenePaths(self.root, scene)
