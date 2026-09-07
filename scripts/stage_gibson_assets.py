#!/usr/bin/env python3
"""Copy one licensed Gibson/F3Loc scene into the shared AV-FPLoc data root."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopyItem:
    source: Path
    destination: Path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(data_root: Path, mesh_root: Path, alignment_root: Path, target_root: Path, scene: str) -> list[CopyItem]:
    required = [
        (data_root / "gibson_t" / scene / "map.png", target_root / "datasets/f3loc/gibson_t" / scene / "map.png"),
        (data_root / "gibson_t" / scene / "poses.txt", target_root / "datasets/f3loc/gibson_t" / scene / "poses.txt"),
        (data_root / "desdf" / scene / "desdf.npy", target_root / "datasets/f3loc/desdf" / scene / "desdf.npy"),
        (alignment_root / scene / f"{scene}_stage_metrics.json", target_root / "alignment" / scene / f"{scene}_stage_metrics.json"),
    ]
    mesh_dir = mesh_root / scene
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Gibson mesh directory does not exist: {mesh_dir}")
    required.extend((path, target_root / "meshes/gibson_v2" / scene / path.relative_to(mesh_dir)) for path in mesh_dir.rglob("*") if path.is_file())
    plan = [CopyItem(source, destination) for source, destination in required]
    missing = [str(item.source) for item in plan if not item.source.is_file()]
    if missing:
        raise FileNotFoundError("Required Gibson assets are missing:\n" + "\n".join(missing))
    return plan


def stage(plan: list[CopyItem], target_root: Path, scene: str, dry_run: bool) -> None:
    if dry_run:
        for item in plan:
            print(f"COPY {item.source} -> {item.destination}")
        return
    manifest = []
    for item in plan:
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, item.destination)
        manifest.append({"source": str(item.source), "destination": str(item.destination), "sha256": _hash(item.destination)})
    manifest_path = target_root / "alignment" / scene / "asset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"scene": scene, "files": manifest}, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Springhill")
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--source-mesh-root", type=Path, required=True)
    parser.add_argument("--source-alignment-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=Path("/file2/jeongeon/AV-FPLoc"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stage(build_plan(args.source_data_root, args.source_mesh_root, args.source_alignment_root, args.target_root, args.scene), args.target_root, args.scene, args.dry_run)


if __name__ == "__main__":
    main()
