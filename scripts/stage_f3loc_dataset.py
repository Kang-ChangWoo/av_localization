#!/usr/bin/env python3
"""Copy the files training needs off the shared dataset NFS.

The Gibson collections live on a shared NFS export whose throughput is outside
this project's control; when it degrades, seven-GPU training goes fully I/O
bound and the page cache does not hold the working set. This stages a local
copy of exactly the files the training path opens, so a run is insulated from
the source export.

The copy is resumable: a file whose destination already has the same size is
skipped, so the script can be re-run after an interruption.

    python scripts/stage_f3loc_dataset.py --datasets gibson_f --splits train val
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-scene files the training and evaluation paths actually read.
SCENE_FILES = ("depth40.txt", "depth160.txt", "poses.txt", "map.png")


def plan_scene(src_scene: Path, dst_scene: Path) -> list[tuple[Path, Path, int]]:
    jobs: list[tuple[Path, Path, int]] = []
    for name in SCENE_FILES:
        src = src_scene / name
        if src.exists():
            jobs.append((src, dst_scene / name, src.stat().st_size))
    src_rgb = src_scene / "rgb"
    if src_rgb.is_dir():
        for name in os.listdir(src_rgb):
            src = src_rgb / name
            jobs.append((src, dst_scene / "rgb" / name, src.stat().st_size))
    return jobs


def copy_one(job: tuple[Path, Path, int]) -> int:
    src, dst, size = job
    if dst.exists() and dst.stat().st_size == size:
        return 0  # already staged
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="/root/jeongeon/AV-FPLoc/datasets/f3loc")
    parser.add_argument("--dst", default=str(REPO_ROOT / "data" / "f3loc"))
    parser.add_argument("--datasets", nargs="+", default=["gibson_f"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--desdf", action="store_true", help="also stage the DESDF caches (needed for evaluation)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    src_root, dst_root = Path(args.src), Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path, int]] = []
    for name in args.datasets:
        src_ds, dst_ds = src_root / name, dst_root / name
        dst_ds.mkdir(parents=True, exist_ok=True)
        split_file = src_ds / "split.yaml"
        shutil.copyfile(split_file, dst_ds / "split.yaml")
        split = yaml.safe_load(split_file.read_text())
        for key in args.splits:
            for scene in split.get(key, []):
                jobs.extend(plan_scene(src_ds / scene, dst_ds / scene))

    if args.desdf:
        src_desdf = src_root / "desdf"
        for scene in os.listdir(src_desdf):
            f = src_desdf / scene / "desdf.npy"
            if f.exists():
                jobs.append((f, dst_root / "desdf" / scene / "desdf.npy", f.stat().st_size))

    total_bytes = sum(size for _, _, size in jobs)
    print(f"staging {len(jobs)} files, {total_bytes / 1e9:.1f} GB")
    print(f"  {src_root} -> {dst_root}")

    start = time.time()
    copied = done = skipped = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(copy_one, job) for job in jobs]
        for future in as_completed(futures):
            size = future.result()
            copied += size
            done += 1
            if size == 0:
                skipped += 1
            if done % 2000 == 0:
                elapsed = time.time() - start
                rate = copied / 1e6 / elapsed if elapsed else 0
                remaining = (total_bytes - copied) / 1e6 / rate if rate else 0
                print(f"  {done}/{len(jobs)} files, {copied/1e9:.1f} GB, "
                      f"{rate:.0f} MB/s, ~{remaining/60:.0f} min left")

    elapsed = time.time() - start
    print(f"done: {done} files ({skipped} already present), {copied/1e9:.1f} GB "
          f"in {elapsed/60:.1f} min ({copied/1e6/elapsed:.0f} MB/s)")
    print(f"point configs at: data.root: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
