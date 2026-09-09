#!/usr/bin/env python3
"""Pull the Gibson rgb files through the page cache before training.

The dataset lives on NFS, which delivers roughly 460 img/s cold across 64
readers. Multi-GPU DDP needs several times that, and the same files re-read
from the page cache run at roughly 7,000 img/s. This script does one parallel
read pass so the first training epoch is not I/O bound.

It only reads. Nothing is written to the dataset tree or copied to local disk.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

CHUNK = 4 << 20


def scenes_for(dataset_dir: Path, splits: list[str]) -> list[str]:
    with open(dataset_dir / "split.yaml") as handle:
        split = yaml.safe_load(handle)
    names: list[str] = []
    for key in splits:
        names.extend(split.get(key, []))
    return names


def read_file(path: str) -> int:
    total = 0
    with open(path, "rb", buffering=0) as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            total += len(block)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("AVFPLOC_F3LOC_DATA", "/root/jeongeon/AV-FPLoc/datasets/f3loc"))
    parser.add_argument("--datasets", nargs="+", default=["gibson_f", "gibson_g"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.root)
    files: list[str] = []
    for name in args.datasets:
        dataset_dir = root / name
        for scene in scenes_for(dataset_dir, args.splits):
            rgb_dir = dataset_dir / scene / "rgb"
            if rgb_dir.is_dir():
                files.extend(str(rgb_dir / f) for f in os.listdir(rgb_dir))

    print(f"warming {len(files)} files from {root} with {args.threads} threads")
    start = time.time()
    total = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for size in pool.map(read_file, files):
            total += size
            done += 1
            if done % 20000 == 0:
                elapsed = time.time() - start
                print(f"  {done}/{len(files)} files, {total/1e9:.1f} GB, {total/1e6/elapsed:.0f} MB/s")

    elapsed = time.time() - start
    print(f"done: {done} files, {total/1e9:.1f} GB in {elapsed:.0f}s ({total/1e6/elapsed:.0f} MB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
