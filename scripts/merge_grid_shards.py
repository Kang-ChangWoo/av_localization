#!/usr/bin/env python3
"""Merge the sharded candidate-grid renders back into one file per scene.

The re-render is split round-robin across workers so that a partial run still
covers the whole room rather than a corner of it. This puts the shards back
together and checks the two things that would silently corrupt the result: that
no cell appears twice, and that every shard agrees on the render settings.

    python scripts/merge_grid_shards.py --scenes office_4 apartment_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dir", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_v2")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_v2")
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--require-complete", action="store_true",
                   help="refuse to merge unless every shard is present")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for scene in args.scenes:
        shards = sorted(args.shard_dir.glob(f"{scene}_shard*.npz"))
        if not shards:
            print(f"[merge] {scene}: no shards")
            continue
        rirs, idxs, cfgs, grids = [], [], [], []
        for s in shards:
            b = np.load(s, allow_pickle=False)
            rirs.append(b["rir"])
            idxs.append(b["index"])
            cfgs.append(json.loads(str(b["config"])))
            grids.append(str(b["grid"]))
        expected = cfgs[0].get("shards", len(shards))
        if args.require_complete and len(shards) < expected:
            print(f"[merge] {scene}: {len(shards)}/{expected} shards, skipping")
            continue

        # every shard must have rendered with the same settings, or the merged
        # grid would be a mixture of two acoustic models
        varying = {k for c in cfgs[1:] for k, v in c.items()
                   if k not in ("shard", "out") and cfgs[0].get(k) != v}
        if varying:
            print(f"[merge] {scene}: shards disagree on {sorted(varying)}, refusing")
            continue
        if len(set(grids)) != 1:
            print(f"[merge] {scene}: shards disagree on the pose grid, refusing")
            continue

        # The engine sizes each impulse response to its own decay, so shards
        # differ in length by a few hundred samples. The scoring window is
        # 128 ms, far shorter than any of them, so truncating to the common
        # length discards only tail that is never read. The shipped recordings
        # vary the same way.
        n = min(r.shape[-1] for r in rirs)
        spread = max(r.shape[-1] for r in rirs) - n
        if spread:
            print(f"[merge] {scene}: lengths span {spread} samples "
                  f"({1000*spread/cfgs[0]['sample_rate']:.0f} ms), truncating to {n}")
        rir = np.concatenate([r[..., :n] for r in rirs], axis=0)
        index = np.concatenate(idxs, axis=0)
        keys = index[:, 0].astype(np.int64) * 100000 + index[:, 1].astype(np.int64)
        if len(np.unique(keys)) != len(keys):
            print(f"[merge] {scene}: {len(keys) - len(np.unique(keys))} duplicate cells, refusing")
            continue
        order = np.lexsort((index[:, 1], index[:, 0]))
        rir, index = rir[order], index[order]

        out = args.out_dir / f"{scene}.npz"
        cfg = dict(cfgs[0]); cfg.pop("shard", None); cfg["shards_merged"] = len(shards)
        np.savez(out, rir=rir, index=index, grid=grids[0], config=json.dumps(cfg))
        print(f"[merge] {scene}: {len(shards)} shards -> {rir.shape} "
              f"({out.stat().st_size / 1e6:.0f} MB)  "
              f"{cfg.get('sample_rate')} Hz, diffraction={cfg.get('diffraction')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
