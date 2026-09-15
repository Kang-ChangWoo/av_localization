#!/usr/bin/env python3
"""Queue: validation grids -> validation extractions -> validation-only selection.

Runs unattended behind scripts/render_val_grids.sh. As each benchmark's
validation grids finish merging, its extractions start on free GPUs: the three
backbones, without a projection and under each projection source whose weights
exist. When a benchmark's validation tables are all on disk, scripts/val_select.py
chooses every part of the method on them and evaluates the test rooms once.

    python -u feasible/run_val_pipeline.py
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys
sys.path.insert(0, str(HERE))
from run_projection_matrix import BASE, COND, GPUS, PREFIX, PY, SOURCES, W, gpu_free, log  # noqa: E402

AN = ROOT / "outputs" / "analysis"
VAL_ROOMS = {
    "replica": "apartment_1 frl_apartment_4 office_3",
    "mp3d": "EU6Fwq7SyZv_f2 1LXtFkjw3qL_f2 1LXtFkjw3qL_f0 1pXnuDYAj8r_f1 r47D5H71a5s_f0 ZMojNkEp431_f0",
    "s3d": ("scene_03242 scene_03209 scene_03203 scene_03220 scene_03248 scene_03215 "
            "scene_03231 scene_03236 scene_03223 scene_03208"),
}


def val_tag(backbone, source):
    return f"{PREFIX[backbone]}_val" + (f"proj{source}" if source else "")


def grids_ready(ds):
    return all((ROOT / "outputs" / "acoustic_grid_val" / ds / f"{s}.npz").exists()
               for s in VAL_ROOMS[ds].split())


def table(ds, tag):
    return AN / f"queries_{COND[ds]}_{tag}.csv"


def command(ds, backbone, source, gpu):
    base = BASE[(ds, backbone)]
    base = re.sub(r"--scenes .*? --condition", f"--scenes {VAL_ROOMS[ds]} --condition", base)
    base = re.sub(r"--grid-dir \S+", f"--grid-dir outputs/acoustic_grid_val/{ds}", base)
    tag = val_tag(backbone, source)
    suffix = tag[len(PREFIX[backbone]):]
    proj = f" --acoustic-projection {W[source]}" if source else ""
    return (f"{PY} scripts/extract_mode_table.py --gpu {gpu} {base} --desdf-dir outputs/desdf_val/{ds} "
            f"--tag {suffix}{proj} > logs/extract_{tag}.log 2>&1"), tag


def main() -> int:
    jobs = [(ds, bb, src) for ds in VAL_ROOMS for bb in ("f3loc", "unloc", "disco")
            for src in (None,) + tuple(SOURCES)]
    jobs = [j for j in jobs if not table(j[0], val_tag(j[1], j[2])).exists()]
    log(f"{len(jobs)} validation extractions to run")
    running, selected = {}, set()
    while jobs or running or len(selected) < len(VAL_ROOMS):
        for g, (p, key) in list(running.items()):
            if p.poll() is not None:
                ok = table(key[0], val_tag(key[1], key[2])).exists()
                log(f"gpu {g}: {key} {'done' if ok else 'FAILED (no table)'}")
                del running[g]
        for g in GPUS:
            if g in running or not gpu_free(g):
                continue
            i = next((i for i, (ds, bb, src) in enumerate(jobs)
                      if grids_ready(ds) and (src is None or W[src].exists())), None)
            if i is None:
                break
            key = jobs.pop(i)
            cmd, tag = command(*key, g)
            running[g] = (subprocess.Popen(cmd, shell=True, cwd=ROOT), key)
            log(f"gpu {g}: started {key} -> {tag}")
        # a benchmark is selected once none of its extractions are pending or running
        for ds in VAL_ROOMS:
            if ds in selected:
                continue
            pending = [j for j in jobs if j[0] == ds] + [k for _, k in running.values() if k[0] == ds]
            if pending or not grids_ready(ds):
                continue
            log(f"validation selection on {ds}")
            subprocess.run(f"{PY} scripts/val_select.py --dataset {ds} > feasible/logs/VAL_{ds}.log 2>&1",
                           shell=True, cwd=ROOT)
            selected.add(ds)
        time.sleep(90)
    log("VAL_PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
