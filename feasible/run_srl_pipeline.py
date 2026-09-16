#!/usr/bin/env python3
"""SemRayLoc as a fourth visual backbone, end to end under the headline protocol.

For each benchmark: wait for scripts/train_semrayloc.py to finish both ray
networks on the benchmark's training rooms, take the checkpoint with the lowest
validation loss (the validation rooms choose, as everywhere), link it as
best.ckpt, then extract the mode tables on the test rooms and on the validation
rooms, each without a projection and under the benchmark's own projection
(R for Replica, M for Matterport3D, the pooled B for Structured3D, the same
weights the other backbones use, since W is trained on recordings alone), and
run scripts/val_select.py --fixed --source indomain on them: the three scalars
chosen on validation, test evaluated once. Results land in
feasible/results/VAL_<ds>_fixed_indomain_srl.{md,json}.

    python -u feasible/run_srl_pipeline.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_projection_matrix import BASE, COND, IDENT, PREFIX, PY, SRL, W, log, tag_of  # noqa: E402
from run_val_pipeline import VAL_ROOMS, gpu_free_mib, val_tag  # noqa: E402

AN = ROOT / "outputs" / "analysis"
OWN = {"replica": "R", "mp3d": "M", "s3d": "B"}
GPUS = list(range(8))
NEED = 5000


def best_checkpoint(d: Path) -> Path | None:
    if not (d / "final_metrics.json").exists():
        return None
    cands = []
    for f in d.glob("*_net-epoch=*-loss-valid=*.ckpt"):
        m = re.search(r"loss-valid=([0-9.]+)\.ckpt$", f.name)
        if m:
            cands.append((float(m.group(1)), f))
    if not cands:
        return None
    return min(cands)[1]


def link_best(ds: str) -> bool:
    for net in ("semantic", "depth"):
        d = SRL / ds / net
        b = best_checkpoint(d)
        if b is None:
            return False
        lnk = d / "best.ckpt"
        if lnk.is_symlink() or lnk.exists():
            lnk.unlink()
        lnk.symlink_to(b.name)
        log(f"{ds}/{net}: best {b.name}")
    return True


def table(ds, tag):
    return AN / f"queries_{COND[ds]}_{tag}.csv"


def jobs_for(ds):
    """(tag, command) for the four extractions of one benchmark."""
    out = []
    base = BASE[(ds, "srl")]
    src = OWN[ds]
    for split in ("test", "val"):
        b = base
        if split == "val":
            b = re.sub(r"--scenes .*? --condition", f"--scenes {VAL_ROOMS[ds]} --condition", b)
            b = re.sub(r"--grid-dir \S+", f"--grid-dir outputs/acoustic_grid_val/{ds}", b)
            b += f" --desdf-dir outputs/desdf_val/{ds}"
        for s in (None, src):
            tag = (IDENT[(ds, "srl")] if s is None else tag_of(ds, "srl", s)) if split == "test" else val_tag("srl", s, ds)
            suffix = tag[len(PREFIX["srl"]):]
            proj = f" --acoustic-projection {W[s]}" if s else ""
            out.append((tag, f"{PY} scripts/extract_mode_table.py {b} --tag {suffix}{proj} > logs/extract_{tag}.log 2>&1"))
    return out


def main() -> int:
    (ROOT / "logs").mkdir(exist_ok=True)
    pending = {ds: None for ds in OWN}          # ds -> list of (tag, cmd) once training is done
    running = {}                                # gpu -> (Popen, ds, tag)
    selected = set()
    while len(selected) < len(OWN):
        for ds in OWN:
            if pending[ds] is None and link_best(ds):
                pending[ds] = [(t, c) for t, c in jobs_for(ds) if not table(ds, t).exists()]
                log(f"{ds}: {len(pending[ds])} extractions to run")
        for g, (p, ds, tag) in list(running.items()):
            if p.poll() is not None:
                log(f"gpu {g}: {ds} {tag} {'done' if table(ds, tag).exists() else 'FAILED'}")
                del running[g]
        for g in GPUS:
            if g in running or gpu_free_mib(g) < NEED:
                continue
            nxt = next(((ds, j) for ds in OWN if pending[ds] for j in pending[ds][:1]), None)
            if nxt is None:
                continue
            ds, (tag, cmd) = nxt
            pending[ds].remove((tag, cmd))
            running[g] = (subprocess.Popen(f"{cmd.replace('extract_mode_table.py', f'extract_mode_table.py --gpu {g}')}",
                                           shell=True, cwd=ROOT), ds, tag)
            log(f"gpu {g}: started {ds} {tag}")
        for ds in OWN:
            if ds in selected or pending[ds] is None or pending[ds] or any(r[1] == ds for r in running.values()):
                continue
            if not all(table(ds, t).exists() for t, _ in jobs_for(ds)):
                log(f"{ds}: a table is missing after extraction; not selecting")
                selected.add(ds)
                continue
            log(f"{ds}: validation selection (SemRayLoc)")
            subprocess.run(f"{PY} scripts/val_select.py --dataset {ds} --backbones srl --fixed --source indomain "
                           f"--out feasible/results/VAL_{ds}_fixed_indomain_srl.md > feasible/logs/VAL_{ds}_srl.log 2>&1",
                           shell=True, cwd=ROOT)
            selected.add(ds)
        time.sleep(120)
    log("SRL_PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
