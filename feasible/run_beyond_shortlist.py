#!/usr/bin/env python3
"""Answers from outside the visual shortlist: re-extract, then evaluate A, B, C.

Re-extracts the UnLoc tables the paper's protocol uses (test and validation on
the three benchmarks, under each benchmark's own projection) and the 28
degradation levels, with the extractor's new columns: the acoustic field's own
peaks as injected hypotheses (A), from which the shortlist-rejection gap (B)
follows, and the cell-wise product errors on a temperature grid (C). Then
scripts/val_select.py --fixed --source indomain chooses each variant's setting
on validation rooms and evaluates the test rooms once, and
run_degradation_final.py applies the chosen settings, frozen, to every level.

    python -u feasible/run_beyond_shortlist.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_projection_matrix import BASE, PY, W, log  # noqa: E402
from run_val_pipeline import VAL_ROOMS, gpu_free_mib, command as val_command  # noqa: E402
from run_degradation_final import LEVELS, BASE as DEG_BASE, tag as deg_tag  # noqa: E402

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
SLOTS = [(g, i) for g in GPUS for i in (0, 1)]
NEED = 5000
SRC = {"replica": "R", "mp3d": "M", "s3d": "B"}
TEST_TAG = {("replica", "unloc"): "unloc_uproj", ("mp3d", "unloc"): "unloc_mp3d12proj", ("s3d", "unloc"): "unloc_s3dprojB",
            ("replica", "f3loc"): "f3loc_mono_proj", ("mp3d", "f3loc"): "f3loc_mono_mp3d12proj", ("s3d", "f3loc"): "f3loc_mono_s3dprojB",
            ("replica", "disco"): "disco_rrpprojR", ("mp3d", "disco"): "disco_rrp_mp3d12proj", ("s3d", "disco"): "disco_rrp_s3dprojB"}
BACKBONES = sys.argv[1:] or ["unloc"]


def jobs():
    out = []
    from run_projection_matrix import PREFIX
    for bb in BACKBONES:
        for ds in ("replica", "mp3d", "s3d"):
            t = TEST_TAG[(ds, bb)]
            suffix = t[len(PREFIX[bb]):]
            out.append((f"{PY} scripts/extract_mode_table.py --gpu {{g}} {BASE[(ds, bb)]} --tag {suffix} "
                        f"--acoustic-projection {W[SRC[ds]]} > logs/extract_{t}.log 2>&1", t))
            cmd, tag = val_command(ds, bb, SRC[ds], "{g}")
            out.append((cmd, tag))
    for fam, lvl in (LEVELS if "unloc" in BACKBONES else []):
        t = deg_tag(fam, lvl)
        if fam == "clean":
            continue      # the clean level is the Replica test table above
        out.append((f"{PY} scripts/extract_mode_table.py --gpu {{g}} {DEG_BASE} --degrade {fam}:{lvl} "
                    f"--tag {t[len('unloc'):]} --acoustic-projection {W['R']} > logs/extract_{t}.log 2>&1", t))
    return out


def has_injection(tag):
    """True when the table carries the injected rows and the per-cell field dump."""
    for cond in ("raw_scan_open", "floorplan_closed"):
        p = ROOT / "outputs" / "analysis" / f"modes_{cond}_{tag}.csv"
        if p.exists():
            with open(p) as f:
                head = f.readline()
            return ",source," in head and (ROOT / "outputs" / "analysis" / f"fields_{cond}_{tag}.npz").exists()
    return False


def main() -> int:
    todo = [(c, t) for c, t in jobs() if not has_injection(t)]
    log(f"{len(todo)} extractions to redo with injected candidates")
    running = {}
    while todo or running:
        for slot, (p, t) in list(running.items()):
            if p.poll() is not None:
                log(f"gpu {slot[0]}: {t} {'done' if has_injection(t) else 'FAILED'}")
                del running[slot]
        for slot in SLOTS:
            if not todo or slot in running or gpu_free_mib(slot[0]) < NEED:
                continue
            cmd, t = todo.pop(0)
            running[slot] = (subprocess.Popen(cmd.replace("{g}", str(slot[0])), shell=True, cwd=ROOT), t)
            log(f"gpu {slot[0]}: started {t}")
            time.sleep(20)
        time.sleep(60)
    log("EXTRACTIONS_DONE")
    for ds in ("replica", "mp3d", "s3d"):
        subprocess.run(f"{PY} scripts/val_select.py --dataset {ds} --fixed --source indomain "
                       f"--variants > feasible/logs/VAL_{ds}_variants.log 2>&1", shell=True, cwd=ROOT)
        log(f"variants selected on {ds}")
    if "unloc" in BACKBONES:
        subprocess.run(f"{PY} -u feasible/run_degradation_final.py > logs/V_final_unloc.log 2>&1", shell=True, cwd=ROOT)
    log("BEYOND_SHORTLIST_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
