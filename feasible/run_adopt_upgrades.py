#!/usr/bin/env python3
"""Adopt the validation-preferred checkpoints for two headline cells, unattended.

Replica DisCo RRP moves from epoch 2 to epoch 368 of the same validation-monitored
run (the two tie on val_action_loss; epoch 368 is 3 points higher in vision
recall@1m on the validation rooms). Matterport3D SemRayLoc moves from the
batch-8, 4-epoch semantic net to the batch-16 retrain (lower validation loss,
higher ray accuracy). Both paths are already updated in
feasible/run_projection_matrix.py BASE, which is where every command below reads
them from.

Per cell: archive the four current tables (test / validation, with and without
the in-domain projection) and the selection JSON/MD, extract the four tables
again on one card, re-run scripts/val_select.py --fixed --source indomain, and
regenerate the LaTeX tables. Gibson DisCo is untouched: its 50-epoch retrain
tied the current checkpoint on validation.

    python -u feasible/run_adopt_upgrades.py --gpu 7
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_projection_matrix import BASE, COND, DISCO, PREFIX, PY, SRL, log  # noqa: E402
import run_disco_reextract as RD  # noqa: E402
import run_srl_pipeline as RS  # noqa: E402

AN = ROOT / "outputs" / "analysis"
RES = HERE / "results"
STAMP = "20261013"
CELLS = {
    # (dataset, backbone key): (archive dir, selection-file infix, tag list, command list)
    ("replica", "disco"): ("archive_disco_replica_epoch02", ".disco_epoch02"),
    ("mp3d", "srl"): ("archive_srl_mp3d_b8", ".srl_b8"),
}


def commands(ds, bb):
    if bb == "disco":
        ckpt = BASE[(ds, "disco")].split("--checkpoint ")[1].split()[0]
        assert "epoch=368" in ckpt, ckpt
        return RD.commands(ds, ckpt)
    assert "mp3d_retrain_b16_e50" in BASE[(ds, "srl")]
    return RS.jobs_for(ds)


def archive(ds, bb):
    adir, infix = CELLS[(ds, bb)]
    dst = AN / adir
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for tag, _ in commands(ds, bb):
        for f in AN.glob(f"*_{tag}.*"):
            shutil.move(str(f), str(dst / f.name)); moved += 1
    sel = f"VAL_{ds}_fixed_indomain" + ("_srl" if bb == "srl" else "")
    for f in RES.glob(f"{sel}.*"):
        if ".disco_epoch02" in f.name or ".srl_b8" in f.name or ".testselected" in f.name:
            continue
        shutil.copy(str(f), str(f.with_name(f.stem + infix + f.suffix)))
    log(f"{ds}/{bb}: {moved} files archived to {dst.relative_to(ROOT)}; selection copied with {infix}")


def extract(ds, bb, gpu, max_parallel):
    todo = [(t, c) for t, c in commands(ds, bb) if not RD.table(ds, t).exists()]
    running = []
    while todo or running:
        running = [(t, p) for t, p in running if p.poll() is None]
        while todo and len(running) < max_parallel:
            tag, cmd = todo.pop(0)
            cmd = cmd.format(gpu=gpu) if "{gpu}" in cmd else cmd.replace("extract_mode_table.py", f"extract_mode_table.py --gpu {gpu}")
            log(f"{ds}/{bb}: extracting {tag} on gpu {gpu}")
            running.append((tag, subprocess.Popen(cmd, shell=True, cwd=ROOT)))
        time.sleep(30)
    # Replica's identity DisCo table keeps the legacy name discoID
    if bb == "disco":
        for f in AN.glob(f"*_{COND[ds]}_disco_rrp.*"):
            f.rename(f.with_name(f.name.replace("_disco_rrp.", "_discoID.")))
    missing = [t for t, _ in commands(ds, bb) if not RD.table(ds, t).exists()]
    if missing:
        log(f"{ds}/{bb}: MISSING tables {missing}")
        return False
    return True


def select(ds, bb):
    if bb == "srl":
        cmd = (f"{PY} scripts/val_select.py --dataset {ds} --backbones srl --fixed --source indomain "
               f"--out feasible/results/VAL_{ds}_fixed_indomain_srl.md")
    else:
        cmd = f"{PY} scripts/val_select.py --dataset {ds} --fixed --source indomain"
    log(f"{ds}/{bb}: {cmd}")
    subprocess.run(cmd + f" > feasible/logs/VAL_{ds}_{bb}_adopt_{STAMP}.log 2>&1", shell=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    if a.plan:
        for key in CELLS:
            for tag, cmd in commands(*key):
                print(tag, "\n   ", cmd[:200])
        return 0
    ok = True
    for ds, bb in CELLS:
        archive(ds, bb)
        if not extract(ds, bb, a.gpu, a.max_parallel):
            ok = False
            continue
        select(ds, bb)
    subprocess.run(f"{PY} scripts/val_tables.py", shell=True, cwd=ROOT)
    log("tables regenerated" if ok else "finished with missing tables; tables regenerated from what exists")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
