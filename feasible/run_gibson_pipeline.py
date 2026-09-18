#!/usr/bin/env python3
"""Gibson under the headline protocol, end to end, unattended.

Waits for what the other benchmarks already have: the three visual backbones
trained on Gibson's training floors (F3Loc mono by scripts/train_visual.py,
UnLoc and DisCo-FLoc RRP by their own trainers on symlinked data), the
projection W_G trained on the training floors' recording pairs, and the
acoustic candidate grids of the 69 test floors and 6 validation floors. Then,
per backbone, extracts the mode tables on test and validation floors without a
projection and under W_G, and runs scripts/val_select.py --fixed --source
indomain: three scalars chosen on the validation floors, test evaluated once.
Results: feasible/results/VAL_gibson_fixed_indomain.{md,json}, from which
scripts/val_tables.py adds the Gibson rows.

    python -u feasible/run_gibson_pipeline.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_projection_matrix import COND, DISCO, IDENT, PREFIX, PY, UNLOC, W, log, tag_of  # noqa: E402
from run_val_pipeline import VAL_ROOMS, gpu_free_mib, val_tag  # noqa: E402

AN = ROOT / "outputs" / "analysis"
DS = "gibson"
ROOT_DS = Path("/root/storage/echoloc_dataset/gibson")
GRID = ROOT / "outputs" / "acoustic_grid_gibson"
GRID_VAL = ROOT / "outputs" / "acoustic_grid_val" / "gibson"
NEED = {"unloc": 12000, "f3loc": 4000, "disco": 4000}
GPUS = list(range(8))
TEST = json.loads((ROOT_DS / "dataset_meta.json").read_text())["split"]["test"]


def ckpt(backbone):
    if backbone == "f3loc":
        p = ROOT / "outputs" / "visual" / "gibson" / "mono_lr3e4" / "mono.ckpt"
        return p if p.exists() else None
    if backbone == "unloc":
        # UnLoc keeps the best-validation checkpoint as epoch=*.ckpt and the running one as last.ckpt;
        # the run is finished once its trainer process is gone
        d = UNLOC / "version_12" / "checkpoints"
        done = subprocess.run(["pgrep", "-f", "train.py --dataset_path data_gibson"], capture_output=True).returncode != 0
        c = sorted(d.glob("epoch=*.ckpt")) if d.exists() else []
        return c[-1] if c and done else None
    if backbone == "disco":
        runs = sorted((DISCO / "logs" / "rrp_runs").glob("rrp_gibson_*"))
        if not runs:
            return None
        d = runs[-1] / "checkpoints"
        done = subprocess.run(["pgrep", "-f", "rrp_gibson.yaml"], capture_output=True).returncode != 0
        best = [(float(m.group(1)), f) for f in d.glob("epoch=*-val_action_loss=*.ckpt")
                if (m := re.search(r"val_action_loss=([0-9.]+)\.ckpt$", f.name))] if d.exists() else []
        return min(best)[1] if best and done else None
    return None


def base(backbone, split):
    scenes = TEST if split == "test" else VAL_ROOMS[DS].split()
    extra = {"f3loc": f"--backbone f3loc_mono --checkpoint {ckpt('f3loc')}",
             "unloc": f"--backbone unloc --checkpoint {ckpt('unloc')}",
             "disco": f"--backbone disco_rrp --checkpoint {ckpt('disco')}"}[backbone]
    grid = GRID if split == "test" else GRID_VAL
    desdf = "" if split == "test" else f" --desdf-dir outputs/desdf_val/{DS}"
    return (f"--dataset-root {ROOT_DS} --collections gibson_f gibson_g --scenes {' '.join(scenes)} "
            f"--condition {COND[DS]} --grid-dir {grid.relative_to(ROOT)}{desdf} {extra} "
            f"--feature stft_band --nfft 256 --hop 64 --n-poses 40")


def table(tag):
    return AN / f"queries_{COND[DS]}_{tag}.csv"


def jobs_for(backbone):
    out = []
    for split in ("test", "val"):
        for s in (None, "G"):
            tag = (IDENT[(DS, backbone)] if s is None else tag_of(DS, backbone, s)) if split == "test" else val_tag(backbone, s, DS)
            suffix = tag[len(PREFIX[backbone]):]
            proj = f" --acoustic-projection {W[s]}" if s else ""
            out.append((tag, f"{PY} scripts/extract_mode_table.py {base(backbone, split)} --tag '{suffix}'{proj} "
                             f"> logs/extract_{tag}.log 2>&1"))
    return out


def grids_ready():
    return all((GRID / f"{s}.npz").exists() for s in TEST) and all((GRID_VAL / f"{s}.npz").exists() for s in VAL_ROOMS[DS].split())


def main() -> int:
    (ROOT / "logs").mkdir(exist_ok=True)
    pending = {bb: None for bb in NEED}
    running = {}
    done = set()
    while len(done) < len(NEED):
        ready = grids_ready() and W["G"].exists()
        for bb in NEED:
            if pending[bb] is None and ready and ckpt(bb) is not None:
                pending[bb] = [(t, c) for t, c in jobs_for(bb) if not table(t).exists()
                               and subprocess.run(["pgrep", "-f", f"extract_{t}.log"], capture_output=True).returncode != 0]
                log(f"{bb}: checkpoint {ckpt(bb)}; {len(pending[bb])} extractions to run")
        for g, (p, bb, tag) in list(running.items()):
            if p.poll() is not None:
                log(f"gpu {g}: {bb} {tag} {'done' if table(tag).exists() else 'FAILED'}")
                del running[g]
        for g in GPUS:
            if g in running:
                continue
            free = gpu_free_mib(g)
            nxt = next(((bb, j) for bb in NEED if pending[bb] and free >= NEED[bb] for j in pending[bb][:1]), None)
            if nxt is None:
                continue
            bb, (tag, cmd) = nxt
            pending[bb].remove((tag, cmd))
            running[g] = (subprocess.Popen(cmd.replace("extract_mode_table.py", f"extract_mode_table.py --gpu {g}"),
                                           shell=True, cwd=ROOT), bb, tag)
            log(f"gpu {g}: started {bb} {tag}")
        for bb in NEED:
            if bb in done or pending[bb] is None or pending[bb] or any(r[1] == bb for r in running.values()):
                continue
            if all(table(t).exists() for t, _ in jobs_for(bb)):
                done.add(bb)
                log(f"{bb}: all four tables on disk")
            else:
                log(f"{bb}: a table is missing after extraction; giving up on it")
                done.add(bb)
        time.sleep(180)
    ok = [bb for bb in NEED if all(table(t).exists() for t, _ in jobs_for(bb))]
    log(f"validation selection on Gibson for {ok}")
    subprocess.run(f"{PY} scripts/val_select.py --dataset gibson --backbones {' '.join(ok)} --fixed --source indomain "
                   f"> feasible/logs/VAL_gibson.log 2>&1", shell=True, cwd=ROOT)
    subprocess.run(f"{PY} scripts/val_tables.py > /dev/null 2>&1", shell=True, cwd=ROOT)
    log("GIBSON_PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
