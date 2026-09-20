#!/usr/bin/env python3
"""DisCo-FLoc RRP rows re-extracted from validation-monitored retrains, unattended.

The four RRP runs behind the tables were trained with `val_split: test`, so the
checkpoints the tables used were chosen by the test-split action loss. The
retrains (scripts/run_rrp_val_queue.sh, configs *_val.yaml, identical except
`val_split: val`) fix that. This driver, per benchmark and in the order the
retrains finish: waits for the run to end, takes its lowest val_action_loss
checkpoint, moves the test-selected DisCo tables to
outputs/analysis/archive_testselected_disco/, extracts the four DisCo tables
(test / validation, identity / in-domain projection) on one card, re-runs
scripts/val_select.py --fixed --source indomain, and regenerates the tables.
The previous VAL_*.json/.md are kept beside the new ones with a
`.testselected_disco` infix so the change in the DisCo rows can be read off.

    python -u feasible/run_disco_reextract.py            # wait and run everything
    python -u feasible/run_disco_reextract.py --plan     # print the commands only
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_projection_matrix import BASE, COND, DISCO, IDENT, PREFIX, PY, W, log, tag_of  # noqa: E402
from run_val_pipeline import VAL_ROOMS, gpu_free_mib, val_tag  # noqa: E402
from run_gibson_pipeline import usable_test  # noqa: E402

AN = ROOT / "outputs" / "analysis"
ARCHIVE = AN / "archive_testselected_disco"
RES = HERE / "results"
RUN_NAME = {"s3d": "rrp_s3d_val", "gibson": "rrp_gibson_val", "mp3d": "rrp_mp3d_lr1e4_val", "replica": "rrp_replica_f_val"}
INDOMAIN = {"replica": "R", "mp3d": "M", "s3d": "B", "gibson": "G"}
ORDER = ("s3d", "gibson", "mp3d", "replica")          # the order the retrains are queued
GPU = 1                                                 # the one card this session may use
NEED_MIB = 4000
POLL_S = 600


def run_dir(ds):
    runs = sorted((DISCO / "logs" / "rrp_runs").glob(f"{RUN_NAME[ds]}_*"))
    return runs[-1] if runs else None


def training(ds):
    """Is the retrain for this benchmark still running? (its config name is in the command line)"""
    return subprocess.run(["pgrep", "-f", f"{RUN_NAME[ds]}.yaml"], capture_output=True).returncode == 0


def best_ckpt(ds):
    d = run_dir(ds)
    if d is None:
        return None
    best = [(float(m.group(1)), f) for f in (d / "checkpoints").glob("epoch=*-val_action_loss=*.ckpt")
            if (m := re.search(r"val_action_loss=([0-9.]+)\.ckpt$", f.name))]
    return min(best)[1] if best else None


def base_cmd(ds, split, ckpt):
    if ds == "gibson":
        scenes = usable_test()[0] if split == "test" else VAL_ROOMS[ds].split()
        base = (f"--dataset-root /root/storage/echoloc_dataset/gibson --collections gibson_f gibson_g "
                f"--scenes {' '.join(scenes)} --condition raw_scan_open --grid-dir outputs/acoustic_grid_gibson "
                f"--backbone disco_rrp --checkpoint {ckpt} --feature stft_band --nfft 256 --hop 64 --n-poses 40")
    else:
        base = re.sub(r"--checkpoint \S+", f"--checkpoint {ckpt}", BASE[(ds, "disco")])
    if split == "val":
        base = re.sub(r"--scenes .*? --condition", f"--scenes {VAL_ROOMS[ds]} --condition", base)
        base = re.sub(r"--grid-dir \S+", f"--grid-dir outputs/acoustic_grid_val/{ds}", base)
        base += f" --desdf-dir outputs/desdf_val/{ds}"
    return base


def tags(ds):
    out = []
    for split in ("test", "val"):
        for src in (None, INDOMAIN[ds]):
            if split == "test":
                tag = IDENT[(ds, "disco")] if src is None else tag_of(ds, "disco", src)
            else:
                tag = val_tag("disco", src, ds)
            out.append((split, src, tag))
    return out


def table(ds, tag):
    return AN / f"queries_{COND[ds]}_{tag}.csv"


def commands(ds, ckpt):
    out = []
    for split, src, tag in tags(ds):
        suffix = tag[len(PREFIX["disco"]):]
        proj = f" --acoustic-projection {W[src]}" if src else ""
        out.append((tag, f"{PY} scripts/extract_mode_table.py --gpu {GPU} {base_cmd(ds, split, ckpt)} "
                         f"--tag '{suffix}'{proj} > logs/extract_{tag}.log 2>&1"))
    return out


def archive(ds):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    moved = 0
    for _, _, tag in tags(ds):
        for f in AN.glob(f"*_{tag}.*"):
            shutil.move(str(f), str(ARCHIVE / f.name)); moved += 1
    for f in RES.glob(f"VAL_{ds}_fixed_indomain.*"):
        if ".testselected_disco" not in f.name:
            shutil.copy(str(f), str(f.with_name(f.stem + ".testselected_disco" + f.suffix)))
    log(f"{ds}: {moved} test-selected DisCo files moved to {ARCHIVE.relative_to(ROOT)}")


def extract(ds, ckpt):
    todo = [(t, c) for t, c in commands(ds, ckpt) if not table(ds, t).exists()]
    tries = {}
    while todo:
        if gpu_free_mib(GPU) < NEED_MIB:
            time.sleep(60); continue
        tag, cmd = todo.pop(0)
        log(f"{ds}: extracting {tag}")
        subprocess.run(cmd, shell=True, cwd=ROOT)
        if not table(ds, tag).exists():
            tries[tag] = tries.get(tag, 0) + 1
            log(f"{ds}: {tag} produced no table (try {tries[tag]})")
            if tries[tag] < 3:
                todo.append((tag, cmd))
            else:
                return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(ORDER))
    a = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True); (HERE / "logs").mkdir(exist_ok=True)
    if a.plan:
        for ds in a.datasets:
            ck = best_ckpt(ds)
            print(f"== {ds}: run {run_dir(ds)}  training={training(ds)}  best={ck}")
            for tag, cmd in commands(ds, ck or Path("<ckpt>")):
                print(f"  [{tag}] {cmd}")
        return 0
    for ds in a.datasets:
        while training(ds) or best_ckpt(ds) is None:
            time.sleep(POLL_S)
        ck = best_ckpt(ds)
        log(f"{ds}: retrain finished, checkpoint {ck.relative_to(DISCO)}")
        archive(ds)
        if not extract(ds, ck):
            log(f"{ds}: extraction failed; stopping before selection"); return 1
        log(f"{ds}: validation selection")
        subprocess.run(f"{PY} scripts/val_select.py --dataset {ds} --fixed --source indomain "
                       f"> feasible/logs/VAL_{ds}_fixed_indomain_reselect.log 2>&1", shell=True, cwd=ROOT)
        subprocess.run(f"{PY} scripts/val_tables.py > logs/val_tables_reselect.log 2>&1", shell=True, cwd=ROOT)
        log(f"{ds}: tables regenerated; DisCo rows now from the validation-monitored run (not committed)")
    log("all benchmarks re-extracted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
