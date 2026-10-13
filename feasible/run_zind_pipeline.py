#!/usr/bin/env python3
"""ZInD under the headline protocol, end to end, unattended.

ZInD ships no mesh, so there is no furnished condition: the query recordings and
the candidate grid are both rendered on the same wall-only floorplan proxy. The
row therefore belongs in the *matched* group with Structured3D, it does not
measure the furniture domain gap, and no amount of rendering fixes that.

Like Structured3D, ZInD cannot train an acoustic projection of its own -- a
projection is fitted on (furnished, floorplan) recording pairs and ZInD has no
furnished twin -- so its in-domain source is the pooled projection B. That is
registered in scripts/val_select.py's INDOMAIN table, not chosen here.

Per backbone this extracts the mode tables on the 252 test floors and on the 8
validation floors, each without a projection and under W_B, then runs
scripts/val_select.py --fixed --source indomain: the three scalars chosen on the
validation floors, test evaluated once. Results:
feasible/results/VAL_zind_fixed_indomain.{md,json} for the three visual
backbones and VAL_zind_fixed_indomain_srl.{md,json} for SemRayLoc, from which
scripts/val_tables.py fills the ZInD rows.

UnLoc joined on 2026-10-05, when its ZInD run finished on n6. Its checkpoint is
epoch 7 of 30: `loss-valid` bottomed there and the remaining 22 epochs were
overfitting, which the epoch budget (kept at 30 to match the Matterport3D and
Structured3D UnLoc runs) could not have known in advance.

ZIND_STAGE=val extracts the validation tables only and stops before the
selection. The validation grids are eight floors and finish hours before the
252 test ones, so running them early turns a bug on the validation side -- a
missing DESDF, a frame that disagrees, a tag that lands under the wrong name --
into something found with the test render still going, rather than at the moment
the table was due.

    python -u feasible/run_zind_pipeline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_projection_matrix import COND, IDENT, PREFIX, PY, log, tag_of, W  # noqa: E402
from run_val_pipeline import VAL_ROOMS, gpu_free_mib, val_tag  # noqa: E402

AN = ROOT / "outputs" / "analysis"
DS = "zind"
ROOT_DS = Path("/root/storage/echoloc_dataset/zind")
# The candidate grids live on this node's local disks, not in the repo: the
# shared NFS export is full. GRID is a farm of symlinks over the two render
# batches (30 floors on /data2 from 2026-09-24, the rest on /data3), because
# extract_mode_table.py takes a single --grid-dir.
GRID = Path("/root/local2/changwoo/zind_grid_all")
GRID_VAL = Path("/root/local2/changwoo/zind_acoustic_grid_val")
# the dataset ships desdf/ for its 252 test floors only
DESDF_VAL = Path("/root/local1/changwoo/zind_desdf_val")
SEMDESDF = Path("/root/local1/changwoo/zind_semdesdf")
CKPT = Path("/root/local2/changwoo/zind_table/ckpt")
MONO = Path("/root/local1/changwoo/zind_visual/zind_mono_lr3e4/mono.ckpt")
STATE = Path("/root/local1/changwoo/zind_table/state")
SRC = "B"
# UnLoc's ViT-L needs about 12 GB where the two ResNet backbones need 4
NEED = {"f3loc": 4000, "unloc": 12000, "disco": 4000, "srl": 5000}
# every card on this node can carry another tenant, so a job starts when enough
# is left for it rather than when the card is empty
GPUS = list(range(8))
# an extraction holds one floor's whole candidate grid in RAM (the rir array of
# every cell: 553 KB per cell, so 13 GB on the largest floor). Four at a time
# leaves headroom in the node's 376 GB and does not saturate /data3's read path,
# which the render lanes are still writing to.
MAX_CONC = 4
# ZInD's panorama positions, median 19 per floor, are not a dense sequence; 20
# matches Structured3D, the sibling matched-geometry benchmark whose floors carry
# 17-31 poses. Fixed in advance.
N_POSES = 20
# no floor's grid comes near this; kept as the same guard Gibson carries
MAX_GRID_GB = float(os.environ.get("ZIND_MAX_GRID_GB", 40))
ALL_TEST = json.loads((ROOT_DS / "dataset_meta.json").read_text())["split"]["test"]


def usable_test():
    """(floors to extract, floors excluded as too large, floors with no grid yet)."""
    keep, drop, pending = [], [], []
    for s in ALL_TEST:
        f = GRID / f"{s}.npz"
        if not f.exists():
            pending.append(s)
        elif f.stat().st_size > MAX_GRID_GB * 1e9:
            drop.append(s)
        else:
            keep.append(s)
    return keep, drop, pending


def ckpt(backbone):
    if backbone == "f3loc":
        return MONO if MONO.exists() else None
    if backbone == "disco":
        p = CKPT / "disco_rrp_zind_epoch08_val0.96.ckpt"
        return p if p.exists() else None
    if backbone == "unloc":
        # the only epoch=*.ckpt the run kept, i.e. the loss-valid minimum; the
        # trainer reported 'loss-valid was not in top 1' for every epoch after it
        p = CKPT / "unloc_zind_epoch07_step22632.ckpt"
        return p if p.exists() else None
    if backbone == "srl":
        p = CKPT / "srl_semantic_zind_epoch20_loss0.2507.ckpt"
        if not (p.exists() and MONO.exists()):
            return None
        # the semantic DESDF is built floor by floor by a separate fan-out, and
        # extract_mode_table.py loads it per floor with no guard, so an
        # extraction started before the fan-out reaches a floor dies partway
        # through. SemRayLoc is held back until every floor it will be scored on
        # has one.
        need = sorted(VAL_ROOMS[DS].split()) if STAGE == "val" \
            else sorted(VAL_ROOMS[DS].split()) + usable_test()[0]
        missing = [s for s in need if not (SEMDESDF / s / "semdesdf.npy").exists()]
        if missing:
            log(f"srl: waiting on {len(missing)} semantic DESDFs (e.g. {missing[:3]})")
            return None
        return p
    return None


def base(backbone, split):
    scenes = usable_test()[0] if split == "test" else sorted(VAL_ROOMS[DS].split())
    extra = {"f3loc": f"--backbone f3loc_mono --checkpoint {ckpt('f3loc')}",
             "unloc": f"--backbone unloc --checkpoint {ckpt('unloc')}",
             "disco": f"--backbone disco_rrp --checkpoint {ckpt('disco')}",
             "srl": (f"--backbone semrayloc --checkpoint {ckpt('srl')} "
                     f"--depth-ckpt {MONO} --depth-arch f3loc --semdesdf-dir {SEMDESDF}")}[backbone]
    grid = GRID if split == "test" else GRID_VAL
    desdf = "" if split == "test" else f" --desdf-dir {DESDF_VAL}"
    # no --f-w and no --n-rays: ZInD renders at the same 106.26 deg horizontal
    # field of view as Replica, Matterport3D and Gibson, so F_W = 0.375 and the
    # 11-ray fan are already the defaults. Structured3D overrides them because it
    # alone renders at 80 deg.
    return (f"--dataset-root {ROOT_DS} --collections zind --scenes {' '.join(scenes)} "
            f"--condition {COND[DS]} --grid-dir {grid}{desdf} {extra} "
            f"--feature stft_band --nfft 256 --hop 64 --n-poses {N_POSES}")


def table(tag):
    return AN / f"queries_{COND[DS]}_{tag}.csv"


def running_elsewhere(tag):
    """Is some extraction (this driver's or an earlier one's) already writing this table?"""
    return subprocess.run(["pgrep", "-f", f"extract_{tag}.log"], capture_output=True).returncode == 0


def gpu_busy(g):
    return subprocess.run(["pgrep", "-f", f"extract_mode_table.py --gpu {g} "],
                          capture_output=True).returncode == 0


STAGE = os.environ.get("ZIND_STAGE", "all")


def jobs_for(backbone):
    out = []
    for split in (("val",) if STAGE == "val" else ("test", "val")):
        for s in (None, SRC):
            tag = (IDENT[(DS, backbone)] if s is None else tag_of(DS, backbone, s)) \
                if split == "test" else val_tag(backbone, s, DS)
            suffix = tag[len(PREFIX[backbone]):]
            proj = f" --acoustic-projection {W[s]}" if s else ""
            out.append((tag, f"{PY} scripts/extract_mode_table.py {base(backbone, split)} "
                             f"--tag '{suffix}'{proj} > logs/extract_{tag}.log 2>&1"))
    return out


def grids_ready():
    """Every test floor's grid is on disk (or too large) and every validation grid is."""
    _, _, pending = usable_test()
    return not pending and all((GRID_VAL / f"{s}.npz").exists() for s in VAL_ROOMS[DS].split())


def mark(name, text=""):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / name).write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {text}\n")


def main() -> int:
    (ROOT / "logs").mkdir(exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    if not W[SRC].exists():
        log(f"the pooled projection {W[SRC]} is missing; nothing to do")
        mark("FAILED_no_projection", str(W[SRC]))
        return 1
    keep, drop, pending = usable_test()
    log(f"stage={STAGE}: {len(keep)} test floors usable, {len(drop)} too large, "
        f"{len(pending)} without a grid")
    missing_val = [s for s in sorted(VAL_ROOMS[DS].split()) if not (GRID_VAL / f"{s}.npz").exists()]
    if missing_val:
        log(f"validation grids missing: {missing_val}")
        mark("FAILED_val_grids_incomplete", " ".join(missing_val))
        return 1
    if STAGE != "val" and pending:
        log("test grids are not complete; this driver runs after the render lanes finish")
        mark("FAILED_grids_incomplete", f"{len(pending)} floors pending")
        return 1
    mark(f"EXTRACT_START_{STAGE}",
         f"{0 if STAGE == 'val' else len(keep)} test floors, {len(VAL_ROOMS[DS].split())} val floors")

    pending_jobs = {bb: None for bb in NEED}
    running = {}
    done = set()
    skipped = {}
    # A backbone whose prerequisite never arrives must not park this driver
    # forever: a driver that waits silently is indistinguishable, from the
    # outside, from one that is working. After the budget the backbone is
    # dropped, recorded, and the rest of the table is produced without it.
    t0 = time.monotonic()
    WAIT_BUDGET_S = float(os.environ.get("ZIND_WAIT_BUDGET_S", 3 * 3600))
    while len(done) < len(NEED):
        for bb in NEED:
            if pending_jobs[bb] is None and ckpt(bb) is not None:
                pending_jobs[bb] = [(t, c) for t, c in jobs_for(bb) if not table(t).exists()
                                    and not running_elsewhere(t)]
                log(f"{bb}: checkpoint {ckpt(bb)}; {len(pending_jobs[bb])} extractions to run")
            elif pending_jobs[bb] is None and time.monotonic() - t0 > WAIT_BUDGET_S:
                reason = f"prerequisite still missing after {WAIT_BUDGET_S / 3600:.1f} h"
                log(f"{bb}: {reason}; dropping it from this run")
                skipped[bb] = reason
                mark(f"SKIPPED_{bb}", reason)
                done.add(bb)
        for g, (p, bb, tag) in list(running.items()):
            if p.poll() is not None:
                log(f"gpu {g}: {bb} {tag} {'done' if table(tag).exists() else 'FAILED'}")
                del running[g]
        for g in GPUS:
            if g in running or len(running) >= MAX_CONC or gpu_busy(g):
                continue
            free = gpu_free_mib(g)
            nxt = next(((bb, j) for bb in NEED if pending_jobs[bb] and free >= NEED[bb]
                        for j in pending_jobs[bb][:1]), None)
            if nxt is None:
                continue
            bb, (tag, cmd) = nxt
            pending_jobs[bb].remove((tag, cmd))
            running[g] = (subprocess.Popen(
                cmd.replace("extract_mode_table.py", f"extract_mode_table.py --gpu {g}"),
                shell=True, cwd=ROOT), bb, tag)
            log(f"gpu {g}: started {bb} {tag}")
        for bb in NEED:
            if bb in done or pending_jobs[bb] is None or pending_jobs[bb] \
               or any(r[1] == bb for r in running.values()):
                continue
            missing = [t for t, _ in jobs_for(bb) if not table(t).exists()]
            if not missing:
                done.add(bb)
                log(f"{bb}: all four tables on disk")
            elif any(running_elsewhere(t) for t in missing):
                continue
            else:
                log(f"{bb}: {missing} missing and nothing is producing them; giving up")
                mark(f"EXTRACT_FAILED_{bb}", " ".join(missing))
                done.add(bb)
        time.sleep(120)

    if STAGE == "val":
        done_val = {bb: all(table(t).exists() for t, _ in jobs_for(bb)) for bb in NEED}
        log(f"validation tables: {done_val}")
        mark("EXTRACT_DONE_val", str(done_val))
        return 0 if any(done_val.values()) else 1

    ok_vis = [bb for bb in ("f3loc", "unloc", "disco")
              if all(table(t).exists() for t, _ in jobs_for(bb))]
    ok_srl = "srl" not in skipped and all(table(t).exists() for t, _ in jobs_for("srl"))
    if skipped:
        log(f"backbones dropped: {skipped}")
    (HERE / "results" / "zind_scenes.json").write_text(json.dumps(
        dict(used=keep, excluded=drop, max_grid_gb=MAX_GRID_GB, n_poses=N_POSES,
             val_floors=sorted(VAL_ROOMS[DS].split()), projection_source=SRC,
             backbones_extracted=ok_vis + (["srl"] if ok_srl else []),
             unloc_checkpoint="epoch 7 of 30; loss-valid minimum, the remaining 22 "
                              "epochs did not improve it. Trained with --L 0, so every "
                              "train frame was used and each image is paired with its "
                              "own depth row",
             note="query recordings and candidate grid share the same wall-only floorplan "
                  "proxy, so this is a matched-geometry row and does not measure the "
                  "furniture domain gap"), indent=1))
    mark("EXTRACT_DONE", f"visual={ok_vis} srl={ok_srl}")
    if not ok_vis and not ok_srl:
        log("no backbone has a complete table set; not selecting")
        mark("FAILED_no_complete_backbone")
        return 1

    if ok_vis:
        log(f"validation selection on ZInD for {ok_vis}")
        subprocess.run(f"{PY} scripts/val_select.py --dataset zind --backbones {' '.join(ok_vis)} "
                       f"--fixed --source indomain > feasible/logs/VAL_zind.log 2>&1",
                       shell=True, cwd=ROOT)
        mark("SELECT_VIS_DONE" if (HERE / "results" / "VAL_zind_fixed_indomain.json").exists()
             else "SELECT_VIS_FAILED")
    if ok_srl:
        log("validation selection on ZInD for SemRayLoc")
        subprocess.run(f"{PY} scripts/val_select.py --dataset zind --backbones srl --fixed "
                       f"--source indomain --out feasible/results/VAL_zind_fixed_indomain_srl.md "
                       f"> feasible/logs/VAL_zind_srl.log 2>&1", shell=True, cwd=ROOT)
        mark("SELECT_SRL_DONE" if (HERE / "results" / "VAL_zind_fixed_indomain_srl.json").exists()
             else "SELECT_SRL_FAILED")
    subprocess.run(f"{PY} scripts/val_tables.py > feasible/logs/zind_val_tables.log 2>&1",
                   shell=True, cwd=ROOT)
    mark("TABLES_DONE")
    log("ZIND_PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
