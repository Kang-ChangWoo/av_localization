#!/usr/bin/env python3
"""The projection transfer matrix: where W was trained against where it is used.

Rows are the training source of the acoustic projection, columns the benchmark
it is applied to, and every cell is measured by the full pipeline with rooms
held out. The off-diagonal cells are the point: a projection trained on
Matterport3D houses and applied to Replica rooms has seen nothing of Replica,
so if it still closes part of the furniture gap there, the invariance it
learned is a property of furniture rather than of those rooms, and a deployment
can ship W without training on the target building. Structured3D has no
furnished recordings, so it cannot train a projection and appears only as a
column, where it says what W costs when there is no gap to close.

Sources: R (Replica train, 11 rooms), M (Matterport3D train, 131 rooms),
B (both pooled). Targets: Replica, Matterport3D, Structured3D, each with the
three visual backbones. Extractions are spread over free GPUs; the room-CV
evaluations run once every table for a target is on disk.

    python feasible/run_projection_matrix.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = "/opt/conda/envs/f3loc/bin/python"
MET = ROOT / "outputs" / "metrics"
AN = ROOT / "outputs" / "analysis"
LOG = ROOT / "logs"
GPUS = [1, 2, 3, 4, 5, 6]

W = {"R": MET / "acoustic_projection.npz",
     "M": MET / "acoustic_projection_mp3d.npz",
     "B": MET / "acoustic_projection_both.npz",
     # T: the two above plus Structured3D's train rooms, whose recordings have no
     # furnished twin and so enter as (x, x) pairs: a position-separation term
     # only. Included because the reader will ask; the honest expectation is
     # that it changes little.
     "T": MET / "acoustic_projection_three.npz"}
SOURCES = ("R", "M", "B", "T")
POOLED = {"B": [MET / "proj_feature_cache_mp3d.npz"],
          "T": [MET / "proj_feature_cache_mp3d.npz", MET / "proj_feature_cache_s3d.npz"]}

S3D_SCENES = ("scene_03250 scene_03253 scene_03258 scene_03259 scene_03267 scene_03290 "
              "scene_03310 scene_03316 scene_03319 scene_03346 scene_03363 scene_03367 "
              "scene_03379 scene_03383 scene_03386 scene_03391 scene_03399 scene_03400 "
              "scene_03413 scene_03422 scene_03432 scene_03437 scene_03440 scene_03447 "
              "scene_03460 scene_03461 scene_03474 scene_03478 scene_03479 scene_03483")
MP3D_SCENES = ("8WUmhLawc2A_f0 EDJbREhghzL_f0 EDJbREhghzL_f1 Z6MFQCViBuw_f0 gTV8FGcVJC9_f0 "
               "gTV8FGcVJC9_f2 gTV8FGcVJC9_f4 gTV8FGcVJC9_f5 pLe4wQe7qrG_f0 q9vSo1VnCiC_f0 "
               "sT4fr6TAbpF_f0 uNb9QFRL6hY_f1")
UNLOC = ROOT.parent / "UnLoc" / "tb_logs" / "my_model"
DISCO = ROOT.parent / "DisCo-FLoc"

# base extraction command per (target, backbone), without gpu, tag or projection
BASE = {
    ("replica", "f3loc"): (
        "--dataset-root /root/storage/echoloc_dataset/replica --collections replica_f replica_g "
        "--scenes office_4 apartment_2 frl_apartment_5 --condition raw_scan_open "
        "--grid-dir outputs/acoustic_grid_v2 --backbone f3loc_mono "
        "--checkpoint outputs/echoloc_mono_fg/mono.ckpt --feature stft_band --nfft 256 --hop 64 --n-poses 100"),
    ("replica", "unloc"): (
        "--dataset-root /root/storage/echoloc_dataset/replica --collections replica_f replica_g "
        "--scenes office_4 apartment_2 frl_apartment_5 --condition raw_scan_open "
        "--grid-dir outputs/acoustic_grid_v2 --backbone unloc "
        f"--checkpoint {UNLOC}/version_1/checkpoints/epoch=19-step=1040.ckpt "
        "--feature stft_band --nfft 256 --hop 64 --n-poses 100"),
    ("replica", "disco"): (
        "--dataset-root /root/storage/echoloc_dataset/replica --collections replica_f replica_g "
        "--scenes office_4 apartment_2 frl_apartment_5 --condition raw_scan_open "
        "--grid-dir outputs/acoustic_grid_v2 --backbone disco_rrp "
        # the in-domain ray predictor, trained on Replica's training rooms like the
        # other two backbones; the Gibson checkpoint reads 31.0% here against 40.5%
        f"--checkpoint {DISCO}/logs/rrp_runs/rrp_replica_f_20260909_191458/checkpoints/epoch=05-val_action_loss=0.30.ckpt "
        "--feature stft_band --nfft 256 --hop 64 --n-poses 100"),
    ("mp3d", "f3loc"): (
        "--dataset-root /root/storage/echoloc_dataset/mp3d --collections mp3d_f mp3d_g "
        f"--scenes {MP3D_SCENES} --condition raw_scan_open --grid-dir outputs/acoustic_grid_mp3d "
        "--backbone f3loc_mono --checkpoint outputs/visual/mp3d/mono_lr3e4/mono.ckpt "
        "--feature stft_band --nfft 256 --hop 64 --n-poses 40"),
    ("mp3d", "unloc"): (
        "--dataset-root /root/storage/echoloc_dataset/mp3d --collections mp3d_f mp3d_g "
        f"--scenes {MP3D_SCENES} --condition raw_scan_open --grid-dir outputs/acoustic_grid_mp3d "
        f"--backbone unloc --checkpoint {UNLOC}/version_6/checkpoints/epoch=12-step=10829.ckpt "
        "--feature stft_band --nfft 256 --hop 64 --n-poses 40"),
    ("mp3d", "disco"): (
        "--dataset-root /root/storage/echoloc_dataset/mp3d --collections mp3d_f mp3d_g "
        f"--scenes {MP3D_SCENES} --condition raw_scan_open --grid-dir outputs/acoustic_grid_mp3d "
        f"--backbone disco_rrp --checkpoint {DISCO}/logs/rrp_runs/rrp_mp3d_lr1e4_20260911_202717/"
        "checkpoints/epoch=03-val_action_loss=0.92.ckpt --feature stft_band --nfft 256 --hop 64 --n-poses 40"),
    ("s3d", "f3loc"): (
        "--dataset-root /root/storage/echoloc_dataset/s3d --collections s3d "
        f"--scenes {S3D_SCENES} --condition floorplan_closed --grid-dir outputs/acoustic_grid_s3d "
        "--backbone f3loc_mono --checkpoint outputs/visual/s3d/mono_lr3e4/mono.ckpt "
        "--feature stft_band --nfft 256 --hop 64 --n-rays 7 --f-w 0.5959 --n-poses 20"),
    ("s3d", "unloc"): (
        "--dataset-root /root/storage/echoloc_dataset/s3d --collections s3d "
        f"--scenes {S3D_SCENES} --condition floorplan_closed --grid-dir outputs/acoustic_grid_s3d "
        f"--backbone unloc --checkpoint {UNLOC}/version_10/checkpoints/epoch=13-step=7532.ckpt "
        "--feature stft_band --nfft 256 --hop 64 --n-rays 7 --f-w 0.5959 --n-poses 20"),
    ("s3d", "disco"): (
        "--dataset-root /root/storage/echoloc_dataset/s3d --collections s3d "
        f"--scenes {S3D_SCENES} --condition floorplan_closed --grid-dir outputs/acoustic_grid_s3d "
        f"--backbone disco_rrp --checkpoint {DISCO}/logs/rrp_runs/rrp_s3d_20260911_132644/"
        "checkpoints/epoch=06-val_action_loss=0.90.ckpt --feature stft_band --nfft 256 --hop 64 "
        "--n-rays 7 --f-w 0.5959 --n-poses 20"),
}
# SemRayLoc (depth rays + semantic rays, ICCV 2025). Its depth net is F3Loc's
# mono net, so the F3Loc mono checkpoint trained here on the benchmark's own
# training rooms serves as it and the row differs from F3Loc's by the semantic
# branch alone; the semantic net is trained by scripts/train_semrayloc.py, the
# semantic DESDF by scripts/build_semantic_desdf.py. Added after the three
# backbones above; feasible/run_srl_pipeline.py runs it end to end.
SRL = ROOT / "outputs" / "srl"
BASE[("replica", "srl")] = (
    "--dataset-root /root/storage/echoloc_dataset/replica --collections replica_f replica_g "
    "--scenes office_4 apartment_2 frl_apartment_5 --condition raw_scan_open "
    f"--grid-dir outputs/acoustic_grid_v2 --backbone semrayloc --checkpoint {SRL}/replica/semantic/best.ckpt "
    f"--depth-ckpt outputs/echoloc_mono_fg/mono.ckpt --depth-arch f3loc --semdesdf-dir outputs/semdesdf/replica "
    "--feature stft_band --nfft 256 --hop 64 --n-poses 100")
BASE[("mp3d", "srl")] = (
    "--dataset-root /root/storage/echoloc_dataset/mp3d --collections mp3d_f mp3d_g "
    f"--scenes {MP3D_SCENES} --condition raw_scan_open --grid-dir outputs/acoustic_grid_mp3d "
    f"--backbone semrayloc --checkpoint {SRL}/mp3d/semantic/best.ckpt --depth-ckpt outputs/visual/mp3d/mono_lr3e4/mono.ckpt --depth-arch f3loc "
    "--semdesdf-dir outputs/semdesdf/mp3d --feature stft_band --nfft 256 --hop 64 --n-poses 40")
BASE[("s3d", "srl")] = (
    "--dataset-root /root/storage/echoloc_dataset/s3d --collections s3d "
    f"--scenes {S3D_SCENES} --condition floorplan_closed --grid-dir outputs/acoustic_grid_s3d "
    f"--backbone semrayloc --checkpoint {SRL}/s3d/semantic/best.ckpt --depth-ckpt outputs/visual/s3d/mono_lr3e4/mono.ckpt --depth-arch f3loc "
    "--semdesdf-dir outputs/semdesdf/s3d --feature stft_band --nfft 256 --hop 64 --n-rays 7 --f-w 0.5959 --n-poses 20")
PREFIX = {"f3loc": "f3loc_mono", "unloc": "unloc", "disco": "disco_rrp", "srl": "semrayloc"}
COND = {"replica": "raw_scan_open", "mp3d": "raw_scan_open", "s3d": "floorplan_closed"}
# the identity (no projection) table each target already has
IDENT = {("replica", "f3loc"): "f3STFT", ("replica", "unloc"): "unlocSTFT", ("replica", "disco"): "discoID",
         ("mp3d", "f3loc"): "f3loc_mono_mp3d12", ("mp3d", "unloc"): "unloc_mp3d12",
         ("mp3d", "disco"): "disco_rrp_mp3d12",
         ("s3d", "f3loc"): "f3loc_mono_s3d", ("s3d", "unloc"): "unloc_s3d", ("s3d", "disco"): "disco_rrp_s3d",
         ("replica", "srl"): "semrayloc", ("mp3d", "srl"): "semrayloc_mp3d12", ("s3d", "srl"): "semrayloc_s3d"}
# tables that already exist for a (target, backbone, source) and need no extraction
EXISTING = {("replica", "f3loc", "R"): "f3loc_mono_proj", ("replica", "unloc", "R"): "unloc_uproj",
            ("mp3d", "f3loc", "M"): "f3loc_mono_mp3d12proj", ("mp3d", "unloc", "M"): "unloc_mp3d12proj",
            ("mp3d", "disco", "M"): "disco_rrp_mp3d12proj"}


def tag_of(target, backbone, source):
    if (target, backbone, source) in EXISTING:
        return EXISTING[(target, backbone, source)]
    suffix = {"replica": "", "mp3d": "_mp3d12", "s3d": "_s3d"}[target]
    return f"{PREFIX[backbone]}{suffix}proj{source}"


def table(target, tag):
    return AN / f"queries_{COND[target]}_{tag}.csv"


def gpu_free(g):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                          "-i", str(g)], capture_output=True, text=True).stdout.strip()
    return out.isdigit() and int(out) < 1500


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def wait_for(paths, what):
    paths = [Path(p) for p in paths]
    while not all(p.exists() for p in paths):
        log(f"waiting for {what}: {[p.name for p in paths if not p.exists()]}")
        time.sleep(120)


def main() -> int:
    LOG.mkdir(exist_ok=True)
    # One scheduler for everything before the room CV. A pooled projection is
    # trained as soon as its caches exist; an extraction starts as soon as its
    # projection exists and a GPU is free. Nothing waits for the Matterport3D
    # projection unless it needs it, so the extractions under the Replica
    # projection run while that cache is still being read.
    jobs = []
    for target in ("replica", "mp3d", "s3d"):
        for backbone in ("f3loc", "unloc", "disco"):
            for source in SOURCES:
                key = (target, backbone, source)
                if key in EXISTING or table(target, tag_of(*key)).exists():
                    continue
                jobs.append(key)
    log(f"{len(jobs)} extractions to run")
    running: dict[int, tuple[subprocess.Popen, object]] = {}
    training: set = set()
    while jobs or running or any(not W[s].exists() for s in POOLED):
        for g, (p, key) in list(running.items()):
            if p.poll() is not None:
                if isinstance(key, str):
                    log(f"gpu {g}: projection {key} {'done' if W[key].exists() else 'FAILED'}")
                    training.discard(key)
                else:
                    ok = table(key[0], tag_of(*key)).exists()
                    log(f"gpu {g}: {key} {'done' if ok else 'FAILED (no table)'}")
                del running[g]
        for g in GPUS:
            if g in running or not gpu_free(g):
                continue
            src = next((s for s, extra in POOLED.items()
                        if not W[s].exists() and s not in training
                        and all(e.exists() for e in extra)), None)
            if src is not None:
                extra = POOLED[src]
                cmd = (f"{PY} feasible/run_invariant_projection.py --cache {MET}/proj_feature_cache.npz "
                       f"--extra-cache {' '.join(map(str, extra))} --skip-grid-test --gpu {g} "
                       f"--weights {W[src]} --out feasible/results/P_invariant_projection_{src}.md "
                       f"> logs/P_{src}.log 2>&1")
                running[g] = (subprocess.Popen(cmd, shell=True, cwd=ROOT), src)
                training.add(src)
                log(f"gpu {g}: training pooled projection {src}")
                continue
            i = next((i for i, k in enumerate(jobs) if W[k[2]].exists()), None)
            if i is None:
                continue
            key = jobs.pop(i)
            tag = tag_of(*key)
            suffix = tag[len(PREFIX[key[1]]):]
            cmd = (f"{PY} scripts/extract_mode_table.py --gpu {g} {BASE[(key[0], key[1])]} "
                   f"--tag {suffix} --acoustic-projection {W[key[2]]} > logs/extract_{tag}.log 2>&1")
            running[g] = (subprocess.Popen(cmd, shell=True, cwd=ROOT), key)
            log(f"gpu {g}: started {key} -> {tag}")
        time.sleep(60)

    # 4. room-CV per target over identity and the three sources
    for target, extra in (("replica", ""), ("mp3d", "--folds 4"), ("s3d", "--folds 2")):
        tags = []
        for backbone in ("f3loc", "unloc", "disco"):
            tags.append(IDENT[(target, backbone)])
            tags += [tag_of(target, backbone, s) for s in SOURCES]
        wait_for([table(target, t) for t in tags], f"{target} tables")
        log(f"room CV on {target}")
        subprocess.run(f"{PY} scripts/room_cv_eval.py --backbones {' '.join(tags)} --condition {COND[target]} "
                       f"--fix-structure centre quantile {extra} "
                       f"--out feasible/results/P_matrix_{target}.md "
                       f"--json-out feasible/results/P_matrix_{target}.json > feasible/logs/P_matrix_{target}.log 2>&1",
                       shell=True, cwd=ROOT)
    log("MATRIX_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
