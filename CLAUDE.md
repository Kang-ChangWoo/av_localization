# CLAUDE.md — av_localization (AV-FPLoc / EchoLoc)

Shared research discipline (number provenance, verdict discipline, PyTorch correctness,
watchers, where results go) lives in the **`research-coding-rules` skill**. Follow it; it is
deliberately not repeated here.

Repo documents to read first:

- **`README.md`** — research goal, the three F3Loc gaps this repo closes, the network table,
  the visual reproduction sequence.
- **`docs/`** — `architecture.md`, `data_contracts.md`, `f3loc_baseline_facts.md`,
  `decision_log.md`, `paper_tables.md`.

Everything below is specific to this repository and was verified in these files.

## 1. The repo is the nested directory, and it depends on its siblings

The git repo is `.../av_localization/av_localization`. The **outer** `av_localization/` is a
workspace directory, not a repo, and it holds the sibling baselines the acoustic matrix reads:

```
av_localization/        <- workspace, NOT a repo
  av_localization/      <- this repo
  UnLoc/  DisCo-FLoc/  SemRayLoc/  f3loc/  vis/
```

`feasible/run_projection_matrix.py` resolves them relative to the repo:
`UNLOC = ROOT.parent/"UnLoc"/"tb_logs"/"my_model"`, `DISCO = ROOT.parent/"DisCo-FLoc"`
(`:59-60`). Moving or cloning this repo alone breaks every acoustic cell.

`third_party/f3loc` is vendored **on purpose** — `.gitignore` excludes `third_party/` and then
re-includes it, so visual training reproduces from this repo alone. Upstream's network and
localization code is byte-identical to `third_party/f3loc/PINNED_REVISION`; read it, never
write to it.

## 2. The real commands

The interpreter is pinned at `feasible/run_projection_matrix.py:31`:
`/opt/conda/envs/f3loc/bin/python` — the `f3loc` conda env (py3.8, torch 2.4.1+cu124,
lightning 2.4.0), **not** the default env.

```bash
# visual observation networks, in order (comp freezes and reuses the first two)
python scripts/train_visual.py --net mono --config configs/visual_mono.yaml
python scripts/train_visual.py --net mv   --config configs/visual_mv.yaml
python scripts/train_visual.py --net comp --config configs/visual_comp.yaml \
    --mono-ckpt outputs/mono/mono.ckpt --mv-ckpt outputs/mv/mv.ckpt
```

Each run writes `outputs/<run_name>/checkpoints/` (top-3 by `loss-valid` + `last.ckpt`), a CSV
metric log, and a stable `outputs/<run_name>/<net>.ckpt` alias at the best epoch.

Acoustic + visual matrix cells are produced by `scripts/extract_mode_table.py`, driven from
`feasible/run_*.py` (`run_projection_matrix.py`, `run_val_pipeline.py`, `run_zind_pipeline.py`,
`run_gibson_pipeline.py`, `run_srl_pipeline.py`, …).

## 3. `BASE` is the only checkpoint mapping — never hand-pick a path

`feasible/run_projection_matrix.py:70` defines `BASE`, a dict from `(dataset, backbone)` to the
**full argument string** (dataset root, collections, scenes, condition, grid dir, backbone,
checkpoint, feature, pose count). `run_projection_matrix.py:246` builds the command from it.

Directory names do not reveal which dataset a checkpoint belongs to:
`UnLoc/tb_logs/my_model/version_N` numbers **runs**, not datasets — verified in `BASE`:
`version_1` = replica, `version_6` = mp3d, `version_10` = s3d.

SemRayLoc entries are appended separately (`:129`, `:135`, `:140`) and take **two** checkpoints:
`--checkpoint outputs/srl/<ds>/semantic/best.ckpt` plus a `--depth-ckpt` and a
`--semdesdf-dir`. Its Replica depth checkpoint is `outputs/echoloc_mono_fg/mono.ckpt` while
mp3d / s3d use `outputs/visual/<ds>/mono_lr3e4/mono.ckpt` — the same asymmetry as the F3Loc row
below. There is no Gibson SemRayLoc row.

`PREFIX`, `COND` and `IDENT` right after the dict map a backbone key to its column prefix, a
dataset to its condition (`gibson` = `raw_scan_open`, `zind` = `floorplan_closed`) and each
target to its identity (no-projection) table.

## 4. Per-dataset asymmetries that break horizontal comparison

Read straight off `BASE`, not assumed:

| | replica | mp3d | s3d |
|---|---|---|---|
| `--n-poses` | 100 | 40 | 20 |
| `--condition` | `raw_scan_open` | `raw_scan_open` | `floorplan_closed` |
| extra args | — | — | `--n-rays 7 --f-w 0.5959` |
| `--grid-dir` | `outputs/acoustic_grid_v2` | `outputs/acoustic_grid_mp3d` | `outputs/acoustic_grid_s3d` |

`_f` / `_g` in a collection name is **camera motion** (forward / general), not furniture.
Furniture is the acoustic condition (`raw_scan_open` vs `floorplan_closed`).

A row from one dataset is not comparable to another's without saying which pose count and
condition it came from.

## 5. Traps in this repo

- **`scripts/extract_mode_table.py` used to pick Replica weights silently.** `--collections`
  defaults to `replica_f replica_g` (`:62`), and the built-in checkpoint fallback
  `outputs/echoloc_mono_fg/mono.ckpt` applied to anything. It is now guarded: with any other
  collection and no `--checkpoint`, it raises `SystemExit` (`:404-410`). Always let `BASE` pass
  the checkpoint through anyway — do not rely on the guard.
- **The README's dataset root is stale.** README says the configs default to
  `/root/jeongeon/AV-FPLoc/datasets/f3loc`; the actual yaml values are
  `/root/storage/echoloc_dataset[/<dataset>]` and
  `<repo>/data/f3loc` (verified in `configs/*.yaml`, 2026-10-13). Trust the yaml. `[문서 주장]`
  for the README value.
- **`/root/storage/echoloc_dataset` is the shared NFS mount** (`/data/changwoo` on the host),
  and the dataset tree is read-only — nothing is written to, copied from, or cached out of it.
  README measures the cold NFS read at ~460 img/s against ~7,000 img/s once warm, after one
  pass of `scripts/warm_dataset_cache.py` `[문서 주장]`. Benchmarks present under
  `echoloc_dataset/`: `replica`, `mp3d`, `s3d`, `gibson`, `zind`.
- **README's "GPUs 1-7 are used; GPU 0 is deliberately left free" is a property of the machine
  it was written on** `[문서 주장]`. Every card on a lab node may already be driven by another
  project's queue. Check occupancy and pass `--gpus` explicitly; never assume a free card.
- **Replica's F3Loc row uses a different kind of run from the other datasets'.** `BASE` names
  `outputs/echoloc_mono_fg/mono.ckpt` for Replica (trained on `replica_f` **and** `replica_g`)
  but `outputs/visual/<ds>/mono_lr3e4/mono.ckpt` for mp3d and s3d. Intentional, but it means the
  Replica visual row is not the same training protocol as the others — say so whenever the
  table is quoted.

## 6. Baselines and where the citable numbers are

Citable files: **`feasible/results/*.json`** with a `.md` twin each. Structure verified
2026-10-13 — e.g. `P_matrix_replica.json` carries `condition`, `grid`, `fix_structure`, `rule`,
`results`, `provenance`, with each backbone entry holding `dataset`, `rooms`, `n`, `vision`,
`ours`, `gain`, `ci` and per-threshold `recalls`.

Verified anchor (`P_matrix_replica.json`, `results.f3STFT`, Replica `raw_scan_open`, 3 rooms,
n = 600, 1.0 m recall): vision **0.3833** → ours **0.4483**, gain **+0.065**, CI
[0.0333, 0.0983].

`docs/tables/*.tex` and `docs/paper_tables.md` are the paper view and are `[문서 주장]` — read
the json behind a cell before quoting it.

**Superseded rows are kept, not deleted.** A `.testselected_disco.json` suffix marks results
computed from the discarded test-selected DisCo checkpoints, and the old tables live in
`outputs/analysis/archive_testselected_disco/`. Never mix those into a current table.

## 7. This repo tracks its results — the one exception

Unlike the other two research repos, **result files are in git here**: 100 tracked files under
`feasible/results/` as of 2026-10-13 (5 untracked). `.gitignore` excludes `data/` and
`outputs/` but **not** `feasible/results/`.

Whether that belongs in git is Changwoo's open call. Until he decides:

- Follow the existing pattern; do not start a parallel out-of-repo results directory.
- **Never delete or overwrite a tracked result file that a table cites.** Write a new name.
- Still never commit datasets, checkpoints, grids or `outputs/`.

## 8. Working tree and pushing

The checkout on this node is HEAD for node work, and it currently carries **uncommitted
changes** — 9 modified files plus several `.bak-pre-*` untracked, the ZInD table landing.
Commit only the files your change touches, hunk by hunk if needed; never sweep that set into an
unrelated commit.

The container has **no GitHub credentials**. Route: bundle on the node, push from the macmini
clone at `~/work/avloc-push/av_localization`. Never force-push.
