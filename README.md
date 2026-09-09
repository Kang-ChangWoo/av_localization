# AV-FPLoc

Research scaffold for active acoustic-visual floorplan localization under visual degradation.

## Status

The repository includes a collaborator-facing Gibson SoundSpaces RIR and acoustic-likelihood reproduction path, and a trainable reproduction of the F3Loc visual observation models. Fusion remains research work in progress.

## Research goal

Estimate a single-observation 2D floorplan pose posterior by combining visual structural cues with active acoustic geometry cues. The primary question is whether acoustic geometry reduces the structural ambiguity that remains when visual observations are low-information or visually degraded.

## Repository layout

```text
track1_core/    Implementation package and module ownership boundaries.
  datasets/     Dataset adapters (Gibson Floorplan Localization Dataset).
  models/       Trainable LightningModules over the visual observation networks.
configs/        Reproducible experiment configuration.
scripts/        Training, evaluation, and asset-staging entry points.
third_party/    Vendored F3Loc baseline, pinned to one upstream revision.
docs/           Architecture, tensor contracts, baseline facts, and decisions.
```

---

# Visual baseline reproduction (F3Loc)

## Reference

> **F³Loc: Fusion and Filtering for Floorplan Localization** — CVPR 2024 (highlight)
> Changan Chen, Rui Wang, Christoph Vogel, Marc Pollefeys
> [Paper](https://arxiv.org/pdf/2403.03370.pdf) · [arXiv:2403.03370](https://arxiv.org/abs/2403.03370) · [Project page](https://felix-ch.github.io/f3loc-page/) · [Code](https://github.com/felix-ch/f3loc)

```bibtex
@inproceedings{chen2024f3loc,
  title={F$^3$Loc: Fusion and Filtering for Floorplan Localization},
  author={Chen, Changan and Wang, Rui and Vogel, Christoph and Pollefeys, Marc},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

The paper's observation module predicts a 1D structural depth profile per image, converts it into 11 rays, and matches those rays against a precomputed directional range field (DESDF, `(H, W, 36)` at 0.1 m/cell) over all candidate positions and yaw bins.

## Why this exists

Upstream F3Loc released evaluation code, Lightning wrappers, and checkpoints, but **not a training script** — its README states the training pipeline was Azure ML specific. Three concrete gaps block `trainer.fit()` on the released code:

1. `depth_net_pl.training_step` reads `batch["img"]` / `batch["gt_rays"]`, while `GridSeqDataset` emits `ref_img` / `ref_depth`.
2. `mv_depth_net.forward` indexes `x["ref_mask"]` and `x["src_mask"]` unconditionally, but the dataset omits those keys unless roll/pitch augmentation is on — and `None` cannot pass through `default_collate`.
3. `comp_d_net` takes already-trained `mv_net` and `mono_net` objects, so the three networks must be trained in sequence.

`track1_core/datasets` and `track1_core/models` own that adapter layer; `third_party/f3loc` keeps upstream's network and localization code byte-identical to the pinned revision recorded in `third_party/f3loc/PINNED_REVISION`.

## Networks

| Stage | Paper name | Module | Target | Trainable params |
| --- | --- | --- | --- | --- |
| `mono` | Ours_s | `MonoDepthModule` (`depth_net`) | `depth40` | 25.9 M |
| `mv` | Ours_m | `MVDepthModule` (`mv_depth_net`) | `depth160` | 26.4 M |
| `comp` | Ours_f | `CompDepthModule` (`comp_d_net`) | `depth160` | 5.3 K (selector only) |

`comp_d_net` freezes both observation networks in its constructor, so the complementary stage fits only the selector MLP that weights the two probability volumes from the relative poses and the two predicted depths. The frozen networks are additionally pinned to `eval()` so their BatchNorm statistics cannot drift.

## Setup

```bash
conda activate f3loc          # py3.8, torch 2.4.1+cu124, lightning 2.4.0
cd av_localization
```

The dataset root defaults to `/root/jeongeon/AV-FPLoc/datasets/f3loc` in the configs and is **read-only** — nothing is written to, copied from, or cached out of the dataset tree.

## Reproduction

### 0. Warm the page cache (recommended)

The dataset is on NFS, which delivers ~460 img/s cold across 64 readers, while 7-GPU DDP needs ~1,300 img/s for the multi-view stage. One read pass puts the ~88 GB of rgb into the page cache (the machine has ~260 GB free), after which the same files read at ~7,000 img/s. Nothing is copied to local disk.

```bash
python scripts/warm_dataset_cache.py --datasets gibson_f gibson_g --splits train val
```

### 1–3. Train, in order

```bash
python scripts/train_visual.py --net mono --config configs/visual_mono.yaml
python scripts/train_visual.py --net mv   --config configs/visual_mv.yaml
python scripts/train_visual.py --net comp --config configs/visual_comp.yaml \
    --mono-ckpt outputs/mono/mono.ckpt --mv-ckpt outputs/mv/mv.ckpt
```

Each run writes `outputs/<run_name>/checkpoints/` (top-3 by `loss-valid`, plus `last.ckpt`), a CSV metric log, and a stable `outputs/<run_name>/<net>.ckpt` alias pointing at the best epoch.

**GPUs 1–7 are used; GPU 0 is deliberately left free.** Override with `--gpus`.

### 4. Evaluate

```bash
python scripts/eval_visual.py --net mono --dataset gibson_f --ckpt outputs/mono/mono.ckpt
python scripts/eval_visual.py --net mv   --dataset gibson_g --ckpt outputs/mv/mv.ckpt
python scripts/eval_visual.py --net comp --dataset gibson_g --ckpt outputs/comp/comp.ckpt
python scripts/eval_visual.py --net comp_s --dataset gibson_g \
    --mono-ckpt outputs/mono/mono.ckpt --mv-ckpt outputs/mv/mv.ckpt
```

`eval_visual.py` runs the same localization pipeline as upstream `eval_observation.py`, batched on the GPU instead of one sample at a time.

### Or run everything as one queue

`scripts/run_queue.py` schedules the reproduction runs and a hyperparameter sweep over the same GPU pool: the three reproduction jobs take all seven GPUs in turn, then the sweep jobs run seven-wide on one GPU each. Every job is evaluated on both collections when it finishes.

```bash
python scripts/run_queue.py --jobs configs/queue_visual.yaml    # run
python scripts/run_queue.py --status                            # inspect
```

Progress and per-job metrics are written continuously to `outputs/queue_status.json`, so the queue can be left unattended and inspected later.

## Reproduction targets

Recall on the 9 held-out test scenes, from the paper's observation-module tables:

| Dataset | Method | 0.1 m | 0.5 m | 1 m | 1 m / 30° |
| --- | --- | --- | --- | --- | --- |
| Gibson(f) | Ours_s (`mono`) | 4.7% | 28.6% | 36.6% | 35.1% |
| Gibson(f) | Ours_m (`mv`) | 13.2% | 40.9% | 45.2% | 43.7% |
| Gibson(g) | Ours_s (`mono`) | 4.3% | 26.7% | 33.7% | 32.3% |
| Gibson(g) | Ours_m (`mv`) | 9.3% | 27.0% | 31.0% | 29.2% |
| Gibson(g) | Ours_t (`comp_s`) | 10.5% | 34.3% | 39.6% | 38.0% |
| Gibson(g) | Ours_f (`comp`) | 12.2% | 39.4% | 44.5% | 43.2% |

## Hyperparameter provenance

The paper does not publish its training hyperparameters — the implementation details live in supplementary material that is not distributed with the arXiv or CVPR PDF. Every value is therefore labelled by where it comes from.

| Value | Setting | Source |
| --- | --- | --- |
| `D=128`, `d_min=0.1`, `d_max=15.0`, `d_hyp=-0.2`, `F_W=3/8`, `L=3` | all nets | upstream `eval_observation.py` |
| mono loss = L1 + λ·(1 − cosine); mv/comp loss = L1 | — | paper ("a shape loss ... except for the monocular network") |
| λ (shape loss weight) | 1.0 | **AV-FPLoc choice** — paper gives the formula, not the value |
| optimizer | Adam | upstream `configure_optimizers` |
| learning rate | mono/mv 1e-4, comp 5e-4, warmup + cosine decay | **AV-FPLoc choice** — upstream's 1e-3 default is tuned for an unpublished batch size |
| per-GPU batch size | mono 24, mv 12, comp 12 | **AV-FPLoc choice** — fits 24 GB at bf16 |
| epochs | mono 20, mv 30, comp 10 | **AV-FPLoc choice** |
| precision | `bf16-mixed` | **AV-FPLoc choice** |
| roll/pitch augmentation | off | **AV-FPLoc choice** — the Gibson collections are upright and evaluation passes no gravity-alignment mask, so training matches the evaluation condition |
| training collections | `gibson_f` + `gibson_g` | **AV-FPLoc choice** — the paper reports one checkpoint evaluated on both |
| mono sample definition | reference frame only (`views: ref`) | upstream `GridSeqDataset` semantics — one sample per `L+1` chunk |

`d_hyp` deserves a note: `mv_depth_net_pl` defaults to `1.0`, but upstream's evaluation constructs the multi-view net with `-0.2`. Training uses `-0.2` so the depth-hypothesis spacing matches inference.

The monocular sample definition deserves one too. `depth40` carries one ground-truth row per frame, so each of the `L+1` views in a chunk is a legitimate monocular training sample — `GridSeqDataset` simply never exposes the `L` source views. Training on all of them (`data.views=all`) quadruples supervision and measurably overshoots the published recall, so the default stays at `ref`, which matches the baseline's own sample definition, and `views=all` is carried as the `mono_views_all` sweep entry instead. See "Measured deviations" below.

## Measured deviations

| Observation | Setting | Result |
| --- | --- | --- |
| `views=all`, epoch 2 of 20, gibson_f | 4× monocular supervision | 1 m recall 44.4% vs the paper's 36.6% — above target, hence not the default |

---

## Data policy

Datasets, floorplan caches, checkpoints, generated figures, secrets, and machine-specific paths stay out of version control. The F3Loc baseline is the one deliberate exception to the "no vendored external source" rule — see the 2026-09-07 entry in [docs/decision_log.md](docs/decision_log.md).

## Design-first workflow

Use the documents in `docs/` as the source of truth for research design. Record accepted design choices in [docs/decision_log.md](docs/decision_log.md) before implementation changes are made in `track1_core/`.

## Gibson RIR Reproduction

With the shared licensed assets and an existing SoundSpaces environment, run the read-only preflight:

```bash
export AVFPLOC_DATA_ROOT=/root/jeongeon/AV-FPLoc
python scripts/verify_collaborator_env.py --scene Springhill
```

The full workflow is documented in [docs/collaborator_gibson_rir.md](docs/collaborator_gibson_rir.md).
