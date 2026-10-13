# AUDIT — EchoLoc CVPR 2027 experiment programme

Pass 1 audit. Status of each card is what was **executed and verified**, not what was
intended. Everything below was read from code, from each run's own logged hparams, or from
the metadata blob stored inside the rendered grids — never from a README or a `docs/*.md`
claim. Where a repo `.md` is the only source, it is marked `[문서 주장]`.

- Repo SHA at audit time: `8ba1383`
- Date: 2026-10-14
- Audit scripts and raw output: `$NODE_LOCAL1/echoloc_cvpr/` (out of repo, per the results policy)

| Card | Status | One-line verdict |
|---|---|---|
| AUD-01 | PARTIAL | Acoustic/fusion path mapped; full module inventory not yet written |
| AUD-02 | **DONE** | **Zero split leakage, scene and building level, all 5 benchmarks** |
| AUD-03 | PARTIAL | Feature definition and dimension confirmed; full formula↔code trace pending |
| AUD-04 | **DONE** | **No test-selected checkpoint in use; one row deviates in selection metric** |

---

## AUD-02 — Data, asset and split audit · DONE

**Method.** `scripts/aud02_split_audit.py` loads each benchmark's shipped
`dataset_meta.json` split, the validation rooms `scripts/val_select.py` actually uses
(`VAL_ROOMS`, `:81-94`), and the test scenes extraction actually ran on (Replica/MP3D/S3D
from `run_projection_matrix.BASE`; Gibson and ZInD from the merged grid directories the
tables were read from). Overlap is computed at **scene** level and at **building** level,
because MP3D/Gibson/ZInD name floors `<building>_f<N>` and two floors of one building are
different scenes but not independent geometry.

### Split sizes

| dataset | train | val (shipped) | test (shipped) | val used | test used |
|---|---|---|---|---|---|
| Replica | 11 | 3 | 3 | 3 | 3 |
| MP3D | 131 (66 bldg) | 16 (9) | 12 (8) | 6 (5) | 12 (8) |
| Structured3D | 400 | 50 | 250 | 10 | 30 |
| Gibson | 795 (410 bldg) | 81 (40) | 69 (39) | 6 (6) | 68 (39) |
| ZInD | 1948 (1168 bldg) | 243 (141) | 252 (141) | 8 (8) | 252 (141) |

### Overlap result

**0 overlapping pairs.** `train ∩ val`, `train ∩ test` and `val ∩ test` are all empty at
both scene and building level, for all five benchmarks, for both the shipped partition and
the subsets actually used. The splits are scene-disjoint *and* building-disjoint, which is
the stronger condition the execution principles ask for (No test leakage, §0.4).

Two numbers that look like leaks and are not:

- ZInD's shipped val and test both contain exactly **141 buildings**. Coincidence — the
  building-level intersection is 0.
- Gibson renders **68 of 69** test floors. The single missing floor is exactly
  `Sargents_f2`, and nothing is rendered that is outside the shipped test split. This
  matches the documented memory-limit exclusion. Since it is dropped from the grid, it is
  absent from the visual row and the acoustic row alike, so the pairing is preserved
  (§0.6 Matched pairing).

### Subsampling to carry into any table caption

`test used` is not `test shipped` for two benchmarks: **S3D evaluates 30 of 250 test
scenes (12 %)** and Gibson 68 of 69. ZInD, Replica and MP3D use their full test split.
S3D's 12 % subsample is the smallest evidence base of the five (n = 558 queries) and its
confidence intervals are correspondingly the widest in the headline table.

---

## AUD-04 — Checkpoint selection and test-independence audit · DONE

**The question:** was any reported checkpoint chosen by looking at test?

**Answer: no.** Every checkpoint behind the headline table was selected on a validation
metric. One row was selected on a *different* validation metric from its siblings, which is
a protocol-uniformity defect rather than a leak.

| Backbone | Selection metric | Where verified | Uniform across datasets |
|---|---|---|---|
| F3Loc mono | `loss-valid`, min, top-3 | `scripts/train_visual.py:169` (hardcoded) | yes |
| UnLoc | `loss-valid`, min, top-1 | `UnLoc/train.py:161` (hardcoded) | yes |
| DisCo-FLoc RRP | `val_action_loss` on `val_split: val` | each run's own `lightning_logs/*/hparams.yaml` | yes, all 4 |
| SemRayLoc | `loss-valid` (MP3D, S3D) | symlink target of `outputs/srl/<ds>/semantic/best.ckpt` | **no — see below** |

### The DisCo trap is real but was avoided

`DisCo-FLoc/configs/**/rrp_*.yaml` splits into two families, and the difference decides
whether a checkpoint is honest:

- `val_split: test` — `rrp_gibson`, `rrp_mp3d`, `rrp_mp3d_lr1e4`, `rrp_s3d`,
  `rrp_s3d_lr1e4`, `rrp_zind_lr1e4`, `rrp_replica_f`, `configs/paper/rrp_gibson`
- `val_split: val` — every `*_val.yaml`, plus `configs/paper/rrp_s3d`

A checkpoint from the first family is chosen on the test split. **None of the four DisCo
checkpoints in `BASE` comes from that family**: verified not from the run-directory name
(`CLAUDE.md` §3 warns that directory names lie) but from `val_split: val` in each run's own
logged `hparams.yaml`, for `rrp_replica_f_val_20260920_074725`,
`rrp_mp3d_lr1e4_val_20260921_054659`, `rrp_s3d_val_20260920_074724` and
`rrp_gibson_val_20260920_124720`. The superseded test-selected tables are kept, correctly
quarantined, as `feasible/results/VAL_<ds>_fixed_indomain.testselected_disco.{json,md}`.

### Finding: the SemRayLoc rows are not selected the same way — and S3D is the odd one out

**Corrected 2026-10-14 after reading the per-epoch metrics.** An earlier version of this
section read the selection metric off the checkpoint *filenames* and concluded that Replica
was the single deviating row. That was wrong. Filenames record which metric the
`ModelCheckpoint` was monitoring, not which epoch was ultimately linked. Reading the linked
epoch against each run's own `lightning_logs/*/metrics.csv` gives the opposite answer:

| dataset | `best.ckpt` resolves to | `acc_rays_val` best | `loss-valid` best | linked epoch is |
|---|---|---|---|---|
| Replica | `semantic_acc/…epoch=07-acc_rays_val=0.819` | **ep 07** (0.8187) | ep 00 (0.8242) | **accuracy argmax** |
| MP3D | `…epoch=02-loss-valid=0.656` | **ep 02** (0.7686) | ep 01 (0.6479) | **accuracy argmax** |
| S3D | `…epoch=30-loss-valid=0.436` | ep 32 (0.8413) | **ep 30** (0.4363) | **loss argmin** |

MP3D's link is the giveaway: it points at epoch 02 even though epoch 01 has the *lower*
`loss-valid` (0.6479 vs 0.656) and both are on disk. Epoch 02 is MP3D's `acc_rays_val`
argmax. So Replica and MP3D are both accuracy-selected and **only S3D is loss-selected**.

This matches the code's stated intent. `feasible/run_srl_pipeline.py:40-49`
(`best_checkpoint`) globs for `*-acc_rays_val=*.ckpt` **first** and only falls through to
`*-loss-valid=*.ckpt` when no accuracy-monitored checkpoint exists, and its docstring
(`:55-57`) says the checkpoint "is chosen on the validation rooms by ray accuracy, which is
what the localiser consumes". S3D got the loss fallback because its run was monitored on
loss and `save_top_k=2` kept only epochs 29 and 30 — **its accuracy-best epoch 32 is not on
disk**.

Both metrics are validation metrics, so **there is no test leakage** either way. What breaks
is `docs/headline_config.md` §1's claim that every row is picked the same way.

Why it matters: on Replica the two metrics point in **opposite directions**. `loss-valid`
is minimised at epoch 00 (0.830) and rises monotonically to 1.518 by epoch 11, while
`acc_rays_val` climbs from 0.718 to a 0.819 peak. The network overfits in cross-entropy
while getting better at the task the localiser actually uses. So "make them all
`loss-valid`" selects a barely-trained epoch-0 network, and is not obviously the right fix.

Two coherent ways to make the four SemRayLoc rows uniform:

1. **Uniform on `loss-valid`** — fully available on disk, no retraining: Replica ep 00
   (0.830), MP3D ep 01 (0.6479), S3D ep 30 (already). Changes **two** rows, and moves
   Replica to a network 10 accuracy points worse than the one it uses now.
2. **Uniform on `acc_rays_val`** — matches the code's stated intent, but S3D's ep 32
   checkpoint does not exist. Requires retraining S3D's semantic net (40 epochs) or
   re-running it with `--monitor acc_rays_val`.

**Resolved 2026-10-14 by measurement — see `EXECUTION_LOG.md`, AUD-04-FIX.** Option 1 was
run on Replica: the `loss-valid` checkpoint scores **41.3 vs 46.7** on validation recall@1 m,
the metric this protocol uses for every other choice, and drops test vision 35.0 → 29.0.
**Option 1 is rejected** — it would adopt a model the selection criterion rejects. The
coherent uniform rule (select the semantic net by validation localisation recall@1 m) is
already satisfied by Replica's epoch 07 and MP3D's epoch 02, so **neither row needs
changing**; the documentation claim was the thing that was wrong, and it has been corrected
in `docs/headline_config.md` §1. Only S3D stays unverified, needing a ≈6 h retrain with
`--monitor acc_rays_val` to produce its missing epoch-32 checkpoint.

That run also produced the more consequential result: **`ΔR@1m` is not invariant to the
backbone checkpoint** (gain +7.3 → +9.7 while vision fell 35.0 → 29.0). Gains are therefore
confounded by backbone strength and must always be quoted beside the vision column.

### Test-informed decisions already disclosed, and confirmed here

Two choices upstream of the scalars were made after test numbers existed. Both are disclosed in
`docs/headline_config.md` §3; this audit confirms the mechanism rather than discovering it:

1. The fixed structure (`centre/quantile`, three-scalar rule) was settled by room-CV on
   Replica's **test** rooms (2026-09-14, `feasible/results/H_*.json`).
2. The `INDOMAIN` source map (`val_select.py:127`) was fixed after per-source test numbers
   existed — commit `303f5fa`, verified present, dated 2026-09-15, message "One method
   everywhere: fixed structure and three-scalar rule, projection from the benchmark's own
   training rooms, scalars chosen on validation rooms, test rooms evaluated once".

These are the honest limitation to state in the paper: the three scalars are validation-
selected, but the *structure* they sit in and the *source map* that feeds them were not.
`--source indomain` also pins the projection source by design (`val_select.py:234` skips
the no-projection candidate whenever the source is fixed), so the no-projection row is an
after-the-fact ablation, never a competitor in the selection.

---

## AUD-01 / AUD-03 — what is confirmed so far

Confirmed, and the basis of the 1순위 uniformity verdict:

- **Zero dataset branching in the acoustic or fusion path.** The strings `replica`, `mp3d`,
  `s3d`, `gibson`, `zind` do not appear in `track1_core/likelihood/grid_score.py`,
  `track1_core/likelihood/mode_fusion.py` or `track1_core/modes.py`. Dataset names enter
  only in `scripts/val_select.py` (`VAL_ROOMS`, `INDOMAIN`, tag naming) and
  `scripts/extract_mode_table.py` (default collections, a Gibson checkpoint fallback and
  its guard at `:404-410`).
- **All five candidate grids are rendered with identical acoustics**, read from the
  `config` blob stored inside each merged `.npz`: `sample_rate 48000`,
  `indirect_rays 20000`, `ray_depth 50`, `source_rays 512`, `diffraction` on with
  `max_diffraction_order 10`, `layout ring`, 6 microphones at `ring_radius_m 0.05`,
  `clearance_m 0.25`, `yaw_bins 1`. Checked for Replica `acoustic_grid_v2`, MP3D, S3D,
  Gibson and ZInD (`$NODE_LOCAL2/zind_grid_all`).
- **Identical feature arguments** in all five pipelines: `--feature stft_band --nfft 256
  --hop 64` (12 `BASE` cells, plus `run_gibson_pipeline.py:102` and
  `run_zind_pipeline.py:146`).
- **The feature is defined in physical time, not sample counts.**
  `extract_mode_table.py:439-444` reads `sample_rate` from the grid's own config and
  derives `direct_guard_samples` as 2 ms and `usable_samples` as 128 ms from it. Bands are
  fixed in Hz. At 48 kHz this gives 93 frames, so the feature is
  6 mics × 3 bands × 93 frames = 1674 plus the 18 × 4 energy-split term from
  `grid_score.shape()` = **1746**, exactly the input width of the learned projection.

Still open (needs no GPU, next in queue): the full module inventory for `REPO_MAP.md`, and
the end-to-end single-query tensor trace that AUD-03's completion criterion requires.

### Correction issued to the code during this audit

`grid_score.to_config_rate`'s docstring asserted "the candidate grid stayed at 8 kHz". That
is false for every current benchmark and would mislead anyone reasoning about the time
convention; the decimation path is a no-op today. Fixed in `8ba1383`, together with a note
that `GridScoreConfig`'s defaults (`envelope`, 8 kHz, nfft 64, hop 16) are overridden at
every call site and are not the reported configuration.

---

## Per-dataset differences, classified

The 1순위 requirement is that acoustic encoding and the comparison method are identical
everywhere, while projection/scale/alignment scalars may differ. Status:

| What | Varies by dataset? | Class |
|---|---|---|
| Render physics (rate, rays, depth, diffraction, mic ring) | no | must match — **matches** |
| Feature (`stft_band`, nfft, hop, bands, guard, window, normalise, distance, quantile) | no | must match — **matches** |
| Hypotheses (K=10, NMS 1.5 m, 0.5 m disc) and score definitions | no | must match — **matches** |
| Fusion rule form, `τa = −2`, `standard` transform | no | must match — **matches** |
| `--n-poses` | 100 / 40 / 40 / 20 / 20 | query count; paired, so fair, but n spans 558–5440 |
| `--condition` | `raw_scan_open` ×3, `floorplan_closed` ×2 | the benchmark's physical nature; the furnished/matched grouping |
| `--n-rays 7 --f-w 0.5959` | S3D only | **visual** camera intrinsics, not acoustic |
| Projection source `INDOMAIN` | R / M / G / B / B | S3D and ZInD have no furnished recordings and borrow pooled R+M |
| Rule scalars `(w, s, τv)` | per (dataset, backbone) | validation-selected; explicitly allowed to differ |

**Verdict: the acoustic encoding and the comparison method are uniform across all five
benchmarks. No alignment experiments are required.** The remaining non-uniformities are
visual-side intrinsics, the benchmarks' physical nature, or the scalars that were allowed
to differ.

One visual-side asymmetry does cross a table row and is not in the allowed set: Replica's
F3Loc and SemRayLoc depth checkpoint is `outputs/echoloc_mono_fg/mono.ckpt`, trained on
`replica_f` **and** `replica_g`, while MP3D and S3D use
`outputs/visual/<ds>/mono_lr3e4/mono.ckpt`. Intentional, already noted in `CLAUDE.md` §5,
but it means the Replica visual row is not the same training protocol as the others.
Together with the SemRayLoc selection-metric finding above, **two of the four Replica rows
rest on a different protocol from their cross-dataset counterparts.**
