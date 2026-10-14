# FEASIBILITY — all 48 experiment IDs, triaged

Written 2026-10-14 at `ad55b86`. Every row is classified from **what the code actually
exposes**, enumerated from `scripts/extract_mode_table.py` and `scripts/val_select.py`
argument lists and from `GridScoreConfig` / `ModeFusionConfig` field lists — not from the
spec's wording and not from any `docs/*.md` claim.

Three categories:

- **RUN** — dispatchable today with existing flags and existing assets.
- **CODE** — the asset exists but the knob is not reachable; needs a reviewed code change.
- **ASSET** — no code problem, the data does not exist; needs rendering, training, or
  physical capture.

## What the CLI actually exposes

`extract_mode_table.py`: `--condition --grid-dir --backbone --checkpoint --depth-ckpt
--depth-arch --semdesdf-dir --sem-weight --window-ms --feature {envelope,stft_band} --nfft
--hop --tag --nms-radius-m --n-modes --local-radius-m --peak-rank --legacy-weight --topk
--n-poses --n-rays --f-w --degrade --degrade-seed --acoustic-projection
--align-ring {none,vis,gt} --orn-slice --out-dir --gpu`

`val_select.py`: `--dataset --backbones --boot --simplicity-margin --out --source
{val,indomain,B,T} --variants --fixed`

**Reachable by flag, which many spec items assumed was missing:** `--degrade/--degrade-seed`
(image corruptions), `--align-ring` (array/heading alignment), `--nms-radius-m/--n-modes/
--topk/--local-radius-m` (candidate-set sweeps), `--condition` (matched vs furnished),
`--sem-weight` (SemRayLoc's semantic weight).

**Not reachable:** every `GridScoreConfig` field except `feature`, `window_ms`, `nfft`, `hop`
and the rate-derived ones. In particular **`distance` (`symmetric` / `one_sided`),
`normalise` (`sum` / `peak` / `quantile`), `bands`, `first_frame`, `last_frame` and
`shape_feature` are set at the call site and never passed**, so everything has run on
`distance='symmetric'`, `normalise='sum'`, `shape_feature=True`, 3 bands, full window.

## Triage

| ID | Class | Why / what is needed |
|---|---|---|
| AUD-01 | RUN | partially done; module inventory outstanding |
| AUD-02 | **DONE** | zero split leakage, scene and building level |
| AUD-03 | RUN | feature definition and dimension confirmed; single-query trace outstanding |
| AUD-04 | **DONE** | no test-selected checkpoint; SemRayLoc selection metric non-uniform |
| REP-01 | **DONE** | pairing verified; metrics internally consistent |
| REP-02 | **DONE** | 16/16 Replica+S3D extractions, per-query identical |
| REP-03 | **16/19 DONE** | Replica 4, S3D 4, Gibson 3, MP3D SemRayLoc 1 bit-exact; ZInD 6/8 done, SemRayLoc pair running; MP3D's other 3 not yet re-run |
| REP-04 | RUN | render configs read from each grid's own blob; sanity tests on 5–10 samples outstanding |
| BAS-01 | **CODE** | raw-waveform normalised cross-correlation is implemented nowhere |
| BAS-02 | **DONE (as ablation)** | `--feature stft_band` with and without `--acoustic-projection` **is** the W=I vs learned-W contrast; cosine-vs-L2 is not separable, `distance` is not exposed |
| BAS-03 | **RUN** | `--feature envelope --window-ms` — 9/10 cells extracted |
| BAS-04 | **CODE** | `GridScoreConfig.distance='one_sided'` exists but `extract_mode_table` never passes it and no flag exposes it. One-line plumb + flag. **Highest-value code change outstanding** |
| BAS-05 | **DONE (as ablation)** | the projected vs identity arms already measure this on all 19 cells |
| BAS-06 | RUN (a); ASSET (b) | conditional acoustic argmax within K is derivable from existing per-query columns; a global acoustic lookup needs a full-grid cache that does not exist |
| FUS-01 | **RUN** | CPU only. `ModeFusionConfig` carries `weight`, `sigmoid_scale`, `tau`, `ac_transform`; a constant-λ arm is a large `sigmoid_scale` |
| FUS-02 | **CODE (small)** | entropy gate alone needs the separability factor disabled; `choose()` applies "both gates as soft factors" (`mode_fusion.py:193`) with no switch |
| FUS-03 | **CODE (small)** | separability gate alone, same reason |
| FUS-04 | **RUN (partly)** | no-gate vs both-gates is reachable by parameter choice; isolating each gate needs FUS-02/03's switch |
| FUS-05 | RUN | abstention curve is derivable on CPU from existing acoustic-margin columns |
| GAP-01 | **RUNNING NOW** | **assets verified present** — `replica/rir/replica_{f,g}/` holds both `floorplan_closed/` and `raw_scan_open/` over the same 8 scenes, 300 poses each, so queries pair exactly |
| GAP-02 | **ASSET** | needs empty/sparse/medium/dense furniture variants of one scene. The dataset ships exactly two conditions, not a clutter ladder. Re-rendering needs the furnished meshes plus a furniture-removal pipeline that does not exist |
| GAP-03 | **ASSET** | material/absorption perturbation needs re-rendering query RIRs with altered material coefficients; `soundspaces_render.py` exposes no material flags |
| GAP-04 | **RUN after BAS-04** | compares matchers under GAP-01's conditions; needs the one-sided matcher first |
| GAP-05 | **ASSET** | perturbed floorplan geometry must be re-rendered; no wall-displacement tooling |
| GAP-06 | **CODE** | early/late windows are `first_frame`/`last_frame`, present in the config, not exposed |
| HYP-01 | **RUN** | `--n-modes`, `--nms-radius-m`, `--topk`, `--local-radius-m` all exposed |
| HYP-02 | **RUN** | CPU only; ambiguity columns (`mode_entropy`, `mode_log_ratio`, `n_modes_ge_*`) are already in every queries table |
| HYP-03 | **RUN** | CPU only; `gt_in_top2_modes` and the margin columns are present |
| HYP-04 | **RUN** | CPU only; W→C/C→W from paired `e_vis`/`e_fused` |
| HYP-05 | **CODE** | needs a camera-FoV geometry descriptor (`D_visible`, `D_out`); no such code |
| HYP-06 | **CODE (small)** | audio swap/shuffle counterfactual needs a query-to-grid remap hook; nothing exposes it |
| ROB-01 | **CODE** | SNR sweep needs noise injection into query RIRs; no flag |
| ROB-02 | **CODE** | time-shift sweep needs a guard/offset flag; `direct_guard_samples` is rate-derived, not exposed |
| ROB-03 | **ASSET** | receiver/array geometry errors need re-rendering |
| ROB-04 | **RUN** | `--degrade` and `--degrade-seed` are exposed |
| CHN-01 | **CODE** | channel subsetting needs a mic-selection flag; grids are 6-channel, so the data is there and only the selection is missing |
| CHN-02 | **RUN** | `--align-ring {none,vis,gt}` is exposed; `gt` must be labelled an oracle |
| GEN-01 | **RUN** | `--acoustic-projection` takes any `.npz`; cross-dataset transfer is a matrix over existing W files |
| GEN-02 | **ASSET** | map-resolution sweep needs grids re-rendered at other cell sizes |
| GEN-03 | **ASSET** | same as GAP-05 |
| SYS-01 | **RUN** | latency profiling needs no new assets |
| SYS-02 | **RUN (from logs)** | offline render cost is in the render logs and each grid's stored config |
| SYS-03 | **RUN** | follows HYP-01's sweep |
| SYS-04 | ASSET | end-to-end sensing needs physical capture |
| REAL-01 | **ASSET** | no real measurements exist |
| REAL-02 | **ASSET** | depends on REAL-01 |
| NOV-01 | **CODE** | the proposed asymmetric score is not implemented; depends on BAS-04 |
| NOV-02 | RUN | reuses FUS-05 |

## Counts

| class | n | notes |
|---|---|---|
| DONE | 7 | AUD-02, AUD-04, REP-01, REP-02, BAS-02, BAS-05, (REP-03 partial) |
| RUN today | 17 | the bulk is CPU analysis on tables already extracted |
| CODE | 13 | **BAS-04 is the one that blocks a P0 novelty claim** |
| ASSET | 11 | GAP-02/03/05, GEN-02/03, ROB-03, REAL-*, SYS-04 — mostly re-rendering |

## The two findings that matter most from this triage

**1. A large part of the remaining P0 work is CPU, not GPU.** FUS-01/05 and HYP-02/03/04 are
analyses over per-query columns that already exist in every `queries_*.csv`. They were never
blocked on cards; they were blocked on nobody writing the analysis. That is the main reason
the contribution experiments had not started.

**2. The mismatch-robustness story splits cleanly.** GAP-01, the *same-scene* control the
title depends on, has its assets and is running now. GAP-02/03/05 — the clutter ladder,
material perturbation and geometry-error sweeps — are all **ASSET** blocked, because the
dataset ships exactly two acoustic conditions and `soundspaces_render.py` exposes no material
or wall-perturbation flags. So the paper can support "matched vs furnished on the same scene"
but **cannot currently support a dose-response claim about mismatch severity.** That is a
scope limit to state, not a gap to paper over.

## Re-run commands for the blocked classes

CODE items need the change first; the reviewed one-liner for BAS-04 is to thread
`distance` from a new `--distance {symmetric,one_sided}` flag into the `GridScoreConfig`
built at `extract_mode_table.py:439-444`, defaulting to `symmetric` so every existing result
is unchanged.

ASSET items, to be explicit about what is missing rather than implying it could be run:

    GAP-02  needs: furnished meshes + furniture-removal/placement pipeline, then
            re-render query RIRs per clutter level. Not present.
    GAP-03  needs: material-coefficient flags in soundspaces_render.py, then re-render.
    GAP-05/GEN-03 needs: wall/door perturbation tooling over the floorplan proxy.
    GEN-02  needs: grids re-rendered at 0.1/0.2/0.4 m cell sizes.
    ROB-03  needs: re-render with perturbed receiver/array geometry.
    REAL-01/02, SYS-04 needs: physical measurement campaign.
