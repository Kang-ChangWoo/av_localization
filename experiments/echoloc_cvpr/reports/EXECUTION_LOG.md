# EXECUTION LOG

One card per executed experiment, in the format of §16 B of the programme spec. Only
experiments that actually ran appear here.

---

## GAP-01 — Same-scene matched proxy vs furnished query, Replica · **DONE**

- **Status:** DONE (Replica). MP3D extracting, Gibson queued.
- **Verdict:** **The furniture mismatch is severe, and it is not survived by the matcher.**
  Acoustic-only localisation collapses from **89.7 % to 36.0 % recall@1 m — a 53.7 point
  drop** — when the only thing that changes is whether the query was recorded in the
  furnished room or on the floorplan proxy. **GATE-D therefore fails as stated: the paper
  cannot claim the matcher is robust to furnished-to-proxy mismatch.** What it can claim is
  that fusion degrades gracefully — after fusion the same gap is only 4–13 points.
- **Hypothesis:** acoustic verification should lose accuracy when the query carries furniture
  the floorplan-derived reference cannot know about. Magnitude unknown before the run.
- **Changed variable:** exactly one — `--condition raw_scan_open` → `floorplan_closed`,
  which selects the query RIR set.
- **Fixed controls:** same scenes (`office_4 apartment_2 frl_apartment_5`), same 300 poses
  per scene, **the same candidate grid `outputs/acoustic_grid_v2`** (always rendered from the
  floorplan, so the reference side never moves), same checkpoints, same projection R, same
  `stft_band` feature. Asserted in the driver that the condition swapped and the grid did not.
- **Control that passed:** vision error is **identical on all 600 queries in all four rows**
  (600/600, `|Δe_vis| < 1e-9`). The visual branch never sees audio, so this confirms the two
  arms differ only in query acoustics and the queries pair exactly.
- **Command & hardware:** `gap01.py`, cards 4–7, 07:11→07:16Z, 8 cells (4 backbones ×
  projected/identity). SHA `ad55b86`.

| backbone | acoustic-only `e_ac` | reranked `e_rerank` | fused `e_fused` |
|---|---|---|---|
| | matched / furnished | matched / furnished | matched / furnished |
| F3Loc mono | 89.7 / 36.0 (**+53.7**) | 57.5 / 47.8 (+9.7) | 56.8 / 48.5 (+8.3) |
| UnLoc | 89.7 / 36.0 (**+53.7**) | 77.2 / 58.3 (+18.8) | 73.0 / 60.0 (+13.0) |
| DisCo RRP | 89.7 / 36.0 (**+53.7**) | 65.3 / 52.2 (+13.2) | 63.0 / 53.3 (+9.7) |
| SemRayLoc | 89.7 / 36.0 (**+53.7**) | 47.2 / 40.8 (+6.3) | 46.2 / 42.0 (+4.2) |

n = 600 per cell. `e_ac` is **identical across all four backbones by construction** —
acoustic-only localisation never consults the visual branch — which is a second control: it
confirms the acoustic field and the grid are the same in every row, and that the 53.7-point
drop is a property of the acoustics alone, not of any backbone.

### What this supports

1. **The acoustic branch carries real pose information.** 89.7 % standalone recall@1 m in the
   matched condition, against the best visual backbone's 50.7 %, is not a weak signal — on
   matched geometry acoustics alone beats vision alone decisively.
2. **It is the direct causal evidence for the matched/furnished grouping.** The headline table
   splits into matched benchmarks gaining +22 to +28 and furnished gaining +5 to +14. Until
   now that split rested on a cross-dataset comparison, which the spec explicitly forbids
   using as mismatch evidence (section 7). This is the same-scene control, and it reproduces
   the split from one changed variable.
3. **Fusion is doing real protective work.** The acoustic branch loses 53.7 points but the
   fused system loses only 4–13, so the gate is absorbing most of the damage rather than
   propagating it.

### What it does NOT support

- **The title's mismatch-robustness claim, as written.** A matcher that drops 53.7 points is
  not robust to the mismatch; the *system* tolerates it because vision dominates when
  acoustics get unreliable. Those are different claims and the paper currently makes the
  stronger one. Per GATE-D, "geometry-mismatch robust" should come out of the title and
  abstract unless a mismatch-aware matcher is shown to close part of this gap (BAS-04/GAP-04,
  both blocked on the one-sided flag).
- **Any dose-response statement.** This is two conditions, not a severity ladder. GAP-02/03
  would be needed and both are asset-blocked.

### Limitation

The `e_ac`/`e_rerank`/`e_fused` columns are computed at the extraction-time default fusion
scalars, not the validation-selected ones, so the fused figures here are not the headline
protocol numbers and should not be quoted as such. The `e_ac` contrast is unaffected — an
acoustic-only argmax takes no fusion scalar — and it carries the finding.

### Next decision

Extend to MP3D (running) and Gibson (queued): both ship `floorplan_closed` **and**
`raw_scan_open` over the same scenes, which was verified before dispatch. Three furnished
benchmarks with the same control turns this from one measurement into a pattern.

---

## REP-03-GIBSON — Gibson reproduction, 3/3 cells bit-identical

- **Status:** DONE
- **Verdict:** **All three Gibson cells reproduce bit-exactly**, n = 5440 each.
- **Command & hardware:** `wave2.py`, one cell per card, **cards 2–7** (identity and
  projected for three backbones), 23:37Z → 02:45Z, ~3 h 8 min. Commands built by importing
  `run_gibson_pipeline.base()` rather than transcribing, so the checkpoint globs are the
  pipeline's own.
- **SHA:** `ee3e417`

| backbone | vision | ours | gain | 95% CI (paired) | n | reproduces HEAD |
|---|---|---|---|---|---|---|
| F3Loc mono | 47.0 | 59.7 | +12.7 | [+11.5, +13.9] | 5440 | bit-identical |
| UnLoc | 51.7 | 64.0 | +12.3 | [+11.2, +13.4] | 5440 | bit-identical |
| DisCo RRP | 38.5 | 52.1 | +13.5 | [+12.3, +14.7] | 5440 | bit-identical |

Gibson's own `ckpt()` resolves UnLoc and DisCo by glob, so reproducing these numbers also
confirms the glob still selects the same weights it did when the row was made.

**The no-projection ablation is where Gibson's gain comes from**, and it reproduced too:
F3Loc +2.5, UnLoc +3.8, DisCo +3.0 without the projection against +12.7/+12.3/+13.5 with it.
On Gibson the learned projection supplies roughly three quarters of the gain — the largest
such share of any benchmark, and a point the paper should make explicitly rather than leave
in an ablation table.

Running cell count: **12 of 19 headline cells** now independently verified (Replica 4,
S3D 4, Gibson 3, MP3D SemRayLoc 1). ZInD's 8 are extracting.

---

## REP-02 / REP-03 — Replica and S3D reproduction sweep, 8/8 headline cells bit-identical

- **Status:** DONE for the headline numbers; **3 ablation cells NOT reproduced** (see the
  correction below). **GATE-A is satisfied for Replica and Structured3D.**
- **Verdict:** **All 8 headline cells reproduce bit-exactly.** Re-extracted from scratch and
  re-selected on the validation rooms, they reproduce the committed vision, fused, gain,
  median *and* the validation-chosen scalars with no deviation.

> **Correction (2026-10-14 12:0x KST).** An earlier version of this card said "sixteen
> extractions, every cell reproduces". **13 of 16 extractions succeeded, not 16.** Three
> identity-table extractions — Replica F3Loc, DisCo and SemRayLoc — failed with
> `error: argument --tag: expected one argument`, and I did not check the exit codes before
> reporting. Cause and consequence are below. The headline claim is unaffected, because the
> selected row's vision/ours/gain/median come from the **projected** table and all eight of
> those succeeded; what was not independently reproduced is the **"no projection" ablation
> row** for those three cells, which silently fell back to the repo's own tables through the
> sandbox symlink. Those three ablation figures should not be quoted as re-verified.
- **Changed variable:** none. Determinism check.
- **Command & hardware:** `rep03_sweep.py`, a refilling card pool, one cell per card on
  **cards 2–7** of n1, 06:36→07:17 KST. Selection CPU-only in a real-file sandbox.
- **SHA:** `3f4ad9b` extraction, collected at `2e59a9f`
- **Controls:** validation tables reused from the repo read-only, so only the test
  extraction varies. Nothing in the repo written — scratch `--out-dir` throughout.

| dataset | backbone | vision | ours | gain | 95% CI (paired) | median | reproduces HEAD |
|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | 38.3 | 48.8 | +10.5 | [+7.0, +14.0] | 1.07 | bit-identical |
| Replica | UnLoc | 50.7 | 60.0 | +9.3 | [+6.3, +12.5] | 0.39 | bit-identical |
| Replica | DisCo RRP | 37.0 | 51.3 | +14.3 | [+10.7, +18.0] | 0.87 | bit-identical |
| Replica | SemRayLoc | 35.0 | 42.3 | +7.3 | [+4.0, +10.5] | 1.47 | bit-identical |
| S3D | F3Loc mono | 24.2 | 48.6 | +24.4 | [+20.3, +28.7] | 1.11 | bit-identical |
| S3D | UnLoc | 30.1 | 57.9 | +27.8 | [+23.7, +31.9] | 0.58 | bit-identical |
| S3D | DisCo RRP | 22.6 | 45.7 | +23.1 | [+19.2, +27.1] | 1.33 | bit-identical |
| S3D | SemRayLoc | 32.6 | 55.4 | +22.8 | [+18.8, +26.7] | 0.80 | bit-identical |

n = 600 per Replica cell, 558 per S3D cell.

### What it supports

- **The headline numbers for these eight cells are real and regenerable.** Together with
  REP-03-MP3D-SRL (also bit-exact) that is **nine of nineteen** headline cells verified by
  independent re-extraction. Gibson and ZInD are running now.
- **No hidden run-to-run state.** Reproducing the validation-selected scalars, not just the
  final recall, means the whole selection path is deterministic.

### What it does NOT support

- It does **not** validate the protocol, only its reproducibility. The test-informed
  structure and source-map choices recorded in `AUDIT.md` are untouched by this and remain
  the honest limitation.
- S3D's figures still rest on 30 of 250 test scenes (n = 558), the thinnest evidence base of
  the five benchmarks, which is why its CIs are the widest here.

### Two defects found by the failure, both worth keeping

**1. Three legacy identity tags are unreachable through `--tag`.** `extract_mode_table.py`
always names its output `queries_<cond>_<PREFIX[backbone]><tag>.csv`, so the only reachable
names are `PREFIX`-prefixed. But `IDENT` holds legacy names from an older scheme that are
**not** extensions of `PREFIX`:

| tag | `PREFIX` | reachable by `--tag`? |
|---|---|---|
| `unlocSTFT` | `unloc` | yes, suffix `STFT` |
| `semrayloc` | `semrayloc` | yes, suffix is legitimately empty — must be quoted `--tag ""` |
| `f3STFT` | `f3loc_mono` | **no** — not a `f3loc_mono*` name at all |
| `discoID` | `disco_rrp` | **no** — not a `disco_rrp*` name at all |

`run_projection_matrix.py` never hits this because it only *extracts* under `tag_of()`, which
always returns `PREFIX+suffix`; the `IDENT` legacy names are only ever *read*. So those two
tables cannot be regenerated under their own names at all — they are **unreproducible by
name**, and re-verifying them requires extracting to a new name and comparing content.

**2. My queue runner built `--tag ` with an empty argument.** Deriving the suffix as
`tag[len(PREFIX):]` yields `""` whenever the tag is not longer than the prefix, and an
unquoted empty shell word makes argparse consume the next flag. Fixed by quoting the value.
The deeper lesson: **the runner recorded `FAILED:rc2` in the index and I reported the sweep
without reading the exit codes.** A per-cell status column is worthless unless it is checked
before the result is believed.

### Artifact paths

`$NODE_LOCAL1/echoloc_cvpr/rep03_sweep/` — `analysis/` (13 table sets),
`VAL_{replica,s3d}_rerun{,_srl}.{md,json}`, `logs/`, `farm/`. Append-only index at
`$NODE_LOCAL1/echoloc_cvpr/RUN_INDEX.tsv`; **one-row-per-run current state** at
`STATUS.tsv`, generated by `status.py`, which also promotes a stale `RUNNING` row to
`DONE(marker)` when a completion marker exists on disk.

---

## REP-03-MP3D-SRL — MP3D SemRayLoc re-extraction, bit-exact reproduction

- **Status:** DONE
- **Verdict:** **The extraction → selection pipeline reproduces bit-exactly.** An independent
  re-run of the MP3D SemRayLoc cell matched the archived batch-8 selection to ten decimal
  places on every reported quantity. This closes D-02 and is the first verified REP-03 cell.
- **Hypothesis:** the committed MP3D SemRayLoc aggregate can be regenerated from scratch, and
  the missing per-query tables restored.
- **Changed variable:** none — same checkpoint (`outputs/srl/mp3d/semantic/best.ckpt`), same
  everything. This is a determinism check, not a comparison.
- **Command & hardware:** `rep03_run.sh`, four independent cells one per card, **cards
  2/3/4/6** on n1, 16:27→18:28 UTC (validation pair finished at 17:21, test pair at 18:28).
  Selection in a real-file sandbox.
- **n:** 960 test queries (12 rooms), 960 validation queries (6 rooms)
- **SHA / ckpt md5:** `3f4ad9b` / `ca9d658a6a8b`

| | n | vision | ours | gain | scalars (w, s, τv) | median |
|---|---|---|---|---|---|---|
| archived batch-8 selection | 960 | 0.228125 | 0.286458 | +0.058333 | 2, 0.02, 0.05 | 4.6958600253 |
| **this re-run** | 960 | **0.228125** | **0.286458** | **+0.058333** | **2, 0.02, 0.05** | **4.6958600253** |

Identical, including the validation-selected scalars — so the selection re-derived the same
configuration from re-extracted tables, not just the same final recall.

### What it supports

- **Determinism.** The pipeline is bit-reproducible end to end on a fixed checkpoint. That
  retires the worry that the committed numbers depend on unrecorded run-to-run state, and it
  is the evidence `TEST-11`/`TEST-12` ask for.
- **D-02 closed.** The per-query tables exist again, so this cell can now carry paired CIs,
  scene-clustered bootstrap, and the W→C/C→W taxonomy.

### What it does NOT support — read this before quoting the numbers

**22.8 → 28.6 is no longer the current MP3D SemRayLoc row.** While this run was in flight,
commit `90fb15a` (another session, not this one) adopted the batch-16 retrain for that cell,
and HEAD now reports **24.9 → 31.8, +6.9**. `best.ckpt` on disk still points at the batch-8
`epoch=02-loss-valid=0.656`, and the adopted row was produced by passing the batch-16
checkpoint explicitly — so **the symlink and the committed row intentionally disagree**, and
anything resolving `best.ckpt` reproduces the superseded number. That is exactly what this
run did. Its value is the determinism result, not the figures.

### Operational hazard found

**Two agents were editing one working tree concurrently.** Within this session,
`feasible/results/VAL_replica_fixed_indomain.json` and
`VAL_mp3d_fixed_indomain_srl.json` both changed underneath an in-progress analysis, and
`git merge --ff-only` silently imported the other session's commits. Two of my findings
(D-01, and the first read of this cell) were distorted by comparing a mid-flight working tree
against a committed table.

Mitigations, now applied here: compare with `git show HEAD:<path>` and never a bare
`open()`; re-read any result file immediately before quoting it; and treat the node checkout
as shared state rather than private scratch.

---

## AUD-04-FIX — Replica SemRayLoc under a loss-valid-selected semantic checkpoint

- **Status:** DONE
- **Verdict:** The uniformity defect is real, but switching to `loss-valid` is the **wrong
  fix** — it selects a checkpoint the protocol's own validation metric rejects. **Keep the
  current checkpoints; correct the documentation instead.** Separately, this run produced a
  finding that matters more than the defect: **the paired gain is not invariant to backbone
  strength.**
- **Hypothesis (pre-registered before the run):** Replica's semantic checkpoint is the
  `acc_rays_val` argmax (epoch 07, acc 0.8187) while S3D's is its `loss-valid` argmin.
  Re-selecting Replica on `loss-valid` (epoch 00, on disk) makes the rows comparable. Since
  `loss-valid` is minimised at epoch 00 and `acc_rays_val` peaks at epoch 07, the epoch-00
  network should be clearly worse, so **vision recall should fall**.
- **Code path & SHA:** `scripts/extract_mode_table.py`, `scripts/val_select.py` at `62ba5fa`
- **Changed variable:** the semantic-ray checkpoint, one file:
  `outputs/srl/replica/semantic/best.ckpt` → `semantic_acc/…epoch=07-acc_rays_val=0.819`
  replaced by `semantic/…epoch=00-loss-valid=0.830`
- **Fixed controls:** depth checkpoint (`outputs/echoloc_mono_fg/mono.ckpt`), semantic DESDF
  (`build_semantic_desdf.py` takes no checkpoint, so the reference side is untouched and was
  reused), grid `acoustic_grid_v2`, projection `acoustic_projection.npz` (R), feature
  `stft_band` nfft 256 hop 64, `--n-poses 100`, test rooms `office_4 apartment_2
  frl_apartment_5`, validation rooms `apartment_1 frl_apartment_4 office_3`, K/NMS, the
  three-scalar rule form. Scalars re-chosen on the validation rooms, as the protocol requires.
- **n:** 600 test queries, 3 test rooms; 600 validation queries, 3 validation rooms
- **Command & hardware:** `srl_lv00_run.sh`, n1 container `changwoo_audio_n1`, tmux window
  `srl_lv00`, cards **2** (test pair) and **3** (validation pair), ~9 min wall clock.
  Selection step is CPU-only.
- **Nothing in the repo was written.** All four tables went to a scratch `--out-dir`; the
  cited originals in `outputs/analysis/` are byte-unchanged (verified by md5 against the
  repo copies, which differ from the new ones).

### Primary result

| semantic checkpoint | selected by | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|
| `epoch=07-acc_rays_val=0.819` (current) | ray accuracy | **46.7** | 35.0 | 42.3 | +7.3 | [+4.0, +10.5] |
| `epoch=00-loss-valid=0.830` (this run) | validation loss | 41.3 | **29.0** | **38.7** | **+9.7** | [+6.3, +13.0] |

Per-query movement on the identity table: vision correct-at-1m went 210/600 → 174/600, with
**86 correctness flips (25 W→C, 61 C→W)**; 345/600 queries kept a bit-identical error. So the
swap genuinely changed the network, it did not merely jitter.

### What this supports

1. **The hypothesis is confirmed on the absolute numbers.** The `loss-valid` checkpoint is
   much weaker: vision 35.0 → 29.0, ours 42.3 → 38.7.
2. **The fix is rejected on the protocol's own terms.** Every discrete choice and scalar in
   this project is chosen by **validation recall@1m**. On that metric the current
   accuracy-selected checkpoint scores **46.7** and the `loss-valid` one **41.3**. Making the
   rows "uniform" on `loss-valid` would mean deliberately adopting a checkpoint that the
   selection criterion used everywhere else rejects by 5.4 points. That is not a fairness
   improvement, it is a worse model chosen on a metric the pipeline does not otherwise use.
3. **The coherent uniform rule already holds for two of three rows.** Select the semantic
   checkpoint by validation localisation recall@1m — the same metric as everything else.
   Replica's epoch 07 and MP3D's epoch 02 are already the accuracy-argmax checkpoints, which
   is the better proxy for it. **No change is needed to Replica or MP3D.**

### What it does NOT support, and the bigger finding

**The paired gain is not invariant to the backbone checkpoint.** It moved **+7.3 → +9.7**,
i.e. *up* by 2.4 points, while both absolute recalls fell. A weaker visual front end leaves
more room for acoustic verification to help.

This matters beyond the audit, and it cuts against the expectation that a within-dataset
paired Δ is unaffected by swapping a checkpoint both arms share. Both arms do share it — that
is what makes the Δ *internally valid at a fixed checkpoint* — but the checkpoint sets how
ambiguous the visual posterior is, and the gain is a function of that ambiguity. This is the
paper's own C-02 mechanism (gain concentrates where vision is ambiguous) showing up as a
confound: **`ΔR@1m` partly measures how weak the backbone is, so gains are not directly
comparable across rows with differently-selected checkpoints.** The two CIs here overlap
([+4.0,+10.5] vs [+6.3,+13.0]), so this single pair does not establish the effect
statistically — but the direction is the opposite of "unaffected", and it is reproducible at
will by re-selecting any checkpoint.

Consequence for the paper: when quoting gains across backbones or datasets, the backbone's
own recall must be shown alongside, which `headline_config.md` §2 already does. A gain
ranking without the vision column is not interpretable.

### Failure / limitation

- One checkpoint pair on one dataset. The Δ movement is a single observation whose CIs
  overlap; treat the direction as a hypothesis, not a measured effect. Testing it properly
  means re-selecting several checkpoints per backbone and regressing Δ on vision recall.
- `loss-valid` epoch 00 is the *first* epoch of training, so this contrast is partly
  "trained vs barely trained" rather than a clean metric comparison.
- S3D remains unverified: its `acc_rays_val` argmax is epoch 32, which is **not on disk**
  (`save_top_k=2` on a loss-monitored run kept only epochs 29 and 30). Cost to settle:
  one 40-epoch SemRayLoc semantic retrain on S3D, ≈6 h on one card at the measured
  2.68 it/s reference, plus one re-extraction. **Not launched — needs approval.**

### Artifact paths

Out of repo, under `$NODE_LOCAL1/echoloc_cvpr/srl_lv00/`:
`analysis/` (the four new table sets), `VAL_replica_srl_lossvalid_ep00.{md,json}`,
`logs/`, `srl_lv00_run.sh`, `farm2/` (the selection sandbox).

### Methodological trap found, worth recording

The first attempt at the selection step returned a result **bit-identical to the committed
one, including `median` to 16 digits**, which is how it was caught. Cause:
`val_select.py:29` is `REPO_ROOT = Path(__file__).resolve().parents[1]`, and `.resolve()`
follows symlinks. The sandbox had symlinked `scripts/` to the real repo, so `REPO_ROOT`
snapped back to the repo and `AN` read the original tables — the sandbox was bypassed
entirely and the run silently re-measured the baseline.

**Any sandbox for this repo must copy `scripts/` and `track1_core/` as real files**, not
symlink them; only data directories may be symlinked. A symlinked-code sandbox produces a
false null that looks like a clean negative result.

### Next decision

- Replica and MP3D: **no change.** Revise `docs/headline_config.md` §1 so it states the
  actual rule (semantic checkpoint by ray accuracy, `run_srl_pipeline.best_checkpoint`)
  instead of claiming every row is picked identically.
- S3D: retrain with `--monitor acc_rays_val` to close the last row, if the ≈6 GPU-hours are
  approved.
