# EXECUTION LOG

One card per executed experiment, in the format of §16 B of the programme spec. Only
experiments that actually ran appear here.

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
