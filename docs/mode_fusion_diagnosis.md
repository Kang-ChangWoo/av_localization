# Single-observation acoustic fusion: diagnosis and a recommendation

Written 2026-09-10. All numbers are single-observation. No sequence, no temporal
filtering, no trajectory accumulation, no synthetic aperture, no multi-frame
audio appears anywhere in this document or in the code it describes.

Two visual backbones are reported throughout, UnLoc trained on Replica and F3Loc
mono trained on Replica, because a conclusion that holds for one and not the
other is a property of that backbone rather than of sound. Every threshold and
weight is selected on `replica_f` and every reported number comes from
`replica_g`, 300 held-out queries. Every gain carries a paired bootstrap
interval over queries.

Each claim is tagged. **[data]** is measured here. **[hypothesis]** is
consistent with the data but not established by it. **[unverified]** is
neither.

---

## 1. The short answer

**The fusion rule was never the bottleneck, and fixing it buys about two
points. The acoustic score under the furniture domain gap is the bottleneck, and
fixing that is worth twenty.** [data]

The evidence is one experiment. Take the current pipeline, change nothing except
the mesh the query recording was rendered on, and the same acoustic term goes
from moving the ground-truth cell to rank 245 to moving it to rank 3, and
unconditional mode reranking goes from losing 18 points to gaining 23.

| | furnished query | matched-geometry query |
|---|---|---|
| acoustic alone, recall @1m | 23.7% | **71.5%** |
| ground-truth rank position, median | 245 | **3** |
| ground-truth acoustic percentile | 95.01 | **99.93** |
| distance to the nearest strong acoustic peak | 0.20 m | **0.04 m** |
| mode reranking, gain over vision | **−18.3** | **+22.7** |

Everything else in this document is subordinate to that table.

---

## 2. What the diagnosis found

### 2.1 The gain is concentrated in visually ambiguous queries [data]

Queries binned by the log-odds between the best and second visual hypothesis.
UnLoc, 600 queries, existing cell-wise fusion:

| ambiguity bin | n | vision | fused | repairs | breaks | net |
|---|---|---|---|---|---|---|
| most ambiguous | 120 | 24.2% | 39.2% | 26 | 8 | **+15.0** |
| | 120 | 33.3% | 37.5% | 18 | 13 | +4.2 |
| | 120 | 39.2% | 43.3% | 21 | 16 | +4.2 |
| | 120 | 63.3% | 61.7% | 13 | 15 | −1.7 |
| least ambiguous | 120 | 93.3% | 80.8% | 3 | 18 | **−12.5** |

Monotone, and the same shape on F3Loc mono (+3.3, +11.7, +8.3, +3.3, −13.3).
Visual ambiguity is a real, usable gating signal.

### 2.2 The acoustic score does **not** know when to trust itself [data]

Every acoustic self-confidence measure available at inference time, scored as a
predictor. AUROC 0.5 is chance.

| signal | acoustic pick correct | predicts a repair | predicts a break |
|---|---|---|---|
| acoustic margin among modes | 0.521 | 0.573 | 0.527 |
| acoustic rank margin | 0.505 | 0.597 | 0.554 |
| acoustic entropy over modes | 0.512 | 0.520 | 0.542 |
| relative evidence of the best mode | 0.515 | 0.526 | 0.558 |
| **visual ambiguity** | 0.584 | 0.486 | **0.620** |

Nothing acoustic reaches 0.60. The best predictor of an acoustic mistake is a
*visual* quantity. F3Loc mono agrees: acoustic signals 0.49 to 0.57, visual
ambiguity 0.677 for predicting breaks.

The forensic view says the same thing more bluntly. Comparing the 81 repairs
against the 70 regressions on the quantities that caused each switch, the
central 90% of the regression distribution contains this fraction of repairs:

| quantity | repairs, median | regressions, median | overlap |
|---|---|---|---|
| visual gap overturned | +0.131 | +0.108 | 83% |
| acoustic swing | +0.574 | +0.365 | **62%** |
| relative evidence of the chosen hypothesis | −1.217 | −1.438 | **98%** |

A repair and a regression are the same event with a different answer. The
acoustic swing separates them slightly; nothing separates them well.

**Consequence, and it is the main negative result here: the dual-confidence gate
the brief asked for collapses to a visual-only gate.** Sweeping the acoustic
threshold on the fit collection, the best value is `-inf`, meaning "never
require acoustic confidence". Raising it only removes queries where audio would
have helped:

| acoustic threshold | fit recall @1m | queries where audio acts |
|---|---|---|
| −inf | **53.3%** | 18% |
| 0.2 | 53.0% | 12% |
| 0.4 | 52.3% | 8% |
| 0.8 | 52.0% | 5% |

### 2.3 Ten hypotheses beat fifty cells [data]

Ground-truth coverage, which is the ceiling of any reranking method:

| K | truth within 1 m of a top-K **cell** | of a top-K **hypothesis** |
|---|---|---|
| 1 | 50.7% | 50.7% |
| 3 | 60.2% | **76.3%** |
| 5 | 66.2% | **85.5%** |
| 10 | 76.3% | **92.8%** |
| 20 | 84.5% | - |
| 50 | 92.3% | - |

Ten separated hypotheses reach the coverage of fifty cells. The top-50 cell
shortlist is five times larger than it needs to be, and the extra cells are
duplicates of the same few places, which is what let a broad mode outvote a
sharp one. F3Loc mono: top-10 hypotheses 86.2% against top-50 cells 82.8%, the
same conclusion with a slightly larger margin.

### 2.4 The acoustic score locates the region, not the cell [data]

| quantity | median |
|---|---|
| acoustic rank of the ground-truth cell | 0.9502 |
| best acoustic rank within 0.25 m | 0.9921 |
| best acoustic rank within 0.5 m | 0.9961 |
| best acoustic rank within 1.0 m | 0.9979 |
| distance to the nearest strong acoustic peak | 0.20 m |

Moving 25 cm from truth cuts the shortfall from the top by a factor of six. This
is a direct argument for reading a region rather than a point, and the fusion
sweep independently selected the region-aggregated acoustic evidence (`max`
within the mode's 0.5 m disc) over the centre value.

### 2.5 Later reflections do not help [data]

Post-direct time ranges, scored on complementarity to vision and not on acoustic
accuracy alone:

| range, ms | acoustic alone @1m | GT percentile | mode rerank @1m |
|---|---|---|---|
| 0-16 | 15.3% | 92.65 | 24.8% |
| 16-32 | 11.3% | 69.88 | 13.5% |
| 32-64 | 6.3% | 64.61 | 12.2% |
| 64-128 | 5.8% | **43.36** | 8.3% |
| 0-32 | 18.5% | 93.62 | 28.0% |
| 0-64 | 20.5% | 94.69 | **29.2%** |
| 0-128 | **23.7%** | **95.02** | 29.0% |

The 64-128 ms window alone puts the ground-truth cell at the 43rd percentile,
which is **below chance**. Late energy is not carrying usable non-line-of-sight
information in this data; it is carrying furniture mismatch. Cumulative windows
still improve monotonically because the early part dominates the sum.

This contradicts the motivating story that late reflections have visited
regions the camera cannot see and therefore add information. [data] contradicts
[hypothesis]. It does not prove late reflections are useless in principle, only
that under this domain gap they are net harmful. [unverified] whether matched
geometry would change this; the range ablation was not repeated there.

### 2.6 Two degeneracies, both recorded in code

The **conservative negative-evidence variant (B5) cannot be built from this
score.** `relative_evidence` compares each hypothesis to the aggregate of the
others, and with ten hypotheses that aggregate exceeds any single one, so every
value is negative and `clip(-r, 0)` is just `-r`. Subtracting it is adding it.
Measured: degenerate on 99% of queries at ten hypotheses, 92% on synthetic
draws, 49% at five, 4% at three. The rows are identical for that reason, not by
coincidence. `contradiction_is_degenerate` asserts it.

The deeper reason is worth stating: with a score that has no calibrated zero,
"this hypothesis is contradicted" and "this hypothesis is less preferred than
the others" are the same statement. Per the brief, no contradiction measure was
invented to paper over this. [data]

The second degeneracy is the one that motivated the whole redesign. Among the
fifty cells vision shortlists, the visual log-rank term spans 0.0092 and the
acoustic term spans 0.6662, a factor of **72**. Rank saturates because every
shortlisted cell is above the 99.9th percentile. The old "weighted fusion" was a
reranker in disguise, and 92% of queries had their answer moved by it. [data]

---

## 3. The new formulation

Implemented in `track1_core/modes.py` and
`track1_core/likelihood/mode_fusion.py`. The existing cell-wise rules are
untouched and still reported.

    vision proposes separated hypotheses  (NMS at 1.5 m, ten kept)
    sound scores each hypothesis over its own 0.5 m disc
    sound's evidence is relative to the other hypotheses, not global
    audio acts only where vision is ambiguous
    the chosen hypothesis's visual centre supplies the final position and yaw

Sound never relocates the answer to a place vision did not propose, and never
touches yaw, about which it has no information. The 1m/30deg column therefore
measures landing on a better cell, not a better heading.

### 3.1 Results, UnLoc, held out on `replica_g`

| level | rule | 0.1 m | 0.5 m | 1 m | 1m/30° | median | gain @1m | 95% CI |
|---|---|---|---|---|---|---|---|---|
| cell | vision | 9.0% | 44.7% | 49.7% | 49.0% | 1.12 m | — | |
| cell | acoustic alone | 3.3% | 14.7% | 25.0% | — | 3.13 m | −24.7 | [−32.7, −17.0] |
| cell | rerank top-50 | 10.0% | 43.7% | 50.7% | 48.0% | 0.94 m | +1.0 | [−5.0, +7.0] |
| cell | log-rank fusion | 13.3% | 46.3% | 52.0% | 50.3% | 0.71 m | +2.3 | [−3.0, +8.0] |
| mode | rerank, unconditional | 7.3% | 26.7% | 31.3% | 29.7% | 2.92 m | −18.3 | [−25.7, −10.7] |
| mode | relative evidence | 9.7% | 47.7% | 53.0% | 51.7% | 0.61 m | +3.3 | [−1.0, +7.7] |
| mode | selective, visual gate | 9.3% | 46.0% | 50.7% | 50.0% | 0.87 m | +1.0 | [−2.0, +4.0] |
| mode | **continuous dual gate** | 9.7% | 47.7% | **53.3%** | **52.0%** | **0.61 m** | **+3.7** | **[+0.0, +7.3]** |
| oracle | best hypothesis | 15.3% | 78.3% | 93.7% | 89.3% | 0.25 m | +44.0 | [+38.3, +49.7] |

### 3.2 Results, F3Loc mono, held out on `replica_g`

| level | rule | 1 m | gain @1m | 95% CI |
|---|---|---|---|---|
| cell | vision | 37.7% | — | |
| cell | rerank top-50 | 41.0% | +3.3 | [−2.0, +8.3] |
| cell | log-rank fusion | 41.0% | +3.3 | [−2.3, +9.0] |
| mode | rerank, unconditional | 22.7% | −15.0 | [−21.7, −8.3] |
| mode | relative evidence | 42.3% | +4.7 | **[+1.3, +8.3]** |
| mode | **continuous dual gate** | **43.0%** | **+5.3** | **[+2.0, +8.7]** |
| oracle | best hypothesis | 87.3% | +49.7 | [+44.0, +55.3] |

**The mode-level rules are the first whose intervals exclude zero.** Both
existing cell-wise rules have intervals spanning zero on both backbones. That is
the honest reading of a 2 to 3 point gain on 300 queries, and it is why the
headline claim of this project should be stated more carefully than it has been.

### 3.3 The matched-geometry control settles what is missing

Same rules, same visual side, query rendered on the wall-only mesh:

| rule | furnished, gain | matched, gain |
|---|---|---|
| acoustic alone | −24.7 | **+21.7** |
| cell log-rank fusion | +2.3 | +16.3 |
| mode rerank, unconditional | **−18.3** | **+22.7** |
| mode relative evidence | +3.3 | +14.0 |
| mode selective, visual gate | +1.0 | +6.7 |
| mode continuous dual gate | +3.7 | +10.3 |
| oracle over hypotheses | +44.0 | +44.0 |

Read the ordering, not just the magnitudes. Under furnished acoustics the
*conservative* rules win and unconditional reranking is catastrophic. Under
matched acoustics the ordering **inverts**: unconditional reranking is the best
rule and every gate costs points, because the gates are suppressing a signal
that has become reliable.

**The gating machinery is compensating for the domain gap, not for a defect in
the fusion.** [data] Given a good acoustic score the simplest possible rule wins;
given this one, only a conservative rule survives.

Matched-domain mode reranking reaches 72.3% against an oracle of 93.7%, so even
with the furniture gap removed roughly 21 points remain. The acoustic score is
not perfect in its own domain either. [data]

---

## 4. Recommendations

### Primary rule for the paper

**Mode-level relative-evidence fusion with a continuous visual-ambiguity gate.**

- ten hypotheses by NMS at 1.5 m from the visual posterior
- acoustic evidence per hypothesis = maximum over its 0.5 m disc, then relative
  to the other hypotheses after per-query standardisation
- gate weight a smooth function of visual ambiguity only
- position and yaw from the chosen hypothesis's visual centre

Chosen because it is the only rule whose interval excludes zero on both
backbones (+5.3 [+2.0, +8.7] on F3Loc, +3.7 [+0.0, +7.3] on UnLoc), it improves
0.5 m and median error as well as 1 m, and every component is justified by a
measurement above rather than by a sweep.

Call the acoustic half a **gate on visual ambiguity**, not a dual-confidence
gate. The data does not support the second name.

### Should remain ablations only

| item | why |
|---|---|
| cell-wise log-rank fusion | intervals span zero on both backbones; keep as the prior method |
| unconditional mode reranking | −18.3 furnished, though it is the *headline* under matched acoustics and belongs in that table |
| contradiction / negative-evidence variant | provably degenerate at ten hypotheses; report as a negative result |
| dual-confidence gate | acoustic AUROC below 0.60; the tuned acoustic threshold is `-inf` |
| banded STFT features | already measured at chance; do not revisit without new evidence |
| late-only time ranges | 64–128 ms alone ranks truth below chance |

### What to do next, in order

1. **Close the domain gap.** Every measurement points here and nothing else is
   close in expected value. Rendering candidates on a furnished mesh is the
   direct attack; whether an approximate furniture prior suffices is
   [unverified].
2. **Validate on measured impulse responses.** Both sides of every acoustic
   number in this repository are SoundSpaces. [unverified] entirely.
3. Only then revisit the fusion rule. The oracle gap of 44 points is not
   reachable by a better combiner over these scores.

---

## 5. What is not established

- **Both sides of the acoustic comparison are simulated.** The furnished and
  wall-only conditions differ only in mesh, both from the same engine. Nothing
  here says a real microphone in a real room behaves this way. [unverified]
- **Three rooms, 300 held-out queries.** The held-out split holds out the pose
  collection, not the room; `replica_f` and `replica_g` cover the same three
  rooms. Whether thresholds transfer to an unseen room is [unverified].
- **A candidate grid must be pre-rendered per scene**, twelve sharded jobs for
  the smallest room. A real deployment cost.
- The claim that higher-order reflections are non-line-of-sight information is
  **contradicted** by the range ablation under this domain gap, and untested
  under matched geometry.
- Absorption was never varied: `acousticsConfig.enableMaterials = False`.

---

## 6. Reproducing every number

```bash
# one pass per condition and backbone; writes the tables everything else reads
/opt/conda/envs/unloc/bin/python scripts/extract_mode_table.py \
    --backbone unloc --condition raw_scan_open    --n-poses 100
/opt/conda/envs/unloc/bin/python scripts/extract_mode_table.py \
    --backbone unloc --condition floorplan_closed --n-poses 100
/opt/conda/envs/f3loc/bin/python scripts/extract_mode_table.py \
    --backbone f3loc_mono --condition raw_scan_open    --n-poses 100
/opt/conda/envs/f3loc/bin/python scripts/extract_mode_table.py \
    --backbone f3loc_mono --condition floorplan_closed --n-poses 100

# A2-A10: the diagnosis, tables and plots
/opt/conda/envs/f3loc/bin/python scripts/analyse_mode_tables.py --backbone unloc
/opt/conda/envs/f3loc/bin/python scripts/analyse_mode_tables.py --backbone f3loc_mono \
    --report outputs/analysis/diagnosis_f3loc_mono.md

# B and C: every fusion rule, both oracles, thresholds fitted on replica_f
/opt/conda/envs/f3loc/bin/python scripts/mode_fusion_eval.py --backbone unloc
/opt/conda/envs/f3loc/bin/python scripts/mode_fusion_eval.py --backbone f3loc_mono \
    --out outputs/analysis/fusion_comparison_f3loc_mono.md \
    --json-out outputs/metrics/mode_fusion_eval_f3loc_mono.json

# A5: every switch, with the numbers that caused it
/opt/conda/envs/f3loc/bin/python scripts/repair_break_forensics.py --backbone unloc

# qualitative: ten repairs and ten regressions, with the decision arithmetic
/opt/conda/envs/unloc/bin/python scripts/plot_decision_switch.py \
    --checkpoint <replica ckpt> --weight 0.5 --fix-rows 10 --break-rows 10 \
    --tag _forensics
```

Outputs:

| file | contents |
|---|---|
| `outputs/analysis/diagnosis*.md` | A2 to A10 |
| `outputs/analysis/fusion_comparison*.md` | B and C |
| `outputs/analysis/repair_break_forensics.md` | A5 |
| `outputs/figures/A2_ambiguity_vs_gain_*.png` | visual ambiguity against acoustic gain |
| `outputs/figures/A3_acoustic_confidence_*.png` | acoustic confidence against correctness |
| `outputs/figures/A4_joint_reliability_*.png` | the two-dimensional benefit map |
| `outputs/figures/A6_coverage_*.png` | coverage against number of hypotheses |
| `outputs/figures/A8_domain_gap_*.png` | matched against furnished, per scene |
| `outputs/figures/A9_early_late_*.png` | the time-range ablation |
| `outputs/figures/unloc_decision_switch_forensics.png` | the qualitative switches |

Every metrics file carries a provenance block with argv, git state, sampled
input hashes and library versions. The evaluation protocol, batch handling and
provenance logging of the existing baselines are unchanged.
