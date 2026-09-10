# Can sound assist visual floorplan localization?

Results as of 2026-09-10. Everything here is measured with F3Loc's own metric
definitions on the Replica-derived dataset at `/root/storage/echoloc_dataset`,
except §1, which is on Gibson against the published table.

§4 and §5 were re-measured on 2026-09-10 against a candidate grid re-rendered to
match the shipped recordings, after the dataset was re-rendered underneath this
code at a different sample rate and every acoustic number collapsed to chance.
`docs/data_provenance_incident.md` records what happened. The recovered numbers
agree with the pre-incident ones, so the earlier results were sound and the
break was in reading the files, not in the method. §2 and §3 still quote the
pre-incident grid; their conclusions were re-checked but their tables were not
re-run.

The question is not whether sound can localize. It cannot, on its own, in this
setting. The question is whether sound can reject visually plausible but
geometrically wrong hypotheses. The answer is yes, but only where vision is
uncertain, and only once the feature keeps frequency.

Reproduce with:

```
python scripts/replica_official_eval.py --per-scene            # §4, §5
python scripts/envelope_rerank_eval.py --grid-dir outputs/acoustic_grid_v2
python scripts/stft_band_sweep.py                              # §2
python scripts/plot_probability_maps.py                        # figures
```

## 1. Reproduction fidelity on Gibson

The authors' released checkpoints are fetched by
`scripts/fetch_official_checkpoints.sh` and evaluated through this project's
`scripts/eval_visual.py` with `--bn-mode upstream`, which reproduces upstream's
own handling: `eval_observation.py` calls `.eval()` on the complementary net
only, leaves the monocular and multi-view nets in train mode, and iterates the
test set one sample at a time. Batch size is therefore part of the protocol,
not a throughput knob, and `eval_visual.py` now forces it to 1 in that mode.

**All three published rows reproduce**, every metric within 0.1 point:

| | 0.1 m | 0.5 m | 1 m | 1 m/30 deg |
|---|---|---|---|---|
| paper, Ours_s (gibson_f) | 4.7% | 28.6% | 36.6% | 35.1% |
| official `mono.ckpt` | **4.7%** | **28.5%** | **36.5%** | **35.0%** |
| paper, Ours_m (gibson_f) | 13.2% | 40.9% | 45.2% | 43.7% |
| official `mv.ckpt` | **13.2%** | **40.8%** | **45.1%** | **43.6%** |
| paper, Ours_f (gibson_g) | 12.2% | 39.4% | 44.5% | 43.2% |
| official `comp.ckpt` | **12.2%** | **39.3%** | **44.4%** | **43.1%** |

Cross-checked by running upstream's unmodified `eval_observation.py` on the same
weights and data: it returns 36.52 / 28.50 / 4.68 / 34.98 for the monocular net,
matching this project's evaluation.

**Our from-scratch checkpoints against the released ones**, same protocol:

| | official | ours | difference |
|---|---|---|---|
| mono gibson_f | 36.5% | **39.6%** | +3.1 |
| mv gibson_f | 45.1% | **47.2%** | +2.1 |
| comp gibson_g | 44.4% | **44.7%** | +0.3 |

Upstream published no training script, so the optimiser settings were this
project's choices; they land slightly above the released weights on all three.

The authors' own settings are recoverable from the checkpoints, which carry
optimizer state even though they carry no `hyper_parameters`: Adam, constant
learning rate 1e-3, no weight decay, no scheduler, 100 epochs for mono, 20 for
mv, 5 for comp, with model selection on `l1_loss-valid` reaching 0.1618, 0.1295
and 0.1512 respectively. Those validation losses are the cleanest target for a
training reproduction, since they are measured before the localization protocol
and so cannot be affected by the batch-size issue above.

## 2. The acoustic feature has to keep frequency

The original feature was energy per 0.125 ms window, summed over all
frequencies. Furniture scatters as a function of wavelength (343/500 Hz = 69 cm
diffracts around a chair; 343/3000 Hz = 11 cm scatters off it), and the
floorplan model has no furniture, so frequency is the one axis along which the
model's error is structured.

GT percentile / recall@1m, scoring candidates against the real scan recordings:

| band | office_4 | apartment_2 | frl_apartment_5 |
|---|---|---|---|
| full (broadband) | 89.4 / 11% | 77.5 / 15% | 66.5 / 2% |
| 0–500 Hz | **93.7 / 26%** | **78.4** / 12% | 74.6 / 4% |
| 500–1500 Hz | 85.3 / 15% | 70.0 / 7% | 76.4 / 8% |
| 2–4 kHz | 89.0 / 27% | 70.3 / 6% | **79.4** / 8% |

Every scene has a band that beats broadband, but which band wins is not stable,
so a single band is overfitting. Three bands used jointly is what §3 uses.

**Matched-domain control.** Rendering the candidates and the recording on the
same mesh (`floorplan_closed`), recall@1m:

| band | office_4 | apartment_2 | frl_apartment_5 |
|---|---|---|---|
| full | 77% | 96% | 92% |
| 0–1 kHz | **98%** | **100%** | **100%** |
| 2–4 kHz | 32% | 30% | 38% |

Two things follow. The low band is near-perfect when the model matches the room,
so the 4–26% it gets on real scans is the furniture gap, not a feature defect.
And the high band is poor even here, where there is no furniture gap at all, so
its weakness is render stochasticity rather than the clutter it was meant to
expose.

**Time-resolution control.** Banding also coarsened the time axis from 0.125 ms
to 2 ms. Running a single 0–4000 Hz band at the same nfft and hop isolates that:

| mono, 1 m recall | always-on | gated q=0.6 | held out |
|---|---|---|---|
| broadband, 0.125 ms | +2.0 | +3.7 | +2.7 |
| one band, 2 ms | **−0.6** | +2.1 | +1.8 |
| three bands, 2 ms | **+4.2** | **+6.4** | **+6.6** |

Coarsening time alone makes it worse. The frequency split is the gain.

## 3. Sound must be gated on visual confidence

Stratifying by the gap between the best visual pose and its strongest spatially
separated competitor, with the old feature:

| margin stratum | n | vision | fused | gain |
|---|---|---|---|---|
| low (ambiguous) | 600 | 20.2% | 26.2% | **+6.0** |
| medium | 600 | 23.3% | 28.2% | +4.8 |
| high (confident) | 600 | 70.7% | 65.8% | **−4.8** |

Applying the acoustic term unconditionally averages these two effects together.
Gating on the margin keeps the first and drops the second. The margin is
computable at inference time, so the gate is part of the method, not an oracle.

Entropy, the obvious alternative signal, orders the strata the wrong way
(+4.2 confident, −1.8 ambiguous) and should not be used.

## 4. Result, on the metric set the papers report

Every earlier Replica number in this project was recall at 1 m, which is one row
of the four that F3Loc, SemRayLoc and DisCo-FLoc all report. It is also the most
forgiving: a fusion rule can gain at 1 m by moving mass into roughly the right
room while getting no better at putting the pose in the right 0.1 m cell or at
recovering heading. `scripts/replica_official_eval.py` runs the full set, error
definitions copied from `f3loc/eval_observation.py`.

Five rows per backbone, 600 poses over the three test scenes and both
collections, candidate grid `outputs/acoustic_grid_v2`, envelope at 2 ms.

**F3Loc mono**, fusion weight 2, gate at q=0.6:

| | 0.1 m | 0.5 m | 1 m | 1 m/30 deg | median | vs vision |
|---|---|---|---|---|---|---|
| vision only | 5.2% | 30.0% | 38.3% | 34.7% | 2.15 m | |
| audio only | 3.0% | 13.8% | 23.7% | — | 3.13 m | −14.7 |
| rerank top-50 | 4.8% | 30.7% | 41.5% | 35.5% | 1.91 m | +3.2 |
| fused | 3.3% | 27.8% | 41.3% | 33.8% | 1.72 m | +3.0 |
| **gated fusion** | **5.7%** | **36.0%** | **45.3%** | **38.8%** | **1.39 m** | **+7.0** |

**DisCo-FLoc RRP**, fusion weight 4:

| | 0.1 m | 0.5 m | 1 m | 1 m/30 deg | median | vs vision |
|---|---|---|---|---|---|---|
| vision only | 0.8% | 15.5% | 31.0% | 29.7% | 2.42 m | |
| audio only | 3.0% | 13.8% | 23.7% | — | 3.13 m | −7.3 |
| rerank top-50 | 2.3% | 20.7% | 34.3% | 31.5% | 2.34 m | +3.3 |
| **fused** | **2.5%** | **23.7%** | **40.3%** | **35.5%** | **1.90 m** | **+9.3** |
| gated fusion | 1.8% | 19.5% | 36.8% | 32.2% | 1.92 m | +5.8 |

Four things the 1 m row alone did not show.

**The gain survives the tight thresholds.** Gated mono improves 0.1 m as well as
1 m, and cuts median error from 2.15 m to 1.39 m. The fusion is not only moving
mass into the right room.

**Heading improves even though sound supplies none.** The acoustic score is one
number per cell, flat over the 36 heading bins, so it cannot move the per-cell
heading argmax; every fused heading is vision's own heading at whichever cell
wins. The 1 m/30 deg gain (+4.2 mono, +5.8 DisCo) comes entirely from landing on
a cell whose visual heading was already right.

**Audio alone is not competitive and does not need to be.** It reaches 23.7% at
1 m against vision's 38.3%, and adds 7.0 points on top of it. What it
contributes is decorrelated error, not accuracy.

**The two backbones want different rules.** Mono is best gated, DisCo is best
fused unconditionally. DisCo's confidence is less informative here: its margin
does not separate its own successes from its failures as cleanly as mono's does,
so gating on it discards good fusions.

## 5. The gain tracks visual weakness, per scene

| backbone | scene | vision 1 m | audio 1 m | best 1 m | gain |
|---|---|---|---|---|---|
| mono | office_4 | 20.0% | 34.0% | 32.5% | **+12.5** |
| mono | apartment_2 | 18.5% | 27.0% | 32.0% | **+13.5** |
| mono | frl_apartment_5 | 76.5% | 10.0% | 75.0% | −1.5 |
| DisCo RRP | office_4 | 32.0% | 34.0% | 43.5% | **+11.5** |
| DisCo RRP | apartment_2 | 32.0% | 27.0% | 41.5% | +9.5 |
| DisCo RRP | frl_apartment_5 | 29.0% | 10.0% | 36.0% | +7.0 |

frl_apartment_5 is the test. It is the one room where sound is nearly useless on
its own, at 10% against a chance of roughly 0. Mono already reaches 76.5% there
and correctly gains nothing; the gate limits the damage to 1.5 points. DisCo
reaches only 29.0% in the same room off the same images, and there sound is
worth +7.0 despite being weak in absolute terms.

So the axis is not the room and not the recording quality. It is how much the
visual posterior is failing, and that holds within a room across two backbones,
not only across rooms. A single-number claim of the form "sound adds N points"
is the wrong shape for this result.

**What caps reranking.** Truth is inside vision's top-50 only 83–84% of the
time, and reranking cannot exceed that. Weighted fusion is not capped the same
way, which is why it beats reranking for DisCo (+9.3 against +3.3).

## 5b. How large is this, in F3Loc's terms

In the paper, going from a single view to multiview is 36.6% to 45.2% at 1 m, so
a whole extra camera view is worth +8.6 points under this metric. The acoustic
gain is +7.0 for mono and +9.3 for DisCo, which is the size of an extra camera,
from a microphone and a pre-rendered grid.

The paper's gap is Gibson and this is Replica, so it is a sense of scale rather
than an equality. What it establishes is that the gain is the size of an
architectural change, not noise.

## 6. What is not established

- **Only three scenes.** They are the entire test split, and no test scene was
  omitted, but replica_g holds out the protocol, never the room. Whether the
  threshold transfers to an unseen room is untested.
- **The candidate grid must be pre-rendered per scene**, which is a real
  deployment cost.
- **Headroom is mostly unused.** The oracle over the same top-50 shortlist is
  77–87% everywhere, against 44% achieved.
- **Absorption was never varied.** The sweep that appeared to test it produced
  bit-identical renders: `acousticsConfig.enableMaterials = False`, so absorption
  is not a parameter of the current render path. `scripts/eval_acoustic_grid.py`
  will score the variants if they are ever re-rendered.

## Figures

`outputs/figures/probability_maps_grid_echoloc_mono_fg.png` — 24 poses spanning
the margin range. At low margin the posterior breaks into scattered clusters and
the shortlist covers the room; at high margin it is a single tight blob and the
gate correctly does nothing.

`outputs/figures/probability_maps_detail_echoloc_mono_fg.png` — five poses with
the visual posterior, the acoustic score over the whole candidate grid, and the
re-ranked shortlist side by side.
