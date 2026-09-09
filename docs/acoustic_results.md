# Can sound assist visual floorplan localization?

Results as of 2026-09-09. Everything here is measured with F3Loc's own metric
definitions on the Replica-derived dataset at `/root/storage/echoloc_dataset`,
except §1, which is on Gibson against the published table.

The question is not whether sound can localize. It cannot, on its own, in this
setting. The question is whether sound can reject visually plausible but
geometrically wrong hypotheses. The answer is yes, but only where vision is
uncertain, and only once the feature keeps frequency.

Reproduce with:

```
python scripts/margin_gated_fusion.py --feature band          # §3, §4
python scripts/stft_band_sweep.py                             # §2
python scripts/f3loc_comparison.py                            # §1, §5
python scripts/plot_probability_maps.py                       # figures
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

## 4. Result, band feature with the margin gate

Vision produces the `(H, W, 36)` posterior and its top-50 poses; the acoustic
term only reorders that shortlist, and only when the margin is below threshold.
Candidate generation is entirely visual, and orientation always comes from
vision.

| | 0.1 m | 0.5 m | 1 m | 1 m/30° | vs vision |
|---|---|---|---|---|---|
| mono, vision only | 4.5% | 30.1% | 38.2% | 34.9% | |
| mono + acoustic | 5.6% | 36.2% | 44.6% | 40.6% | **+6.4** |
| mv, vision only | 5.8% | 24.3% | 32.9% | 28.9% | |
| mv + acoustic | 6.9% | 28.6% | 37.0% | 32.9% | **+4.1** |
| comp, vision only | 5.1% | 31.5% | 38.6% | 35.5% | |
| comp + acoustic | 5.6% | 37.3% | 45.2% | 41.4% | **+6.6** |

`q=0.6` was chosen by looking at the whole sweep, so it is optimistic. Fitting
the threshold on replica_f alone and reporting on replica_g removes that:

| held out on replica_g | 0.1 m | 0.5 m | 1 m | vs vision |
|---|---|---|---|---|
| mono vision | 3.7% | 29.7% | 37.0% | |
| mono gated | 5.1% | 36.2% | **43.6%** | **+6.6** |
| mv vision | 3.2% | 17.7% | 29.1% | |
| mv gated | 4.4% | 22.6% | **32.4%** | **+3.3** |
| comp vision | 4.1% | 30.7% | 37.2% | |
| comp gated | 5.1% | 36.3% | **44.1%** | **+6.9** |

**Per scene** (mono, gate at q=0.6), so an average cannot hide a gain that only
one room produces:

| collection | scene | n | vision 1 m | gated 1 m | gain |
|---|---|---|---|---|---|
| replica_f | apartment_2 | 300 | 24.3% | 29.0% | +4.7 |
| replica_f | frl_apartment_5 | 300 | 74.3% | 72.0% | −2.3 |
| replica_f | office_4 | 300 | 19.3% | 32.7% | **+13.3** |
| replica_g | apartment_2 | 300 | 17.0% | 29.7% | **+12.7** |
| replica_g | frl_apartment_5 | 300 | 71.3% | 73.7% | +2.3 |
| replica_g | office_4 | 300 | 22.7% | 30.3% | +7.7 |

Five of six pairs gain. The gain tracks visual weakness: the two rooms where
vision sits near 20% gain 5–13 points, and frl_apartment_5, where vision already
reaches 71–74%, gains nothing. That is the same effect the gate was built for,
now visible across rooms rather than across poses.

## 5. How large is this, in F3Loc's terms

In the paper, going from a single view to multiview is 36.6% → 45.2% at 1 m, so
a whole extra camera view is worth +8.6 points under this metric. The held-out
acoustic gain is +6.6 (mono), +3.3 (mv), +6.9 (comp), or roughly 40–80% of an
extra view, at no extra view.

The paper's gap is Gibson and ours is Replica, so this is a sense of scale
rather than an equality. What it establishes is that the gain is the size of an
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
