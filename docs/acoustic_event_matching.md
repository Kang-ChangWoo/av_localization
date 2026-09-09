# One-sided acoustic evidence matching

Implemented and measured 2026-09-09. Training-free throughout: no learned
encoder, no scene-specific fitting, no parameter chosen against ground truth,
and no 3D rendering required by the method itself. SoundSpaces recordings are
used only as observations.

The question this answers is the narrow one:

> Does a clutter-robust, training-free comparison of 2D floorplan-induced
> acoustic path structure rank the true pose better than symmetric
> synthetic-vs-real RIR-envelope matching?

**Yes, and by a clear margin on the ranking metrics, but the absolute ranking
stays too weak to help a visual localizer.** Details below, including three
components that did not work.

## A. Audit of the existing implementation

Two acoustic pipelines exist. `track1_core/likelihood/acoustic.py` is the
canonical one; `scripts/gibson_runtime/planar_acoustic_likelihood.py` plus its
CUDA counterpart is the collaborator's Springhill path. The new work extends the
former and duplicates neither.

**The tracer is correct.** `trace_echoes` order-1 wall distances agree with
F3Loc's own DESDF cache to a correlation of 0.9999, median absolute error
0.006 m over 1440 rays. Whatever else is wrong, the floorplan geometry is not.

**It is not specular, and should not be described as such.** The tracer casts a
radial fan, marches to the first obstacle, and `synthesize_proxy` then connects
the hit point straight back to each microphone. For a co-located source and
receiver only the ray striking a surface at normal incidence actually returns,
so the closing leg is a *diffuse back-scatter assumption*. Order 1 is best read
as a radial wall-return proxy, which is the right object for a map-topology
likelihood and the wrong object to call room acoustics. For order 2 and above
the tracer does reflect correctly at the first bounce, but the final closing leg
is still a straight line to the receiver and is not a valid specular path. The
measured result agrees: order 2 and 3 do not improve ranking (see E).

**The comparison was the wrong shape.** `score_envelopes` normalises both
envelopes and takes a symmetric L1, which asks the floorplan to equal the room.
Furniture, scattering, and floor/ceiling paths all cost the candidate under that
distance even though no floorplan could have predicted them.

**Two-millisecond binning destroys the array.** The ring radius is 5 cm, so the
largest inter-microphone path difference is 10 cm, which is 2.3 samples at
8 kHz. A 16-sample energy bin cannot represent it.

**One real bug, found by this work.** Predicted delays were absolute path times
while observed delays were measured from the start of the reflection window. The
window begins at the direct peak plus a 16-sample guard, and the direct peak is
itself the 5 cm ring crossing, so the two sides were on clocks

    r/c + guard/fs  =  0.146 + 2.000  =  2.146 ms

apart. That is more than twice the matching tolerance, so predicted and observed
arrivals did not overlap at all. Fixing it moved GT rank on office_4 from 249 to
85 out of 683. `tests/test_acoustic_events.py::test_time_origin_convention`
pins the convention.

**A second issue in the collaborator CUDA path**, not modified here to preserve
reproducibility: `global_acoustic_likelihood_gpu.py` uses
`gain / path.clamp_min(0.1).sqrt()` as an amplitude, which makes energy fall as
`1/r` rather than `1/r^2`. It matters little for the new method, which
deliberately does not rely on predicted amplitude.

## B. Files changed

| file | purpose |
|---|---|
| `track1_core/likelihood/events.py` | new. Observation profile and sparse event extraction, predicted path tokens from traced echoes, the temporal kernel, the uniqueness constraint, robust aggregation, and both new scores. |
| `track1_core/likelihood/corners.py` | new. One-corner NLOS path hypotheses from the occupancy raster. |
| `tests/test_acoustic_events.py` | new. 17 deterministic tests, no dataset assets. |
| `scripts/event_likelihood_eval.py` | new. The ablation table. |
| `scripts/event_fusion_eval.py` | new. Training-free fusion with vision. |
| `scripts/plot_event_diagnostics.py` | new. Diagnostic panels A to E. |

`acoustic.py` is unchanged, so the `l1` baseline reproduces exactly.

## C. Method

Observation, per recording:

1. align on the direct peak, skip the guard, take the 1024-sample window
2. channel-summed sample-rate energy, Hann-smoothed over 0.25 ms, binned at
   0.25 ms
3. local maxima with 0.5 ms minimum separation, keeping only arrivals within
   10 dB of the strongest, capped at 40

Prediction, per candidate cell, once per scene:

    tau_i = (|source -> hit_i| + |hit_i -> source|) / c  -  (r/c + guard/fs)

merged when two predicted arrivals fall within 0.5 ms, with weight equal to the
summed solid angle. Source and receiver are co-located, so this is a function of
position only and is broadcast across all 36 yaw bins unchanged.

Matching:

    K_ij   = exp(-0.5 ((tau_i - tau_j) / sigma_tau)^2) * confidence_j
    K~_ij  = K_ij / max(1, sum_i K_ij)                   (uniqueness)
    m_i    = max_j K~_ij
    score  = weighted mean of the best `top_fraction` of {m_i}

Only predicted events enter the aggregation. Observed energy the floorplan
cannot explain costs nothing, which is the asymmetry the whole method rests on.

Defaults, all set from physics before any evaluation: `sigma_tau` 1.0 ms because
a 0.1 m pose grid moves a round trip by 0.6 m; `profile_resolution` 0.25 ms
because the ring spans 2.3 samples; `min_peak_separation` 0.5 ms because 17 cm
is the resolvable limit at this bandwidth; `peak_dynamic_range` 10 dB because a
floorplan predicts on the order of ten arrivals and a denser observed set makes
every candidate supported.

## D. Tests

17 pass, all synthetic and asset-free. `pytest tests/` is 34 passed.

| test | result |
|---|---|
| round-trip delay is 2d/c, four distances | pass |
| time-origin convention | pass |
| flat wall merges to one event | pass |
| **clutter robustness** | pass. Six unexplained echoes added: one-sided score 0.999 to 0.999, symmetric L1 0.00 to 1.06. This is the hypothesis as an assertion. |
| missing paths degrade gracefully | pass. Two of five paths removed: full-support 0.60, top-0.6 0.997. |
| wrong candidate ranks below truth | pass, margin 0.79 |
| temporal tolerance is smooth and monotone | pass |
| delay-only score is yaw-invariant | pass |
| uniqueness limits one peak explaining ten | pass, 1.00 to 0.32 |
| greedy and soft agree on a clean case | pass |
| dense and sparse variants agree | pass |
| padding contributes nothing | pass |
| corner paths appear in an L-room, not an open box | pass |
| corner delay is the broken path length | pass |

## E. Results

180 observations, 3 scenes (`office_4`, `apartment_2`, `frl_apartment_5`), about
700 valid candidate cells each under the shared `valid_pose_mask`, `raw_scan_open`
recordings, reflection order 1 unless stated. Chance GT percentile is 50.

| score mode | GT pct | GT rank | top 10% | <1 m | margin | s |
|---|---|---|---|---|---|---|
| **l1 (existing, symmetric)** | 60.0 | 411 | 13% | 14% | −0.67 | 0.3 |
| asymmetric dense energy | 60.6 | 443 | 13% | 6% | −0.43 | 1.0 |
| **event match, full support** | **66.0** | **342** | 16% | 6% | −0.45 | 3.1 |
| event match, top 0.8 | 64.9 | 361 | 17% | 4% | −0.47 | 2.7 |
| event match, top 0.6 | 62.8 | 410 | 15% | 4% | −0.49 | 3.0 |
| event match, top 0.4 | 60.0 | 453 | 18% | 6% | −0.48 | 2.6 |
| event match, 1/r² confidence | 63.1 | 357 | 18% | 10% | −0.50 | 2.8 |
| event match, no uniqueness | 61.7 | 468 | 17% | 6% | −0.47 | 2.2 |
| event match, 2 ms bins | 53.6 | 526 | 11% | 3% | −0.64 | 1.4 |
| event match, tolerance 0.5 ms | 57.8 | 496 | 17% | 5% | −0.55 | 2.8 |
| event match, tolerance 2.0 ms | 58.6 | 436 | 13% | 6% | −0.51 | 3.2 |
| event match + corner NLOS | 56.8 | 509 | 14% | 4% | −0.30 | 54.5 |
| event match, vertical window removed | 46.1 | 637 | 7% | 7% | −0.57 | 2.3 |

Matched-domain control, `floorplan_closed`, where the recording was rendered on
an extrusion of the same wall mask:

| score mode | GT pct | GT rank |
|---|---|---|
| l1 | 72.3 | 267 |
| event match, full support | 75.7 | 250 |
| event match, 1/r² confidence | **77.7** | **199** |

Reflection order, `raw_scan_open`, event match top 0.6:

| order | predicted events | GT pct | GT rank |
|---|---|---|---|
| 1 | 17.0 | 62.8 | 410 |
| 2 | 57.5 | 63.0 | 418 |
| 3 | 71.7 | 63.2 | 419 |

Fusion, 300 poses, mono, rank-normalised log-space combination, split by visual
margin (computed from the visual posterior alone, never the truth):

| stratum | acoustic weight | recall <1 m | GT rank median |
|---|---|---|---|
| all | 0 (vision only) | 39.3% | 278 |
| all | 0.25 | 22.7% | 340 |
| **vision ambiguous** | 0 (vision only) | 23.0% | 492 |
| **vision ambiguous** | 0.25 | 13.0% | **351** |
| vision confident | 0 (vision only) | 72.0% | 52 |
| vision confident | 0.25 | 42.0% | 227 |

Runtime: tracing is 0.7 s per scene for 683 cells and is reused for every
recording; scoring is about 3 s for 180 observations against 700 candidates.
Corner paths cost 55 s, dominated by the per-candidate visibility march.

## F. Failure analysis

**The one-sided formulation is the right shape and is not enough.** It beats the
symmetric baseline on every ranking metric that matters: GT percentile 60.0 to
66.0, GT rank 411 to 342. The unit test shows why, exactly: adding six
unexplainable echoes moves the symmetric distance by 1.06 and the one-sided
score by less than 0.001. But a median GT rank of 342 out of 700 is barely
better than chance in absolute terms, and recall at 1 m is *worse* than the
baseline (6% against 14%) because the score's own maximum lands elsewhere even
when the truth has moved up.

**The matched-domain control bounds the representation, not the clutter.** With
the recording rendered on the floorplan's own extrusion, so no furniture and no
unknown material, GT rank is still 199 to 250 of 700. Since the tracer agrees
with DESDF to 6 mm, the limit is not the geometry and not the clutter: it is
that a delay-only set of radial round trips is close to a rotationally symmetric
signature, and many cells in a room share one.

**Higher orders do not help, as the audit predicted.** Going from 17 to 72
predicted events per candidate changes GT rank by less than 3%. The closing leg
of an order-2 path is not a valid specular path, so the extra events are
geometrically arbitrary and add mostly noise.

**The robust dropout was not needed and mildly hurt.** `top_fraction` is
monotone: 0.4 gives 60.0, 0.6 gives 62.8, 0.8 gives 64.9, 1.0 gives 66.0. The
occlusion the dropout was designed for is real, but discarding the worst-matched
predicted events also lets wrong candidates keep only their lucky coincidences,
and on this data the second effect dominates. The toy test still shows the
intended behaviour in isolation, which is the honest reading: the mechanism
works, the situation does not call for it.

**The vertical-nuisance hypothesis was wrong.** The strongest observed arrivals
sit at 6 to 10 ms in every pose, where floor and ceiling bounces land for a
1.25 m mounting height, and the 2D proxy cannot produce them. Excluding that
window is a purely 2D operation needing no room height, so it was tested: it is
catastrophic, GT rank 411 to 637. The early window carries real positional
information despite being contaminated. Reported as a negative result.

**Corner NLOS paths hurt.** 20.7 additional paths per candidate, GT rank 411 to
509. The corner extraction itself is sound and the unit tests confirm the paths
appear only in an L-shaped room and carry the right broken-path length. The
problem is discrimination: `2(|src-corner| + rho)` for 16 to 64 corners times 32
directions generates a dense, largely position-insensitive set of delays, so it
dilutes the radial evidence rather than adding to it.

**Decay compensation trades one failure for the other.** A global dB threshold
truncates the observation at 19 ms while the floorplan keeps predicting to
35 ms, visible directly in diagnostic panel C. Normalising against the local
reverberant decay recovers the late arrivals and raises the observed set from 14
events to 38, at which point the mean spacing approaches the matching tolerance
and every candidate is supported again. Kept as an ablation, off by default.

**The ring energy term is not implementable as specified.** An arrival at a 5 cm
ring from 3 m away is a near-plane wave: the six path lengths differ by at most
10 cm and their energies by about 3%, so any predicted ring vector is nearly
uniform and its cosine similarity is nearly 1 for every candidate. Direction
lives in the sub-sample delay across the ring, not in energy. The hook is kept;
no misleading row is reported.

**Fusion does not currently pay.** At every weight and in every stratum, recall
falls. The one encouraging signal is confined to where the hypothesis predicts
it: in the ambiguous third, GT rank improves from 492 to 351, a 29% gain,
while in the confident third it degrades from 52 to 227. So the acoustic term
does carry information about which visual hypotheses are wrong, and does not
carry enough to be trusted with the decision.

## G. Recommendation

**Conclude that the 2D acoustic representation is the bottleneck, and stop
improving the matching.** The evidence is that the comparison has now been fixed
and the representation has not moved:

- the one-sided formulation is measurably better than symmetric L1 and is the
  correct shape, verified in isolation by the clutter test
- the tracer is correct to 6 mm against DESDF
- the time convention is fixed and tested
- yet the matched-domain ceiling, with no clutter at all, is GT rank 199 of 700

Every remaining lever inside the 2D delay-only formulation was tried and none
moved it: more reflection orders, corner NLOS paths, robust dropout, amplitude
confidence, three temporal tolerances, two binning resolutions, and two
observation-sparsity policies. A round-trip delay set from a co-located
source and receiver is close to a rotationally symmetric signature of the local
wall distances, and rooms contain many cells with similar ones.

The two directions with actual headroom both leave delay-only 2D:

1. **Inter-channel time difference across the ring.** It is the only
   direction-bearing quantity in the observation and it is currently discarded.
   It needs sub-sample cross-channel estimation at 2.3 samples of span, which is
   demanding but is a signal-processing problem rather than a representation
   one, and it would break the rotational symmetry that bounds the present score.
2. **A rendered candidate grid rather than an analytic proxy.** Measured
   earlier in this project, a SoundSpaces-rendered floorplan grid reaches GT
   percentile 95.9 and GT rank 2 to 6 on the same task where the analytic proxy
   reaches 66.0 and 342. That is the size of the representation gap, and it
   violates the training-free-and-render-free constraint, so it belongs in the
   decision about what the constraint is worth rather than in this method.
