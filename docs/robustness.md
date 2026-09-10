# Robustness of the acoustic fusion

Everything here is computed from the extracted mode tables, so it costs seconds and re-runs after any change to the pipeline.


## Number of spatial hypotheses

The hypothesis list is truncated; nothing else changes. Coverage is the fraction of queries with a hypothesis within 1 m of truth, which is the ceiling of any selection rule over that list.

| backbone | K | coverage | recall @1 m | gain |
|---|---|---|---|---|
| F3Loc mono | 2 | 51.7% | 39.0% | +1.3 |
| F3Loc mono | 3 | 62.7% | 41.3% | +3.7 |
| F3Loc mono | 5 | 71.0% | 43.3% | +5.7 |
| F3Loc mono | 10 | 87.3% | 45.7% | +8.0 |
| UnLoc | 2 | 64.3% | 50.0% | +0.3 |
| UnLoc | 3 | 75.7% | 51.0% | +1.3 |
| UnLoc | 5 | 86.3% | 52.3% | +2.7 |
| UnLoc | 10 | 93.7% | 54.7% | +5.0 |

The gain grows monotonically with K on both backbones and has not saturated at ten, so ten is a floor set by the extraction cost rather than a tuned constant.


## Leave-one-room-out

Thresholds fitted on two test rooms and reported on the third, rotating. Every reported query comes from a room whose thresholds were chosen without seeing it. Both pose collections are pooled here, since the room is now what is held out.

| backbone | held-out room | n | vision | ours | gain | 95% CI |
|---|---|---|---|---|---|---|
| F3Loc mono | apartment_2 | 200 | 18.5% | 27.5% | +9.0 | [+4.5, +14.0] |
| F3Loc mono | frl_apartment_5 | 200 | 76.5% | 67.0% | -9.5 | [-15.0, -4.0] |
| F3Loc mono | office_4 | 200 | 20.0% | 35.5% | +15.5 | [+8.5, +22.5] |
| **F3Loc mono** | **pooled** | 600 | 38.3% | 43.3% | **+5.0** | [+1.7, +8.5] |
| UnLoc | apartment_2 | 200 | 29.5% | 42.0% | +12.5 | [+7.0, +18.5] |
| UnLoc | frl_apartment_5 | 200 | 77.5% | 76.0% | -1.5 | [-5.5, +2.0] |
| UnLoc | office_4 | 200 | 45.0% | 55.0% | +10.0 | [+3.0, +16.5] |
| **UnLoc** | **pooled** | 600 | 50.7% | 57.7% | **+7.0** | [+3.8, +10.3] |

## Sensitivity to each tuned scalar

One scalar moved at a time from the selected value, the rest held. Reported on the held-out collection, so these are not re-tuned.


**F3Loc mono**, selected setting gives 45.7% against vision 37.7%.

| scalar | values and recall @1 m |
|---|---|
| weight | 0.25: 44.7 , 0.5: 45.0 , **1: 45.7** , 2: 41.3 , 4: 37.7 |
| sigmoid_scale | 0.01: 44.3 , **0.02: 45.7** , 0.05: 45.7 , 0.1: 44.7 , 0.3: 44.7 |
| tau_v | 0.005: 46.0 , 0.02: 45.3 , **0.05: 45.7** , 0.1: 44.0 , 0.2: 43.0 , 0.5: 42.7 |
| tau_a | -inf: 45.7 , **0: 45.7** , 0.2: 45.3 , 0.4: 44.3 , 0.8: 41.7 |

**UnLoc**, selected setting gives 54.7% against vision 49.7%.

| scalar | values and recall @1 m |
|---|---|
| weight | 0.25: 54.0 , **0.5: 54.7** , 1: 54.3 , 2: 51.3 , 4: 50.3 |
| sigmoid_scale | 0.01: 54.7 , **0.02: 54.7** , 0.05: 54.3 , 0.1: 53.7 , 0.3: 53.3 |
| tau_v | 0.005: 51.7 , 0.02: 52.3 , 0.05: 53.3 , 0.1: 55.7 , **0.2: 54.7** , 0.5: 54.3 |
| tau_a | **-inf: 54.7** , 0: 55.3 , 0.2: 54.3 , 0.4: 54.0 , 0.8: 52.3 |

The selected value is marked. A gain that survives across a range of each scalar is a property of the method; one that appears only at the selected value would be a fit, and we would have to say so.

