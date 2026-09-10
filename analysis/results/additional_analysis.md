# Additional analysis

Primary backbone `unlocSTFT`, 300 held-out queries on `replica_g`. Query acoustics are the furnished scan; candidates are rendered from the floorplan alone. Every parameter was chosen on `replica_f`.


Vision alone 49.7% at 1 m, with acoustic verification 54.7%.


## A. When does acoustic verification help?

Queries are binned by visual ambiguity, the log-odds between the two strongest hypotheses. Low means the backbone cannot separate its own top two candidates.

| ambiguity | n | vision | + acoustic | repairs | breaks | net |
|---|---|---|---|---|---|---|
| -0.000-0.011 | 60 | 26.7% | 31.7% | 8 | 5 | **+5.0** |
| 0.011-0.037 | 60 | 33.3% | 43.3% | 11 | 5 | **+10.0** |
| 0.037-0.096 | 60 | 33.3% | 43.3% | 7 | 1 | **+10.0** |
| 0.096-0.195 | 60 | 63.3% | 63.3% | 4 | 4 | **+0.0** |
| 0.195-0.805 | 60 | 91.7% | 91.7% | 0 | 0 | **+0.0** |

## B. Can sound choose between two visually plausible hypotheses?

A forced binary choice. For each query we take vision's two strongest hypotheses and keep the *decidable* pairs, those where exactly one lies within 1 m of the true pose, so a coin flip scores 50%. This measures the task the method performs, unlike acoustic-only recall over a whole floorplan, which measures a task it never performs.

| subset | n | acoustic picks correctly | 95% CI | vision picks correctly |
|---|---|---|---|---|
| all decidable pairs | 193 | **67.9%** | [61.0, 74.1] | 77.2% |
| visually close (margin < 0.104) | 96 | **66.7%** | [56.8, 75.3] | 60.4% |
| spatially separated by > 2 m | 169 | **69.2%** | [61.9, 75.7] | 75.1% |
| close and separated | 90 | **67.8%** | [57.6, 76.5] | 60.0% |

Chance is 50%. An interval whose lower bound clears 50 is the cleanest statement of the paper's core ability that this data can produce; one that straddles 50 would say the method works only through the visual prior it is attached to.


## D. Is the shortlist or the verifier the bottleneck?

The oracle picks the hypothesis nearest truth and is not a method. The gap between it and the achieved number is what better acoustic discrimination could still buy at that K.

| K | oracle @1 m | achieved @1 m | unused headroom |
|---|---|---|---|
| 1 | 49.7% | 49.7% | 0.0 |
| 2 | 64.3% | 50.0% | 14.3 |
| 3 | 75.7% | 51.0% | 24.7 |
| 5 | 86.3% | 52.3% | 34.0 |
| 10 | 93.7% | 54.7% | 39.0 |

## E. What limits the acoustic evidence?

The only change between the two rows is the mesh the *query* was rendered on. Candidates are floorplan-only in both. This is a control, not a method: a deployment cannot render furniture it does not know about.

| query geometry | acoustic alone @1 m | median GT rank | with verification @1 m |
|---|---|---|---|
| furnished (the real setting) | 20.7% | 192 | 54.7% |
| matched to the candidates | 94.3% | 1 | 65.7% |

## F. Is the gain actually acoustic?

Two permutation controls. `within query` keeps every acoustic score but reassigns which hypothesis it belongs to; `across queries` gives each query another query's acoustic evidence. Both leave the score distribution exactly as it was and destroy only the correspondence between a recording and the pose it was recorded at.

| audio | recall @1 m | gain over vision | 95% CI |
|---|---|---|---|
| as recorded | 54.7% | +5.0 | [+0.7, +9.3] |
| permuted within query | 42.1% | -7.5 | [-9.9, -5.2] |
| swapped across queries | 45.3% | -4.4 | [-6.5, -2.3] |

If a permuted control retained the gain, the fusion would be exploiting something about the score distribution rather than the recording, and nothing else in this document would mean anything.


## G. Does sound complement visual failure?

Scene-level, both backbones, so the axis is not a property of one model.

| backbone | scene | vision @1 m | + acoustic | gain |
|---|---|---|---|---|
| unlocSTFT | apartment_2 | 24.0% | 39.0% | +15.0 |
| unlocSTFT | frl_apartment_5 | 76.0% | 74.0% | -2.0 |
| unlocSTFT | office_4 | 49.0% | 51.0% | +2.0 |
| f3STFT | apartment_2 | 15.0% | 28.0% | +13.0 |
| f3STFT | frl_apartment_5 | 76.0% | 73.0% | -3.0 |
| f3STFT | office_4 | 22.0% | 36.0% | +14.0 |

Correlation between a scene's visual recall and the acoustic gain there: **-0.97** over 6 scene-backbone pairs. A strong negative value says sound is useful exactly where vision fails, which is a different and stronger statement than sound being better in acoustically easy rooms.


## I. Does the acoustic score know when it is wrong?

Reported because it is negative and because it explains the design: the gate leans on visual ambiguity precisely because no acoustic self-confidence measure predicts its own correctness.

| signal | acoustic pick correct | repairs vision | breaks vision |
|---|---|---|---|
| acoustic margin among hypotheses | 0.608 | 0.583 | 0.377 |
| relative evidence of the top hypothesis | 0.635 | 0.579 | 0.443 |
| visual ambiguity | 0.497 | 0.530 | 0.803 |

AUROC 0.5 is chance.

