# Additional analysis

Primary backbone `unlocSTFT`, 300 held-out queries on `replica_g`. Query acoustics are the furnished scan; candidates are rendered from the floorplan alone. Every parameter was chosen on `replica_f`.


Vision alone 49.7% at 1 m, with acoustic verification 50.7%.


## A. When does acoustic verification help?

Queries are binned by visual ambiguity, the log-odds between the two strongest hypotheses. Low means the backbone cannot separate its own top two candidates.

| ambiguity | n | vision | + acoustic | repairs | breaks | net |
|---|---|---|---|---|---|---|
| -0.000-0.011 | 60 | 26.7% | 31.7% | 13 | 10 | **+5.0** |
| 0.011-0.037 | 60 | 33.3% | 43.3% | 13 | 7 | **+10.0** |
| 0.037-0.096 | 60 | 33.3% | 38.3% | 13 | 10 | **+5.0** |
| 0.096-0.195 | 60 | 63.3% | 48.3% | 8 | 17 | **-15.0** |
| 0.195-0.805 | 60 | 91.7% | 91.7% | 1 | 1 | **+0.0** |

## B. Can sound choose between two visually plausible hypotheses?

A forced binary choice. For each query we take vision's two strongest hypotheses and keep the *decidable* pairs, those where exactly one lies within 1 m of the true pose, so a coin flip scores 50%. This measures the task the method performs, unlike acoustic-only recall over a whole floorplan, which measures a task it never performs.

| subset | n | acoustic picks correctly | 95% CI | vision picks correctly |
|---|---|---|---|---|
| all decidable pairs | 193 | **65.8%** | [58.9, 72.1] | 77.2% |
| visually close (margin < 0.104) | 96 | **63.5%** | [53.6, 72.5] | 60.4% |
| spatially separated by > 2 m | 169 | **67.5%** | [60.1, 74.1] | 75.1% |
| close and separated | 90 | **66.7%** | [56.4, 75.5] | 60.0% |

Chance is 50%. An interval whose lower bound clears 50 is the cleanest statement of the paper's core ability that this data can produce; one that straddles 50 would say the method works only through the visual prior it is attached to.


## D. Is the shortlist or the verifier the bottleneck?

The oracle picks the hypothesis nearest truth and is not a method. The gap between it and the achieved number is what better acoustic discrimination could still buy at that K.

| K | oracle @1 m | achieved @1 m | unused headroom |
|---|---|---|---|
| 1 | 49.7% | 49.7% | 0.0 |
| 2 | 64.3% | 47.0% | 17.3 |
| 3 | 75.7% | 49.0% | 26.7 |
| 5 | 86.3% | 50.0% | 36.3 |
| 10 | 93.7% | 50.7% | 43.0 |

## E. What limits the acoustic evidence?

The only change between the two rows is the mesh the *query* was rendered on. Candidates are floorplan-only in both. This is a control, not a method: a deployment cannot render furniture it does not know about.

| query geometry | acoustic alone @1 m | median GT rank | with verification @1 m |
|---|---|---|---|
| furnished (the real setting) | 20.7% | 192 | 50.7% |
| matched to the candidates | 94.3% | 1 | 79.7% |

## F. Is the gain actually acoustic?

Two permutation controls. `within query` keeps every acoustic score but reassigns which hypothesis it belongs to; `across queries` gives each query another query's acoustic evidence. Both leave the score distribution exactly as it was and destroy only the correspondence between a recording and the pose it was recorded at.

| audio | recall @1 m | gain over vision | 95% CI |
|---|---|---|---|
| as recorded | 50.7% | +1.0 | [-5.3, +7.3] |
| permuted within query | 32.7% | -17.0 | [-19.6, -14.3] |
| swapped across queries | 38.4% | -11.3 | [-13.8, -8.8] |

If a permuted control retained the gain, the fusion would be exploiting something about the score distribution rather than the recording, and nothing else in this document would mean anything.


## G. Does sound complement visual failure?

Scene-level, both backbones, so the axis is not a property of one model.

| backbone | scene | vision @1 m | + acoustic | gain |
|---|---|---|---|---|
| unlocSTFT | apartment_2 | 24.0% | 43.0% | +19.0 |
| unlocSTFT | frl_apartment_5 | 76.0% | 66.0% | -10.0 |
| unlocSTFT | office_4 | 49.0% | 43.0% | -6.0 |
| f3STFT | apartment_2 | 15.0% | 25.0% | +10.0 |
| f3STFT | frl_apartment_5 | 76.0% | 56.0% | -20.0 |
| f3STFT | office_4 | 22.0% | 21.0% | -1.0 |

Correlation between a scene's visual recall and the acoustic gain there: **-0.85** over 6 scene-backbone pairs. A strong negative value says sound is useful exactly where vision fails, which is a different and stronger statement than sound being better in acoustically easy rooms.


## I. Does the acoustic score know when it is wrong?

Reported because it is negative and because it explains the design: the gate leans on visual ambiguity precisely because no acoustic self-confidence measure predicts its own correctness.

| signal | acoustic pick correct | repairs vision | breaks vision |
|---|---|---|---|
| acoustic margin among hypotheses | 0.566 | 0.593 | 0.460 |
| relative evidence of the top hypothesis | 0.592 | 0.592 | 0.392 |
| visual ambiguity | 0.521 | 0.482 | 0.787 |

AUROC 0.5 is chance.

