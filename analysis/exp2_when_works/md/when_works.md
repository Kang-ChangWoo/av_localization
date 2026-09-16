# When does acoustic verification help?

<!-- tables:start -->
## Table 1. Outcome of every test query, recall at 1 m

| benchmark | backbone | queries | vision | with sound | repaired | regressed | both wrong | both right |
|---|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | 600 | 38.3 | 48.8 | 91 | 28 | 279 | 202 |
| Replica | UnLoc | 600 | 50.8 | 60.0 | 74 | 19 | 221 | 286 |
| Replica | DisCo-FLoc RRP | 600 | 40.5 | 52.3 | 103 | 32 | 254 | 211 |
| Matterport3D | F3Loc mono | 960 | 33.3 | 40.1 | 123 | 58 | 517 | 262 |
| Matterport3D | UnLoc | 960 | 45.3 | 50.0 | 51 | 6 | 474 | 429 |
| Matterport3D | DisCo-FLoc RRP | 960 | 22.8 | 32.9 | 139 | 42 | 602 | 177 |
| Structured3D | F3Loc mono | 558 | 24.2 | 48.6 | 156 | 20 | 267 | 115 |
| Structured3D | UnLoc | 558 | 30.5 | 57.9 | 165 | 12 | 223 | 158 |
| Structured3D | DisCo-FLoc RRP | 558 | 21.1 | 48.4 | 166 | 14 | 274 | 104 |

## Table 2. Per scene, UnLoc: where the gain is (sorted by gain)

| benchmark | scene | queries | cells | vision | acoustic alone | with sound | gain |
|---|---|---|---|---|---|---|---|
| Replica | office_4 | 200 | 2734 | 45.5 | 33.5 | 56.5 | +11.0 |
| Replica | apartment_2 | 200 | 5443 | 29.5 | 20.5 | 40.0 | +10.5 |
| Replica | frl_apartment_5 | 200 | 5344 | 77.5 | 54.0 | 83.5 | +6.0 |
| Matterport3D | gTV8FGcVJC9_f4 | 80 | 16432 | 37.5 | 8.8 | 45.0 | +7.5 |
| Matterport3D | gTV8FGcVJC9_f5 | 80 | 5639 | 28.7 | 20.0 | 35.0 | +6.2 |
| Matterport3D | sT4fr6TAbpF_f0 | 80 | 23153 | 45.0 | 13.8 | 51.2 | +6.2 |
| Matterport3D | EDJbREhghzL_f1 | 80 | 10844 | 28.7 | 11.2 | 33.8 | +5.0 |
| Matterport3D | q9vSo1VnCiC_f0 | 80 | 24310 | 56.2 | 18.8 | 61.3 | +5.0 |
| Matterport3D | EDJbREhghzL_f0 | 80 | 9255 | 67.5 | 20.0 | 72.5 | +5.0 |
| Matterport3D | uNb9QFRL6hY_f1 | 80 | 15528 | 51.2 | 2.5 | 55.0 | +3.8 |
| Matterport3D | 8WUmhLawc2A_f0 | 80 | 17809 | 58.8 | 20.0 | 62.5 | +3.7 |
| Matterport3D | gTV8FGcVJC9_f0 | 80 | 10601 | 26.2 | 21.2 | 30.0 | +3.7 |
| Matterport3D | gTV8FGcVJC9_f2 | 80 | 7817 | 46.2 | 11.2 | 50.0 | +3.7 |
| Matterport3D | pLe4wQe7qrG_f0 | 80 | 4704 | 75.0 | 30.0 | 78.8 | +3.7 |
| Matterport3D | Z6MFQCViBuw_f0 | 80 | 56343 | 22.5 | 2.5 | 25.0 | +2.5 |
| Structured3D | scene_03379 | 18 | 5101 | 5.6 | 33.3 | 61.1 | +55.6 |
| Structured3D | scene_03290 | 17 | 3258 | 29.4 | 76.5 | 70.6 | +41.2 |
| Structured3D | scene_03346 | 20 | 3384 | 45.0 | 80.0 | 85.0 | +40.0 |
| Structured3D | scene_03391 | 20 | 9036 | 30.0 | 65.0 | 70.0 | +40.0 |
| Structured3D | scene_03399 | 19 | 4298 | 21.1 | 68.4 | 57.9 | +36.8 |
| Structured3D | scene_03383 | 17 | 3906 | 52.9 | 64.7 | 88.2 | +35.3 |
| Structured3D | scene_03447 | 20 | 4693 | 30.0 | 75.0 | 65.0 | +35.0 |
| Structured3D | scene_03363 | 12 | 3328 | 33.3 | 58.3 | 66.7 | +33.3 |
| Structured3D | scene_03267 | 20 | 6875 | 25.0 | 50.0 | 55.0 | +30.0 |
| Structured3D | scene_03316 | 20 | 6507 | 10.0 | 35.0 | 40.0 | +30.0 |
| Structured3D | scene_03253 | 20 | 7052 | 20.0 | 75.0 | 50.0 | +30.0 |
| Structured3D | scene_03386 | 20 | 6811 | 45.0 | 60.0 | 75.0 | +30.0 |
| Structured3D | scene_03422 | 20 | 7985 | 30.0 | 65.0 | 60.0 | +30.0 |
| Structured3D | scene_03460 | 20 | 3511 | 45.0 | 75.0 | 75.0 | +30.0 |
| Structured3D | scene_03478 | 20 | 5502 | 5.0 | 45.0 | 35.0 | +30.0 |
| Structured3D | scene_03437 | 14 | 7408 | 21.4 | 35.7 | 50.0 | +28.6 |
| Structured3D | scene_03413 | 20 | 7643 | 20.0 | 70.0 | 45.0 | +25.0 |
| Structured3D | scene_03474 | 20 | 10173 | 25.0 | 45.0 | 50.0 | +25.0 |
| Structured3D | scene_03483 | 20 | 5444 | 35.0 | 65.0 | 60.0 | +25.0 |
| Structured3D | scene_03440 | 14 | 3297 | 35.7 | 64.3 | 57.1 | +21.4 |
| Structured3D | scene_03250 | 20 | 5252 | 25.0 | 55.0 | 45.0 | +20.0 |
| Structured3D | scene_03259 | 20 | 6780 | 45.0 | 60.0 | 65.0 | +20.0 |
| Structured3D | scene_03367 | 20 | 4221 | 30.0 | 65.0 | 50.0 | +20.0 |
| Structured3D | scene_03432 | 20 | 7506 | 20.0 | 55.0 | 40.0 | +20.0 |
| Structured3D | scene_03258 | 20 | 7557 | 40.0 | 55.0 | 60.0 | +20.0 |
| Structured3D | scene_03461 | 20 | 8869 | 25.0 | 50.0 | 40.0 | +15.0 |
| Structured3D | scene_03479 | 20 | 7249 | 40.0 | 55.0 | 55.0 | +15.0 |
| Structured3D | scene_03400 | 20 | 6115 | 55.0 | 55.0 | 70.0 | +15.0 |
| Structured3D | scene_03310 | 7 | 4514 | 71.4 | 71.4 | 85.7 | +14.3 |
| Structured3D | scene_03319 | 20 | 4790 | 25.0 | 55.0 | 35.0 | +10.0 |

## Table 3. Recall by visual-ambiguity quintile, UnLoc (Q1 most ambiguous)

| benchmark | quintile | log-odds range | n | vision | with sound | gain | gate open |
|---|---|---|---|---|---|---|---|
| Replica | Q1 | 0.000–0.011 | 120 | 25.0 | 40.0 | +15.0 | 100% |
| Replica | Q2 | 0.011–0.034 | 120 | 33.3 | 46.7 | +13.3 | 100% |
| Replica | Q3 | 0.034–0.079 | 120 | 39.2 | 55.0 | +15.8 | 100% |
| Replica | Q4 | 0.079–0.191 | 120 | 63.3 | 65.0 | +1.7 | 100% |
| Replica | Q5 | 0.191–0.845 | 120 | 93.3 | 93.3 | +0.0 | 56% |
| Matterport3D | Q1 | 0.000–0.007 | 192 | 17.7 | 27.6 | +9.9 | 100% |
| Matterport3D | Q2 | 0.007–0.019 | 192 | 19.8 | 25.0 | +5.2 | 100% |
| Matterport3D | Q3 | 0.019–0.044 | 192 | 37.0 | 42.7 | +5.7 | 100% |
| Matterport3D | Q4 | 0.044–0.099 | 192 | 60.9 | 63.5 | +2.6 | 100% |
| Matterport3D | Q5 | 0.099–0.446 | 192 | 91.1 | 91.1 | +0.0 | 82% |
| Structured3D | Q2 | 0.000–0.006 | 223 | 14.8 | 48.4 | +33.6 | 100% |
| Structured3D | Q3 | 0.006–0.016 | 112 | 24.1 | 53.6 | +29.5 | 100% |
| Structured3D | Q4 | 0.016–0.040 | 111 | 27.0 | 52.3 | +25.2 | 100% |
| Structured3D | Q5 | 0.040–0.264 | 112 | 71.4 | 86.6 | +15.2 | 100% |

## Table 4. Forced choice between the correct hypothesis and an incorrect one

| benchmark | backbone | stratum | pairs | acoustic right | visual right |
|---|---|---|---|---|---|
| Replica | F3Loc mono | ambiguous | 1442 | 81.8 | 70.9 |
| Replica | F3Loc mono | middle | 1533 | 79.8 | 74.1 |
| Replica | F3Loc mono | confident | 1638 | 89.6 | 92.9 |
| Replica | F3Loc mono | all | 4613 | 83.9 | 79.8 |
| Replica | UnLoc | ambiguous | 1528 | 83.1 | 74.1 |
| Replica | UnLoc | middle | 1703 | 86.8 | 83.8 |
| Replica | UnLoc | confident | 1758 | 93.4 | 97.2 |
| Replica | UnLoc | all | 4989 | 88.0 | 85.5 |
| Replica | DisCo-FLoc RRP | ambiguous | 1327 | 83.0 | 68.8 |
| Replica | DisCo-FLoc RRP | middle | 1477 | 85.8 | 79.4 |
| Replica | DisCo-FLoc RRP | confident | 1720 | 92.2 | 93.9 |
| Replica | DisCo-FLoc RRP | all | 4524 | 87.4 | 81.8 |
| Matterport3D | F3Loc mono | ambiguous | 1802 | 77.5 | 69.3 |
| Matterport3D | F3Loc mono | middle | 2117 | 78.8 | 76.8 |
| Matterport3D | F3Loc mono | confident | 2469 | 83.5 | 87.0 |
| Matterport3D | F3Loc mono | all | 6388 | 80.2 | 78.6 |
| Matterport3D | UnLoc | ambiguous | 1803 | 76.3 | 73.2 |
| Matterport3D | UnLoc | middle | 2278 | 81.3 | 83.2 |
| Matterport3D | UnLoc | confident | 2712 | 85.8 | 95.5 |
| Matterport3D | UnLoc | all | 6793 | 81.7 | 85.4 |
| Matterport3D | DisCo-FLoc RRP | ambiguous | 1490 | 78.3 | 61.8 |
| Matterport3D | DisCo-FLoc RRP | middle | 1959 | 78.7 | 72.4 |
| Matterport3D | DisCo-FLoc RRP | confident | 2152 | 81.0 | 80.6 |
| Matterport3D | DisCo-FLoc RRP | all | 5601 | 79.5 | 72.7 |
| Structured3D | F3Loc mono | ambiguous | 1240 | 91.0 | 58.1 |
| Structured3D | F3Loc mono | middle | 1155 | 90.0 | 70.0 |
| Structured3D | F3Loc mono | confident | 1333 | 92.2 | 80.4 |
| Structured3D | F3Loc mono | all | 3728 | 91.1 | 69.8 |
| Structured3D | UnLoc | ambiguous | 1283 | 94.5 | 55.7 |
| Structured3D | UnLoc | middle | 1320 | 91.9 | 71.3 |
| Structured3D | UnLoc | confident | 1536 | 91.7 | 86.2 |
| Structured3D | UnLoc | all | 4139 | 92.6 | 72.0 |
| Structured3D | DisCo-FLoc RRP | ambiguous | 1139 | 92.0 | 61.1 |
| Structured3D | DisCo-FLoc RRP | middle | 1166 | 91.6 | 66.3 |
| Structured3D | DisCo-FLoc RRP | confident | 1338 | 88.3 | 82.1 |
| Structured3D | DisCo-FLoc RRP | all | 3643 | 90.5 | 70.5 |

## Table 5. The acoustic score as the candidate set grows, UnLoc

| benchmark | K | acoustic alone among K | gated fusion over K | oracle over K |
|---|---|---|---|---|
| Replica | 1 | 50.8 | 50.8 | 50.8 |
| Replica | 2 | 53.7 | 51.0 | 65.0 |
| Replica | 3 | 54.5 | 53.7 | 76.2 |
| Replica | 5 | 54.3 | 56.3 | 85.2 |
| Replica | 10 | 50.8 | 60.0 | 92.8 |
| Replica | all cells | 36.0 | – | – |
| Matterport3D | 1 | 45.3 | 45.3 | 45.3 |
| Matterport3D | 2 | 46.7 | 45.3 | 57.5 |
| Matterport3D | 3 | 45.0 | 46.8 | 64.0 |
| Matterport3D | 5 | 39.6 | 48.2 | 69.5 |
| Matterport3D | 10 | 33.2 | 50.0 | 78.9 |
| Matterport3D | all cells | 15.0 | – | – |
| Structured3D | 1 | 30.5 | 30.5 | 30.5 |
| Structured3D | 2 | 41.0 | 41.2 | 45.9 |
| Structured3D | 3 | 44.1 | 45.0 | 52.7 |
| Structured3D | 5 | 51.1 | 50.2 | 65.1 |
| Structured3D | 10 | 56.5 | 57.9 | 82.8 |
| Structured3D | all cells | 59.3 | – | – |

<!-- tables:end -->

## Reading the tables

The tables above are generated by `src/when_works.py` from the test tables of
the headline protocol (projection trained on the benchmark's own rooms, fixed
structure, three scalars chosen on validation rooms, test rooms evaluated
once). This section is written by hand.

### Where the gain is (Table 2, `figs/A_gain_per_scene.png`)

Every Replica and Matterport3D room gains under every backbone except two
Matterport3D rooms at −1 point for one backbone each, which is noise at 80
queries. The gain is largest where vision is weakest: on Matterport3D it is
10–21 points in rooms where UnLoc or DisCo are below 30 %, and 3–5 points in
rooms above 55 %. Replica shows the same slope with three rooms. On
Structured3D, where query and candidates share the wall-only geometry, every
one of the thirty rooms gains, by 7 to 59 points.

### What happens to a query (Table 1)

Sound repairs far more queries than it breaks: on Replica 74–103 repairs
against 19–32 regressions per backbone, on Matterport3D 51–139 against 6–58,
on Structured3D 156–166 against 12–20. The largest group everywhere is
"both wrong": queries whose truth is not among the ten visual hypotheses, or
where sound cannot separate it from the others. That group, not the
regressions, is what caps the method.

### Ambiguity is the signal (Table 3, `figs/C_by_ambiguity.png`)

Binned by the log-odds between vision's two best hypotheses, the whole gain
sits in the ambiguous quintiles: Replica Q1 +15.0, Q2 +13.3, Q5 0.0;
Matterport3D Q1 +9.9, Q5 0.0. In the confident quintile vision is already at
91–93 % and the gate leaves it alone. This is the quantity the gate reads,
so the figure is also a check that the gate does what it is for. Structured3D
is the exception: with no furniture gap the acoustic evidence is strong enough
to add 15 points even in the most confident quintile, and the validation rooms
chose a gate that is open on every query.

### The evidence is real on the decision that matters (Table 4, `figs/D_pairwise.png`)

On a forced choice between the correct hypothesis and one incorrect one, the
acoustic ordering is right 80–88 % of the time on the furnished benchmarks and
90–93 % on Structured3D, and it is flat across ambiguity strata. The visual
ordering is strong where vision is confident (81–97 %) and weak where it is
ambiguous (56–74 %). The two cross exactly where the gate opens, which is the
empirical reason for gating on visual ambiguity rather than on anything
acoustic.

### Sound cannot search, it can help choose (Table 5, `figs/E_candidate_set.png`)

Given the top-K visual hypotheses and asked to choose alone, the acoustic
score gets worse as K grows on the furnished benchmarks (Replica 50.8 → 50.8,
Matterport3D 45.3 → 33.2 from K=1 to 10), and over the whole grid it reaches
36.0 % and 15.0 %. The gated fusion over the same K rises to 60.0 and 50.0.
On Structured3D the acoustic score alone is competitive at every K and
reaches 59.3 % over the whole grid, so there the method's constraint to the
visual shortlist is the thing that limits it, which is why the cell-product
rule (`../../feasible/results/CELL_product.md`) gains most on that benchmark.

### Samples (`figs/samples/`)

Rows drawn under the same protocol: the view, the truth, the fused pick on
the hypothesis discs, the acoustic score over every cell, the visual
posterior. They were chosen as the largest repairs per room with a view that
has content in it, so they are favourable by construction; the counts in
Table 1 say how common a repair is.
