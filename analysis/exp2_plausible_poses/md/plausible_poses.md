# Does acoustic verification resolve visual ambiguity?

<!-- tables:start -->
## Table 1. By number of visually plausible poses (v_(1) - v_k <= tau_v, tau_v from validation)

| benchmark | backbone | tau_v | N_plausible | samples | vision | +audio | gain [95% CI] | repair | break |
|---|---|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | 0.02 | 1 | 172 | 75.6 | 76.2 | +0.6 [-2.9, +4.1] | 5 | 4 |
| Replica | F3Loc mono | 0.02 | 2 | 91 | 37.4 | 49.5 | +12.1 [+2.2, +23.1] | 18 | 7 |
| Replica | F3Loc mono | 0.02 | 3 | 70 | 24.3 | 42.9 | +18.6 [+8.6, +30.0] | 15 | 2 |
| Replica | F3Loc mono | 0.02 | ≥4 | 267 | 18.4 | 32.6 | +14.2 [+8.6, +20.2] | 53 | 15 |
| Replica | UnLoc | 0.1 | 1 | 205 | 83.4 | 83.9 | +0.5 [-1.0, +2.4] | 2 | 1 |
| Replica | UnLoc | 0.1 | 2 | 102 | 47.1 | 58.8 | +11.8 [+2.9, +20.6] | 18 | 6 |
| Replica | UnLoc | 0.1 | 3 | 78 | 43.6 | 55.1 | +11.5 [+1.3, +21.8] | 15 | 6 |
| Replica | UnLoc | 0.1 | ≥4 | 215 | 24.2 | 39.5 | +15.3 [+9.8, +20.9] | 39 | 6 |
| Replica | DisCo-FLoc RRP | 0.02 | 1 | 139 | 83.5 | 84.2 | +0.7 [-2.2, +3.6] | 3 | 2 |
| Replica | DisCo-FLoc RRP | 0.02 | 2 | 87 | 56.3 | 66.7 | +10.3 [+0.0, +20.7] | 16 | 7 |
| Replica | DisCo-FLoc RRP | 0.02 | 3 | 79 | 31.6 | 49.4 | +17.7 [+6.3, +30.4] | 20 | 6 |
| Replica | DisCo-FLoc RRP | 0.02 | ≥4 | 295 | 18.0 | 33.9 | +15.9 [+10.5, +21.7] | 64 | 17 |
| Matterport3D | F3Loc mono | 0.05 | 1 | 72 | 75.0 | 75.0 | +0.0 [-4.2, +4.2] | 1 | 1 |
| Matterport3D | F3Loc mono | 0.05 | 2 | 92 | 57.6 | 57.6 | +0.0 [-7.6, +7.6] | 7 | 7 |
| Matterport3D | F3Loc mono | 0.05 | 3 | 74 | 41.9 | 51.4 | +9.5 [-1.4, +20.3] | 12 | 5 |
| Matterport3D | F3Loc mono | 0.05 | ≥4 | 722 | 25.2 | 33.2 | +8.0 [+4.8, +11.4] | 103 | 45 |
| Matterport3D | UnLoc | 0.05 | 1 | 344 | 79.4 | 80.2 | +0.9 [+0.0, +2.0] | 3 | 0 |
| Matterport3D | UnLoc | 0.05 | 2 | 195 | 46.2 | 53.8 | +7.7 [+3.1, +12.3] | 18 | 3 |
| Matterport3D | UnLoc | 0.05 | 3 | 102 | 34.3 | 45.1 | +10.8 [+3.9, +18.6] | 14 | 3 |
| Matterport3D | UnLoc | 0.05 | ≥4 | 319 | 11.6 | 16.6 | +5.0 [+2.8, +7.5] | 16 | 0 |
| Matterport3D | DisCo-FLoc RRP | 0.1 | 1 | 4 | 75.0 | 75.0 | +0.0 [+0.0, +0.0] | 0 | 0 |
| Matterport3D | DisCo-FLoc RRP | 0.1 | 2 | 10 | 40.0 | 50.0 | +10.0 [-20.0, +40.0] | 2 | 1 |
| Matterport3D | DisCo-FLoc RRP | 0.1 | 3 | 10 | 70.0 | 80.0 | +10.0 [+0.0, +30.0] | 1 | 0 |
| Matterport3D | DisCo-FLoc RRP | 0.1 | ≥4 | 936 | 21.9 | 32.1 | +10.1 [+7.6, +12.9] | 136 | 41 |
| Structured3D | F3Loc mono | 0.05 | 1 | 9 | 77.8 | 77.8 | +0.0 [+0.0, +0.0] | 0 | 0 |
| Structured3D | F3Loc mono | 0.05 | 2 | 21 | 42.9 | 66.7 | +23.8 [+9.5, +42.9] | 5 | 0 |
| Structured3D | F3Loc mono | 0.05 | 3 | 29 | 51.7 | 82.8 | +31.0 [+13.8, +48.3] | 9 | 0 |
| Structured3D | F3Loc mono | 0.05 | ≥4 | 499 | 20.8 | 45.3 | +24.4 [+20.0, +28.9] | 142 | 20 |
| Structured3D | UnLoc | 0.2 | 1 | 5 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 0 | 0 |
| Structured3D | UnLoc | 0.2 | 2 | 31 | 80.6 | 96.8 | +16.1 [+3.2, +29.0] | 5 | 0 |
| Structured3D | UnLoc | 0.2 | 3 | 22 | 68.2 | 81.8 | +13.6 [-4.5, +31.8] | 4 | 1 |
| Structured3D | UnLoc | 0.2 | ≥4 | 500 | 25.0 | 54.0 | +29.0 [+24.8, +33.4] | 156 | 11 |
| Structured3D | DisCo-FLoc RRP | 0.2 | ≥4 | 558 | 21.1 | 48.4 | +27.2 [+23.1, +31.4] | 166 | 14 |

## Table 2. Samples whose plausible set contains a correct hypothesis: which one is picked?

| benchmark | backbone | N_plausible | samples | GT in plausible set | vision picks it | audio alone picks it | fused picks it | fused − vision [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | 1 | 172 | 130 (76%) | 100.0 | 100.0 | 96.9 | -3.1 [-6.2, -0.8] |
| Replica | F3Loc mono | 2 | 91 | 54 (59%) | 63.0 | 81.5 | 77.8 | +14.8 [-1.9, +31.5] |
| Replica | F3Loc mono | 3 | 70 | 45 (64%) | 37.8 | 62.2 | 64.4 | +26.7 [+11.1, +42.2] |
| Replica | F3Loc mono | ≥4 | 267 | 202 (76%) | 24.3 | 45.5 | 43.1 | +18.8 [+11.4, +26.7] |
| Replica | UnLoc | 1 | 205 | 171 (83%) | 100.0 | 100.0 | 99.4 | -0.6 [-1.8, +0.0] |
| Replica | UnLoc | 2 | 102 | 71 (70%) | 67.6 | 77.5 | 77.5 | +9.9 [-1.4, +21.1] |
| Replica | UnLoc | 3 | 78 | 63 (81%) | 54.0 | 61.9 | 65.1 | +11.1 [-1.6, +23.8] |
| Replica | UnLoc | ≥4 | 215 | 163 (76%) | 31.9 | 52.1 | 52.1 | +20.2 [+12.9, +28.2] |
| Replica | DisCo-FLoc RRP | 1 | 139 | 116 (83%) | 100.0 | 100.0 | 98.3 | -1.7 [-4.3, +0.0] |
| Replica | DisCo-FLoc RRP | 2 | 87 | 64 (74%) | 76.6 | 82.8 | 81.2 | +4.7 [-7.8, +17.2] |
| Replica | DisCo-FLoc RRP | 3 | 79 | 62 (78%) | 40.3 | 62.9 | 58.1 | +17.7 [+3.2, +32.3] |
| Replica | DisCo-FLoc RRP | ≥4 | 295 | 193 (65%) | 27.5 | 52.8 | 51.3 | +23.8 [+15.5, +31.6] |
| Matterport3D | F3Loc mono | 1 | 72 | 54 (75%) | 100.0 | 100.0 | 98.1 | -1.9 [-5.6, +0.0] |
| Matterport3D | F3Loc mono | 2 | 92 | 60 (65%) | 88.3 | 83.3 | 85.0 | -3.3 [-15.0, +8.3] |
| Matterport3D | F3Loc mono | 3 | 74 | 49 (66%) | 63.3 | 71.4 | 69.4 | +6.1 [-8.2, +20.4] |
| Matterport3D | F3Loc mono | ≥4 | 722 | 484 (67%) | 37.6 | 43.8 | 49.4 | +11.8 [+7.0, +16.5] |
| Matterport3D | UnLoc | 1 | 344 | 273 (79%) | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] |
| Matterport3D | UnLoc | 2 | 195 | 134 (69%) | 67.2 | 75.4 | 77.6 | +10.4 [+4.5, +17.2] |
| Matterport3D | UnLoc | 3 | 102 | 68 (67%) | 51.5 | 63.2 | 67.6 | +16.2 [+5.8, +27.9] |
| Matterport3D | UnLoc | ≥4 | 319 | 148 (46%) | 25.0 | 42.6 | 35.8 | +10.8 [+6.1, +16.2] |
| Matterport3D | DisCo-FLoc RRP | 1 | 4 | 3 (75%) | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] |
| Matterport3D | DisCo-FLoc RRP | 2 | 10 | 6 (60%) | 66.7 | 83.3 | 83.3 | +16.7 [-33.3, +66.7] |
| Matterport3D | DisCo-FLoc RRP | 3 | 10 | 8 (80%) | 87.5 | 87.5 | 100.0 | +12.5 [+0.0, +37.5] |
| Matterport3D | DisCo-FLoc RRP | ≥4 | 936 | 601 (64%) | 34.1 | 39.4 | 49.8 | +15.6 [+11.6, +20.0] |
| Structured3D | F3Loc mono | 1 | 9 | 7 (78%) | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] |
| Structured3D | F3Loc mono | 2 | 21 | 10 (48%) | 90.0 | 90.0 | 90.0 | +0.0 [+0.0, +0.0] |
| Structured3D | F3Loc mono | 3 | 29 | 22 (76%) | 68.2 | 90.9 | 90.9 | +22.7 [+9.1, +40.9] |
| Structured3D | F3Loc mono | ≥4 | 499 | 353 (71%) | 29.5 | 64.9 | 64.0 | +34.6 [+28.3, +40.8] |
| Structured3D | UnLoc | 1 | 5 | 5 (100%) | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] |
| Structured3D | UnLoc | 2 | 31 | 30 (97%) | 83.3 | 96.7 | 100.0 | +16.7 [+6.7, +30.0] |
| Structured3D | UnLoc | 3 | 22 | 22 (100%) | 68.2 | 81.8 | 81.8 | +13.6 [-4.5, +31.8] |
| Structured3D | UnLoc | ≥4 | 500 | 399 (80%) | 31.3 | 68.4 | 67.2 | +35.8 [+30.6, +41.1] |
| Structured3D | DisCo-FLoc RRP | ≥4 | 558 | 407 (73%) | 29.0 | 66.3 | 66.3 | +37.3 [+31.9, +42.8] |

<!-- tables:end -->

## Reading the tables

Generated by `src/plausible_poses.py` from the headline protocol's test
tables (projection trained on the benchmark's own training rooms, fixed
structure, three scalars chosen on validation rooms, test rooms evaluated
once). This section is by hand.

**Definition.** Among the ten spatial hypotheses after NMS, a hypothesis is
visually plausible when v_(1) − v_k ≤ τ_v, with τ_v the visual threshold the
validation rooms chose for that benchmark and backbone, the same number the
gate uses (Replica UnLoc 0.1, Matterport3D UnLoc 0.05, and so on; first
column of Table 1). Nothing in the grouping is chosen on the test rooms.
N_plausible is the count, grouped as 1, 2, 3, ≥4. Intervals are paired
bootstraps over samples.

**Table 1: the gain lives where vision has more than one plausible pose.**
With a single plausible pose the acoustic term changes nothing (Replica UnLoc
+0.5, Matterport3D +0.9, both within noise) and breaks almost nothing (1 and
0 samples). From two plausible poses on, the gain is +5 to +19 points in eleven of the
twelve furnished cells (the exception is F3Loc on Matterport3D with two
plausible poses, +0.0, seven repairs against seven breaks) and +14 to +31
on Structured3D; everywhere else repairs outnumber breaks by about three to
one or more. This is the setting the title names, several
visually plausible poses, and the numbers say the method is inert outside it
and works inside it.

**Table 2: when the right pose is among the plausible ones, sound picks it.**
This is the direct test of the claim. On samples whose plausible set contains
a correct hypothesis, vision has already proposed the right place and the
whole question is which of the plausible ones to choose. With two plausible
poses vision picks the right one 67 % of the time on Replica and Matterport3D
(a coin toss made slightly better by the posterior); the fused rule picks it
77–78 %, and the acoustic evidence *alone*, choosing among the plausible ones
with no visual prior, 75–77 %. With four or more plausible poses vision is at
25–32 %, the fused rule at 36–52 %, sound alone at 43–52 % (Replica UnLoc
+20.2 points, [12.9, 28.2]; Matterport3D UnLoc +10.8, [6.1, 16.2]). On
Structured3D, where the query and the candidates share geometry, sound alone
resolves four-way ambiguity 68 % of the time against vision's 31 %. The same
holds for F3Loc and DisCo-FLoc, with one exception, F3Loc on Matterport3D
with two plausible poses, where vision is already at 88 % and neither sound
nor the fused rule improves on it; `data/plausible_poses.json` has every
cell.

**What this does not show.** The group sizes depend on τ_v, which differs by
backbone because the posteriors sit on different scales; DisCo-FLoc on
Matterport3D and Structured3D has nearly every sample in the ≥4 group. The
"GT in plausible set" fraction (46–83 % on the furnished benchmarks) is the
share of ambiguous samples the method can help at all; the rest are capped by
the shortlist, which `../exp4_formulation` measures.
