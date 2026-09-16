# Robustness to a degrading camera

<!-- tables:start -->
## Table 1. UnLoc on Replica, recall at 1 m, nothing refit on the degraded queries

Policy: `ModeFusionConfig(vis_evidence='centre', ac_evidence='quantile', rule='continuous', weight=1.0, tau_v=0.1, tau_a=-2.0, sigmoid_scale=0.1, lam=1.0, ac_transform='standard')`. Columns A/B/C use the settings the clean validation rooms chose.

| degradation | vision | acoustic alone | hypothesis rule | gain [95% CI] | gate open | A inject | B reject | C product |
|---|---|---|---|---|---|---|---|---|
| clean | 50.7 | 36.0 | 60.0 | +9.3 [+6.3, +12.5] | 91% | 59.0 | 60.0 | 60.5 |
| blur σ=1 | 50.3 | 36.0 | 59.2 | +8.8 [+5.7, +12.2] | 93% | 59.3 | 59.2 | 59.5 |
| blur σ=2 | 48.0 | 36.0 | 56.8 | +8.8 [+5.5, +12.3] | 94% | 55.8 | 56.8 | 57.0 |
| blur σ=4 | 42.8 | 36.0 | 51.3 | +8.5 [+5.0, +12.0] | 96% | 53.0 | 51.3 | 53.7 |
| blur σ=8 | 28.0 | 36.0 | 37.8 | +9.8 [+6.5, +13.2] | 100% | 37.2 | 37.8 | 44.0 |
| blur σ=16 | 7.2 | 36.0 | 12.2 | +5.0 [+2.2, +7.7] | 100% | 16.7 | 12.2 | 25.5 |
| blur σ=32 | 2.8 | 36.0 | 6.3 | +3.5 [+1.8, +5.2] | 100% | 6.8 | 6.3 | 8.7 |
| dark ×0.75 | 50.5 | 36.0 | 57.5 | +7.0 [+3.8, +10.2] | 91% | 58.2 | 57.5 | 58.7 |
| dark ×0.5 | 47.2 | 36.0 | 55.3 | +8.2 [+4.8, +11.5] | 92% | 56.0 | 55.3 | 57.5 |
| dark ×0.25 | 44.2 | 36.0 | 53.0 | +8.8 [+5.5, +12.2] | 93% | 53.7 | 53.0 | 55.8 |
| dark ×0.1 | 38.5 | 36.0 | 49.2 | +10.7 [+7.2, +14.0] | 95% | 49.5 | 49.2 | 53.0 |
| dark ×0.05 | 27.8 | 36.0 | 37.8 | +10.0 [+7.0, +13.2] | 98% | 38.3 | 37.8 | 45.5 |
| dark ×0.02 | 3.3 | 36.0 | 12.7 | +9.3 [+6.7, +12.0] | 100% | 14.3 | 12.7 | 16.5 |
| noise σ=10 | 47.2 | 36.0 | 54.8 | +7.7 [+4.5, +11.0] | 92% | 55.8 | 54.8 | 57.0 |
| noise σ=25 | 44.3 | 36.0 | 53.8 | +9.5 [+6.2, +12.8] | 93% | 54.3 | 53.8 | 56.3 |
| noise σ=50 | 41.7 | 36.0 | 49.7 | +8.0 [+4.7, +11.3] | 96% | 50.3 | 49.7 | 54.2 |
| noise σ=100 | 23.3 | 36.0 | 33.5 | +10.2 [+6.7, +13.7] | 99% | 31.5 | 33.5 | 41.7 |
| noise σ=150 | 5.5 | 36.0 | 12.0 | +6.5 [+4.0, +9.0] | 100% | 13.5 | 12.0 | 21.2 |
| occlude 10% | 34.3 | 36.0 | 47.0 | +12.7 [+9.2, +16.2] | 98% | 46.5 | 47.0 | 53.2 |
| occlude 30% | 15.7 | 36.0 | 22.3 | +6.7 [+3.8, +9.5] | 100% | 22.5 | 22.3 | 32.3 |
| occlude 50% | 11.0 | 36.0 | 18.0 | +7.0 [+4.5, +9.7] | 100% | 17.5 | 18.0 | 24.0 |
| occlude 70% | 7.3 | 36.0 | 11.8 | +4.5 [+2.0, +7.2] | 100% | 13.3 | 11.8 | 15.5 |
| occlude 90% | 4.2 | 36.0 | 9.8 | +5.7 [+3.7, +7.8] | 100% | 11.0 | 9.8 | 11.3 |
| downscale 2× | 49.5 | 36.0 | 59.2 | +9.7 [+6.3, +13.0] | 92% | 58.2 | 59.2 | 59.8 |
| downscale 4× | 47.7 | 36.0 | 54.8 | +7.2 [+3.8, +10.5] | 93% | 55.7 | 54.8 | 57.7 |
| downscale 8× | 42.5 | 36.0 | 52.0 | +9.5 [+6.3, +12.7] | 96% | 53.3 | 52.0 | 55.3 |
| downscale 16× | 30.5 | 36.0 | 42.7 | +12.2 [+8.8, +15.5] | 99% | 42.3 | 42.7 | 47.5 |
| downscale 32× | 6.7 | 36.0 | 14.7 | +8.0 [+5.7, +10.5] | 100% | 16.0 | 14.7 | 26.5 |

## Table 2. The cell-product rule with λ chosen on clean validation (λ=0.2238), frozen

| degradation | vision | cell product | gain |
|---|---|---|---|
| clean | 50.8 | 63.5 | +12.7 |
| blur σ=1 | 50.5 | 60.7 | +10.2 |
| blur σ=2 | 47.2 | 57.0 | +9.8 |
| blur σ=4 | 43.0 | 54.7 | +11.7 |
| blur σ=8 | 27.3 | 42.0 | +14.7 |
| blur σ=16 | 7.0 | 20.5 | +13.5 |
| blur σ=32 | 3.2 | 8.0 | +4.8 |
| dark ×0.75 | 50.7 | 58.3 | +7.7 |
| dark ×0.5 | 47.5 | 57.7 | +10.2 |
| dark ×0.25 | 43.7 | 57.3 | +13.7 |
| dark ×0.1 | 38.3 | 52.8 | +14.5 |
| dark ×0.05 | 27.7 | 45.2 | +17.5 |
| dark ×0.02 | 3.3 | 13.7 | +10.3 |
| noise σ=10 | 47.2 | 58.8 | +11.7 |
| noise σ=25 | 44.5 | 57.0 | +12.5 |
| noise σ=50 | 40.8 | 55.7 | +14.8 |
| noise σ=100 | 23.3 | 39.2 | +15.8 |
| noise σ=150 | 4.8 | 16.3 | +11.5 |
| occlude 10% | 33.7 | 53.2 | +19.5 |
| occlude 30% | 15.7 | 25.7 | +10.0 |
| occlude 50% | 11.5 | 20.3 | +8.8 |
| occlude 70% | 7.0 | 14.2 | +7.2 |
| occlude 90% | 4.7 | 10.0 | +5.3 |
| downscale 2× | 49.5 | 60.2 | +10.7 |
| downscale 4× | 47.3 | 59.2 | +11.8 |
| downscale 8× | 43.0 | 55.2 | +12.2 |
| downscale 16× | 31.3 | 47.7 | +16.3 |
| downscale 32× | 6.7 | 20.3 | +13.7 |

## Table 3. Can visual collapse be detected and switched on? (from the collapse-switch study)

Indicator AUROC for 'shortlist covers the truth' against 'shortlist broken', per level; then the switch thresholds chosen on *degraded validation* queries and their effect on test.

| level | broken % | entropy | 1-pmax | -za_at_vis | 1-agree |
|---|---|---|---|---|---|
| clean | 7 | 0.58 | 0.85 | 0.76 | 0.63 |
| blur σ=1 | 9 | 0.57 | 0.81 | 0.74 | 0.61 |
| blur σ=2 | 9 | 0.53 | 0.81 | 0.75 | 0.63 |
| blur σ=4 | 10 | 0.43 | 0.76 | 0.72 | 0.62 |
| blur σ=8 | 22 | 0.46 | 0.73 | 0.63 | 0.60 |
| blur σ=16 | 52 | 0.47 | 0.76 | 0.62 | 0.53 |
| blur σ=32 | 70 | 0.73 | 0.75 | 0.66 | 0.52 |
| dark ×0.75 | 8 | 0.58 | 0.86 | 0.75 | 0.63 |
| dark ×0.5 | 10 | 0.62 | 0.86 | 0.78 | 0.65 |
| dark ×0.25 | 10 | 0.58 | 0.86 | 0.74 | 0.62 |
| dark ×0.1 | 12 | 0.53 | 0.84 | 0.76 | 0.61 |
| dark ×0.05 | 18 | 0.48 | 0.73 | 0.72 | 0.60 |
| dark ×0.02 | 54 | 0.62 | 0.78 | 0.63 | 0.49 |
| noise σ=10 | 10 | 0.62 | 0.86 | 0.75 | 0.63 |
| noise σ=25 | 11 | 0.57 | 0.85 | 0.74 | 0.61 |
| noise σ=50 | 14 | 0.45 | 0.77 | 0.75 | 0.62 |
| noise σ=100 | 24 | 0.50 | 0.74 | 0.65 | 0.60 |
| noise σ=150 | 54 | 0.61 | 0.78 | 0.65 | 0.51 |
| occlude 10% | 13 | 0.58 | 0.81 | 0.64 | 0.64 |
| occlude 30% | 29 | 0.57 | 0.80 | 0.59 | 0.54 |
| occlude 50% | 43 | 0.55 | 0.69 | 0.60 | 0.54 |
| occlude 70% | 54 | 0.61 | 0.69 | 0.62 | 0.49 |
| occlude 90% | 62 | 0.62 | 0.70 | 0.58 | 0.50 |
| downscale 2× | 9 | 0.63 | 0.86 | 0.77 | 0.64 |
| downscale 4× | 10 | 0.54 | 0.85 | 0.70 | 0.64 |
| downscale 8× | 9 | 0.51 | 0.81 | 0.75 | 0.61 |
| downscale 16× | 18 | 0.54 | 0.71 | 0.66 | 0.59 |
| downscale 32× | 45 | 0.48 | 0.67 | 0.64 | 0.55 |

Validation levels available: ['clean', 'blur σ=8', 'blur σ=16', 'dark ×0.05', 'dark ×0.02', 'noise σ=100', 'occlude 50%']

### Switch thresholds chosen on the pooled degraded validation queries

| indicator | switch to | val recall, no switch | with switch | threshold |
|---|---|---|---|---|
| entropy | acoustic | 28.0 | 29.0 | 0.914 |
| entropy | cellprod | 28.0 | 33.1 | 0.914 |
| pmax | acoustic | 28.0 | 32.0 | -0.001 |
| pmax | cellprod | 28.0 | 33.1 | -0.002 |
| za_at_vis | acoustic | 28.0 | 32.4 | -0.809 |
| za_at_vis | cellprod | 28.0 | 33.1 | -2.152 |
| agree | acoustic | 28.0 | 28.8 | -0.142 |
| agree | cellprod | 28.0 | 33.1 | -0.142 |

### Test levels with the chosen switches, frozen

| level | vision | acoustic | ours | C | entropy→acoustic | entropy→cellprod | pmax→acoustic | pmax→cellprod | za_at_vis→acoustic | za_at_vis→cellprod | agree→acoustic | agree→cellprod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 50.8 | 36.0 | 60.0 | 60.2 | 37.0 (98%) | 60.7 (98%) | 50.5 (45%) | 60.2 (94%) | 54.7 (32%) | 60.0 (88%) | 34.2 (87%) | 60.3 (87%) |
| blur σ=1 | 50.5 | 36.0 | 59.2 | 59.7 | 36.3 (98%) | 59.3 (98%) | 49.2 (48%) | 59.7 (96%) | 54.3 (33%) | 59.8 (88%) | 34.0 (88%) | 59.8 (88%) |
| blur σ=2 | 47.2 | 36.0 | 56.8 | 57.3 | 36.3 (98%) | 56.7 (98%) | 44.8 (56%) | 56.5 (98%) | 51.5 (33%) | 57.5 (88%) | 34.3 (90%) | 56.5 (90%) |
| blur σ=4 | 43.0 | 36.0 | 51.3 | 53.8 | 36.0 (97%) | 53.2 (97%) | 41.3 (62%) | 53.2 (98%) | 47.7 (42%) | 53.5 (91%) | 34.7 (92%) | 53.2 (92%) |
| blur σ=8 | 27.3 | 36.0 | 37.8 | 43.7 | 35.5 (96%) | 43.0 (96%) | 37.2 (73%) | 43.2 (98%) | 37.2 (54%) | 43.5 (95%) | 35.0 (96%) | 43.5 (96%) |
| blur σ=16 | 7.0 | 36.0 | 12.2 | 25.3 | 35.5 (92%) | 24.8 (92%) | 34.0 (75%) | 25.3 (94%) | 32.0 (80%) | 25.3 (100%) | 35.8 (100%) | 25.2 (100%) |
| blur σ=32 | 3.2 | 36.0 | 6.3 | 8.7 | 34.2 (86%) | 9.0 (86%) | 22.5 (50%) | 9.0 (87%) | 33.8 (89%) | 8.3 (99%) | 35.8 (100%) | 8.5 (100%) |
| dark ×0.75 | 50.7 | 36.0 | 57.5 | 58.2 | 36.7 (98%) | 58.0 (98%) | 49.8 (43%) | 57.5 (93%) | 53.2 (35%) | 58.2 (88%) | 34.0 (87%) | 58.2 (87%) |
| dark ×0.5 | 47.5 | 36.0 | 55.3 | 57.5 | 36.7 (98%) | 57.5 (98%) | 49.3 (44%) | 57.2 (93%) | 51.7 (35%) | 57.5 (88%) | 34.0 (88%) | 57.3 (88%) |
| dark ×0.25 | 43.7 | 36.0 | 53.0 | 56.0 | 36.0 (99%) | 55.8 (99%) | 46.0 (46%) | 56.2 (96%) | 48.8 (37%) | 56.3 (88%) | 34.0 (89%) | 55.8 (89%) |
| dark ×0.1 | 38.3 | 36.0 | 49.2 | 52.8 | 36.0 (100%) | 52.5 (100%) | 44.0 (52%) | 53.2 (99%) | 44.5 (46%) | 53.2 (92%) | 34.3 (90%) | 53.0 (90%) |
| dark ×0.05 | 27.7 | 36.0 | 37.8 | 45.3 | 36.5 (98%) | 45.3 (98%) | 38.3 (55%) | 45.3 (98%) | 39.5 (57%) | 45.3 (96%) | 34.5 (94%) | 45.2 (94%) |
| dark ×0.02 | 3.3 | 36.0 | 12.7 | 16.7 | 35.8 (96%) | 15.7 (96%) | 31.7 (64%) | 16.8 (100%) | 34.0 (83%) | 16.7 (100%) | 35.8 (100%) | 16.8 (100%) |
| noise σ=10 | 47.2 | 36.0 | 54.8 | 57.2 | 36.2 (98%) | 56.8 (98%) | 49.0 (43%) | 57.0 (94%) | 51.2 (35%) | 57.2 (88%) | 34.5 (87%) | 57.2 (87%) |
| noise σ=25 | 44.5 | 36.0 | 53.8 | 56.5 | 36.0 (100%) | 56.3 (100%) | 47.0 (46%) | 56.8 (96%) | 48.5 (39%) | 56.7 (90%) | 34.7 (88%) | 56.7 (88%) |
| noise σ=50 | 40.8 | 36.0 | 49.7 | 54.3 | 36.0 (100%) | 54.3 (100%) | 44.3 (53%) | 54.2 (99%) | 47.3 (46%) | 54.8 (93%) | 34.5 (92%) | 54.0 (92%) |
| noise σ=100 | 23.3 | 36.0 | 33.5 | 41.7 | 36.0 (100%) | 41.7 (100%) | 37.3 (63%) | 41.7 (100%) | 35.3 (59%) | 41.8 (97%) | 35.2 (95%) | 41.7 (95%) |
| noise σ=150 | 4.8 | 36.0 | 12.0 | 21.5 | 36.0 (100%) | 21.5 (100%) | 32.3 (76%) | 21.5 (100%) | 32.3 (83%) | 21.2 (100%) | 35.5 (99%) | 21.7 (99%) |
| occlude 10% | 33.7 | 36.0 | 47.0 | 53.5 | 36.5 (99%) | 53.3 (99%) | 40.8 (68%) | 53.2 (99%) | 42.3 (45%) | 54.0 (93%) | 34.7 (94%) | 53.2 (94%) |
| occlude 30% | 15.7 | 36.0 | 22.3 | 32.0 | 36.3 (99%) | 31.8 (99%) | 36.3 (82%) | 31.7 (99%) | 29.3 (62%) | 31.8 (98%) | 35.7 (99%) | 31.7 (99%) |
| occlude 50% | 11.5 | 36.0 | 18.0 | 24.0 | 36.0 (99%) | 23.5 (99%) | 34.0 (79%) | 23.3 (99%) | 35.2 (71%) | 24.0 (98%) | 35.8 (100%) | 24.0 (100%) |
| occlude 70% | 7.0 | 36.0 | 11.8 | 15.7 | 36.3 (98%) | 15.7 (98%) | 31.3 (67%) | 15.7 (98%) | 32.7 (79%) | 15.5 (100%) | 36.0 (100%) | 15.8 (100%) |
| occlude 90% | 4.7 | 36.0 | 9.8 | 11.2 | 34.5 (94%) | 11.0 (94%) | 23.0 (53%) | 10.8 (96%) | 32.7 (83%) | 11.2 (100%) | 35.8 (100%) | 11.0 (100%) |
| downscale 2× | 49.5 | 36.0 | 59.2 | 60.2 | 36.3 (98%) | 60.0 (98%) | 49.0 (46%) | 60.0 (95%) | 53.8 (34%) | 60.5 (88%) | 34.7 (88%) | 60.8 (88%) |
| downscale 4× | 47.3 | 36.0 | 54.8 | 58.0 | 36.5 (98%) | 57.5 (98%) | 44.7 (51%) | 57.3 (98%) | 50.8 (36%) | 58.2 (89%) | 34.5 (90%) | 57.3 (90%) |
| downscale 8× | 43.0 | 36.0 | 52.0 | 55.3 | 36.0 (97%) | 54.7 (97%) | 41.0 (59%) | 54.7 (98%) | 48.5 (43%) | 55.8 (90%) | 34.3 (91%) | 54.8 (91%) |
| downscale 16× | 31.3 | 36.0 | 42.7 | 47.5 | 35.8 (96%) | 47.0 (96%) | 39.7 (70%) | 46.8 (97%) | 41.5 (55%) | 47.7 (94%) | 34.3 (94%) | 46.8 (94%) |
| downscale 32× | 6.7 | 36.0 | 14.7 | 25.8 | 35.7 (93%) | 25.5 (93%) | 32.7 (74%) | 26.0 (96%) | 34.2 (74%) | 25.7 (99%) | 35.7 (99%) | 25.5 (99%) |

<!-- tables:end -->

## Reading the tables

Assembled by `src/robustness.py` from `../../feasible/results`; this section
is by hand.

**Nothing refit, the gain holds (Table 1).** With the projection and the
three scalars frozen at their clean-validation values, the query image is
corrupted at 28 levels across blur, darkness, noise, occlusion and
downscaling. The gain from sound is positive at every level, +3.5 to +12.7
with the lower interval above +1.8 everywhere, and it is largest in the
middle of the range (occlude 10 % +12.7, downscale 16× +12.2, noise σ=100 +10.2,
dark ×0.05 +10.0), where vision is at 20–40 % and its shortlist still covers the truth
often enough to be worth verifying.

**The shortlist is the ceiling.** Below roughly 25 % visual recall the fused
rule falls under the acoustic score alone (36.0 %), because a hypothesis rule
can only answer with a place vision proposed. The cell-product rule (column C in
Table 1 with the 16-point λ grid; Table 2 with the wide grid, 1–3 points
higher) does not have that ceiling in the same way and is ahead of the
hypothesis rule at every level, by under a point on clean queries and by
9–13 points in the mid-severe range (blur σ=16, noise σ=150, occlude 30 %,
downscale 32×), though it too stays below acoustic-alone once vision is
under 10 %.

**Collapse cannot be detected well enough to switch (Table 3).** Four
indicators of a broken shortlist, entropy and peak mass of the posterior, the
acoustic score at vision's own argmax, and the posterior mass near the
acoustic peak, separate covered from broken samples only moderately (AUROC
0.43–0.86). A switch to acoustic-alone with a threshold chosen on *degraded
validation* queries rescues the extreme levels, but the two usable indicators
(peak mass, acoustic score at vision's argmax) fire on 32–45 % of clean
queries and cost 5–10 points there, and the other two fire on nearly every
query; a switch to the cell product fires on
88–98 % and is simply the cell product. The honest statement is a limit: when
the camera is far gone, the acoustic score alone is the better answer, and
the system cannot tell from the inside when that is.
