# Can acoustics resolve the strongest visual ambiguity?

<!-- tables:start -->
## Table 1. Main protocol: the two strongest visual hypotheses, exactly one within 1 m

| benchmark | backbone | subset | decidable samples | acoustic right [95% CI] | visual right [95% CI] | acoustic − visual [95% CI] |
|---|---|---|---|---|---|---|
| Replica | F3Loc mono | all decidable | 312 | 80.8 [76.3, 85.3] | 73.6 [68.6, 78.4] | +7.2 [+1.1, +13.3] |
| Replica | F3Loc mono | hardest fifth (Q1 of m_v) | 38 | 71.1 [55.3, 84.2] | 53.9 [38.2, 69.7] | +17.1 [-5.3, +38.2] |
| Replica | F3Loc mono | Q1–Q2 | 81 | 72.8 [63.0, 81.5] | 53.7 [43.2, 64.2] | +19.1 [+5.6, +32.7] |
| Replica | UnLoc | all decidable | 389 | 82.5 [78.7, 86.1] | 78.1 [74.0, 82.4] | +4.4 [-0.5, +9.4] |
| Replica | UnLoc | hardest fifth (Q1 of m_v) | 52 | 63.5 [50.0, 76.9] | 57.7 [44.2, 71.2] | +5.8 [-11.5, +24.0] |
| Replica | UnLoc | Q1–Q2 | 114 | 72.8 [64.9, 80.7] | 61.4 [52.6, 70.2] | +11.4 [+0.4, +21.9] |
| Replica | DisCo-FLoc RRP | all decidable | 319 | 82.1 [78.1, 86.2] | 75.4 [70.5, 80.1] | +6.7 [+0.9, +12.5] |
| Replica | DisCo-FLoc RRP | hardest fifth (Q1 of m_v) | 37 | 83.8 [70.3, 94.6] | 50.0 [33.8, 66.2] | +33.8 [+10.8, +56.8] |
| Replica | DisCo-FLoc RRP | Q1–Q2 | 75 | 76.0 [65.3, 85.3] | 51.3 [40.0, 62.0] | +24.7 [+9.3, +39.3] |
| Matterport3D | F3Loc mono | all decidable | 426 | 80.5 [76.5, 84.3] | 75.2 [71.1, 79.3] | +5.3 [-0.2, +10.8] |
| Matterport3D | F3Loc mono | hardest fifth (Q1 of m_v) | 48 | 77.1 [64.6, 87.5] | 59.4 [45.8, 72.9] | +17.7 [-1.0, +36.5] |
| Matterport3D | F3Loc mono | Q1–Q2 | 103 | 71.8 [63.1, 80.6] | 64.6 [55.3, 73.8] | +7.3 [-5.3, +19.4] |
| Matterport3D | UnLoc | all decidable | 551 | 81.1 [77.9, 84.4] | 78.8 [75.3, 82.0] | +2.4 [-2.4, +6.7] |
| Matterport3D | UnLoc | hardest fifth (Q1 of m_v) | 58 | 82.8 [72.4, 91.4] | 58.6 [46.5, 70.7] | +24.1 [+10.3, +37.9] |
| Matterport3D | UnLoc | Q1–Q2 | 125 | 77.6 [69.6, 84.8] | 57.6 [48.8, 66.4] | +20.0 [+9.6, +30.4] |
| Matterport3D | DisCo-FLoc RRP | all decidable | 333 | 75.4 [70.9, 79.9] | 64.7 [59.5, 69.7] | +10.7 [+3.9, +17.4] |
| Matterport3D | DisCo-FLoc RRP | hardest fifth (Q1 of m_v) | 30 | 56.7 [40.0, 73.3] | 48.3 [30.0, 65.0] | +8.3 [-13.3, +30.0] |
| Matterport3D | DisCo-FLoc RRP | Q1–Q2 | 84 | 70.2 [60.7, 79.8] | 42.3 [32.1, 53.0] | +28.0 [+13.7, +42.3] |
| Structured3D | F3Loc mono | all decidable | 208 | 86.5 [81.7, 90.9] | 65.1 [59.1, 71.2] | +21.4 [+13.7, +28.8] |
| Structured3D | F3Loc mono | hardest fifth (Q1 of m_v) | 38 | 84.2 [71.1, 94.7] | 48.7 [43.4, 52.6] | +35.5 [+22.4, +47.4] |
| Structured3D | F3Loc mono | Q1–Q2 | 56 | 82.1 [71.4, 91.1] | 50.9 [42.9, 58.9] | +31.2 [+17.0, +44.6] |
| Structured3D | UnLoc | all decidable | 252 | 89.3 [85.3, 92.9] | 66.5 [60.9, 72.0] | +22.8 [+16.5, +29.2] |
| Structured3D | UnLoc | hardest fifth (Q1 of m_v) | 26 | 84.6 [69.2, 96.2] | 50.0 [50.0, 50.0] | +34.6 [+19.2, +46.2] |
| Structured3D | UnLoc | Q1–Q2 | 64 | 90.6 [82.8, 96.9] | 53.9 [44.5, 63.3] | +36.7 [+25.8, +47.7] |
| Structured3D | DisCo-FLoc RRP | all decidable | 192 | 89.1 [84.4, 93.2] | 59.6 [53.1, 66.1] | +29.4 [+21.4, +37.2] |
| Structured3D | DisCo-FLoc RRP | hardest fifth (Q1 of m_v) | 26 | 84.6 [69.2, 96.2] | 55.8 [50.0, 61.5] | +28.8 [+13.5, +42.3] |
| Structured3D | DisCo-FLoc RRP | Q1–Q2 | 53 | 84.9 [75.5, 94.3] | 51.9 [41.5, 62.3] | +33.0 [+17.9, +48.1] |

## Table 2. Supplementary protocol: all correct × incorrect pairs among the ten, samples weighted equally

| benchmark | backbone | subset | samples | pairs | acoustic right [95% CI] | visual right [95% CI] |
|---|---|---|---|---|---|---|
| Replica | F3Loc mono | all | 517 | 4891 | 83.7 [81.9, 85.6] | 78.5 [76.1, 80.7] |
| Replica | F3Loc mono | Q1 of m_v | 96 | 889 | 81.9 [77.7, 86.0] | 68.8 [63.2, 74.1] |
| Replica | UnLoc | all | 557 | 5156 | 87.9 [86.3, 89.4] | 84.9 [83.0, 86.8] |
| Replica | UnLoc | Q1 of m_v | 102 | 959 | 81.9 [77.4, 86.1] | 73.6 [68.4, 78.5] |
| Replica | DisCo-FLoc RRP | all | 507 | 4828 | 87.3 [85.6, 88.8] | 80.4 [78.1, 82.7] |
| Replica | DisCo-FLoc RRP | Q1 of m_v | 81 | 764 | 81.8 [77.3, 86.0] | 71.5 [65.5, 77.2] |
| Matterport3D | F3Loc mono | all | 713 | 6620 | 80.0 [78.3, 81.7] | 78.0 [76.0, 80.0] |
| Matterport3D | F3Loc mono | Q1 of m_v | 111 | 1020 | 74.4 [69.5, 79.3] | 67.4 [61.3, 73.6] |
| Matterport3D | UnLoc | all | 757 | 6953 | 81.8 [80.2, 83.4] | 84.9 [83.2, 86.5] |
| Matterport3D | UnLoc | Q1 of m_v | 114 | 1040 | 78.4 [73.6, 82.8] | 74.2 [69.1, 79.1] |
| Matterport3D | DisCo-FLoc RRP | all | 626 | 5861 | 79.5 [77.7, 81.3] | 71.9 [69.6, 74.2] |
| Matterport3D | DisCo-FLoc RRP | Q1 of m_v | 87 | 790 | 72.0 [66.4, 77.7] | 57.7 [50.5, 64.8] |
| Structured3D | F3Loc mono | all | 416 | 3856 | 91.1 [89.4, 92.7] | 71.9 [69.2, 74.6] |
| Structured3D | F3Loc mono | Q1 of m_v | 88 | 806 | 94.1 [91.7, 96.2] | 64.2 [57.8, 70.6] |
| Structured3D | UnLoc | all | 462 | 4291 | 92.4 [91.1, 93.7] | 73.5 [70.6, 76.2] |
| Structured3D | UnLoc | Q1 of m_v | 87 | 804 | 94.0 [91.5, 96.2] | 57.2 [50.8, 63.5] |
| Structured3D | DisCo-FLoc RRP | all | 407 | 3803 | 90.6 [89.0, 92.2] | 72.0 [69.4, 74.6] |
| Structured3D | DisCo-FLoc RRP | Q1 of m_v | 76 | 691 | 93.1 [90.4, 95.8] | 66.6 [60.5, 72.4] |

<!-- tables:end -->

## Reading the tables

Generated by `src/top2_disambiguation.py`; this section is by hand.

**Main protocol (Table 1).** The two hypotheses vision ranks highest after
NMS, on the samples where exactly one of them is within 1 m of the truth:
312–551 decidable samples per benchmark and backbone on the furnished
benchmarks, 192–252 on Structured3D. *Acoustic right* is whether the acoustic
evidence of each hypothesis on its own, α, ranks the correct one first; no
visual prior, gate or fused score enters. *Visual right* is whether vision's
own top-1 is the correct one. A tie counts as half. One sample is one pair, so
the bootstrap over samples is also a pair-level interval. "Hardest fifth" is
the decidable samples whose m_v falls in the most ambiguous quintile of the
whole test set (equal-count groups by rank, an analysis-only boundary).

**Sound separates the two places vision confuses most.** On all decidable
samples the acoustic ordering is right 80.8–82.5 % on Replica and 75.4–81.1 %
on Matterport3D, against 73.6–78.1 % and 64.7–78.8 % for vision; on the
hardest fifth the gap opens: DisCo-FLoc on Replica 83.8 against 50.0
(+33.8, [10.8, 56.8]), UnLoc on Matterport3D 82.8 against 58.6 (+24.1,
[10.3, 37.9]), F3Loc on Matterport3D 77.1 against 59.4. The acoustic ordering leads in
all nine hardest-fifth cells, by 6 to 36 points, and the lead clears the
interval in five of them; the weakest is UnLoc on Replica (63.5 against
57.7), where the hardest fifth is small (52 samples) and both are near
chance. On Structured3D the hardest fifth consists largely of exact
ties (two hypotheses with identical posterior), so vision is at 49–56 %
and sound at 84–85 %. The figure shows the same thing as a curve:
accuracy on the decidable samples admitted hardest-first; the acoustic curve
is flat, the visual one climbs only as easy samples are let in.

**Supplementary protocol (Table 2).** Every correct × incorrect pair among
the ten hypotheses, samples weighted equally over their pairs, bootstrap over
samples: 3,800–7,000 pairs per cell, so the intervals are tight, and the
same ordering holds in eight of nine cells (acoustic 80–92 % against visual
72–85 %); the one where vision is ahead on all pairs, UnLoc on Matterport3D
(81.8 against 84.9), reverses on the hardest fifth (78.4 against 74.2). It includes
easy negatives, which is why it is the supplementary and not the main test.
