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

## Table 3. Recall by visual-ambiguity quintile, UnLoc (equal-count groups of m_v; Q1 most ambiguous)

| benchmark | quintile | m_v range | ties at 0 | mean u_v | n | vision | with sound | gain | gate open |
|---|---|---|---|---|---|---|---|---|---|
| Replica | Q1 | 0.000–0.011 | 8 | 0.98 | 120 | 25.0 | 40.0 | +15.0 | 100% |
| Replica | Q2 | 0.011–0.033 | 0 | 0.97 | 120 | 33.3 | 46.7 | +13.3 | 100% |
| Replica | Q3 | 0.034–0.079 | 0 | 0.97 | 120 | 39.2 | 55.0 | +15.8 | 100% |
| Replica | Q4 | 0.079–0.191 | 0 | 0.96 | 120 | 63.3 | 65.0 | +1.7 | 100% |
| Replica | Q5 | 0.192–0.845 | 0 | 0.96 | 120 | 93.3 | 93.3 | +0.0 | 56% |
| Matterport3D | Q1 | 0.000–0.007 | 4 | 1.00 | 192 | 17.7 | 27.6 | +9.9 | 100% |
| Matterport3D | Q2 | 0.007–0.019 | 0 | 1.00 | 192 | 19.8 | 25.0 | +5.2 | 100% |
| Matterport3D | Q3 | 0.019–0.044 | 0 | 1.00 | 192 | 37.0 | 42.7 | +5.7 | 100% |
| Matterport3D | Q4 | 0.044–0.098 | 0 | 1.00 | 192 | 60.9 | 63.5 | +2.6 | 100% |
| Matterport3D | Q5 | 0.099–0.446 | 0 | 0.99 | 192 | 91.1 | 91.1 | +0.0 | 82% |
| Structured3D | Q1 | 0.000–0.000 | 112 | 1.00 | 112 | 10.7 | 47.3 | +36.6 | 100% |
| Structured3D | Q2 | 0.000–0.006 | 4 | 1.00 | 112 | 18.8 | 50.0 | +31.2 | 100% |
| Structured3D | Q3 | 0.006–0.016 | 0 | 1.00 | 112 | 24.1 | 52.7 | +28.6 | 100% |
| Structured3D | Q4 | 0.016–0.040 | 0 | 1.00 | 111 | 27.0 | 53.2 | +26.1 | 100% |
| Structured3D | Q5 | 0.041–0.264 | 0 | 1.00 | 111 | 72.1 | 86.5 | +14.4 | 100% |

## Table 4. Forced choice between a correct and an incorrect hypothesis, no gate (all C×I pairs per sample, samples weighted equally, sample-level bootstrap)

| benchmark | backbone | quintile | samples with pairs | pairs | pairs / sample (max) | acoustic right [95% CI] | visual right [95% CI] |
|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | Q1 | 96 | 889 | 9.3 (16) | 81.9 [77.7, 86.0] | 68.2 [62.7, 73.4] |
| Replica | F3Loc mono | Q2 | 96 | 932 | 9.7 (16) | 80.2 [75.7, 84.7] | 71.4 [66.1, 76.4] |
| Replica | F3Loc mono | Q3 | 107 | 1031 | 9.6 (21) | 79.7 [74.8, 84.3] | 69.7 [63.8, 75.2] |
| Replica | F3Loc mono | Q4 | 102 | 967 | 9.5 (16) | 84.1 [79.8, 88.2] | 83.1 [78.0, 87.6] |
| Replica | F3Loc mono | Q5 | 116 | 1072 | 9.2 (16) | 91.6 [88.5, 94.2] | 96.2 [93.9, 98.2] |
| Replica | F3Loc mono | all | 517 | 4891 | 9.5 (21) | 83.7 [81.8, 85.6] | 78.3 [76.0, 80.7] |
| Replica | UnLoc | Q1 | 102 | 959 | 9.4 (16) | 81.9 [77.4, 86.1] | 71.7 [66.1, 76.9] |
| Replica | UnLoc | Q2 | 107 | 996 | 9.3 (16) | 85.2 [81.7, 88.5] | 78.9 [73.7, 83.7] |
| Replica | UnLoc | Q3 | 114 | 1047 | 9.2 (16) | 87.0 [83.2, 90.5] | 81.3 [76.9, 85.5] |
| Replica | UnLoc | Q4 | 115 | 1070 | 9.3 (16) | 88.4 [84.8, 91.5] | 90.3 [87.1, 93.2] |
| Replica | UnLoc | Q5 | 119 | 1084 | 9.1 (16) | 95.8 [94.1, 97.3] | 98.3 [96.8, 99.4] |
| Replica | UnLoc | all | 557 | 5156 | 9.3 (16) | 87.9 [86.3, 89.4] | 84.6 [82.6, 86.5] |
| Replica | DisCo-FLoc RRP | Q1 | 81 | 764 | 9.4 (16) | 81.8 [77.3, 86.0] | 70.9 [64.7, 77.0] |
| Replica | DisCo-FLoc RRP | Q2 | 95 | 861 | 9.1 (16) | 84.7 [79.9, 89.1] | 67.4 [61.3, 73.3] |
| Replica | DisCo-FLoc RRP | Q3 | 100 | 984 | 9.8 (16) | 84.3 [80.6, 87.6] | 75.2 [70.2, 79.9] |
| Replica | DisCo-FLoc RRP | Q4 | 114 | 1075 | 9.4 (16) | 87.8 [84.3, 91.0] | 86.1 [81.9, 89.8] |
| Replica | DisCo-FLoc RRP | Q5 | 117 | 1144 | 9.8 (16) | 95.1 [93.2, 96.7] | 96.0 [93.7, 97.8] |
| Replica | DisCo-FLoc RRP | all | 507 | 4828 | 9.5 (16) | 87.3 [85.6, 88.8] | 80.3 [78.0, 82.5] |
| Matterport3D | F3Loc mono | Q1 | 111 | 1020 | 9.2 (16) | 74.4 [69.5, 79.3] | 66.9 [60.9, 72.9] |
| Matterport3D | F3Loc mono | Q2 | 136 | 1252 | 9.2 (16) | 78.4 [74.4, 82.4] | 70.6 [65.7, 75.2] |
| Matterport3D | F3Loc mono | Q3 | 140 | 1295 | 9.2 (16) | 78.5 [74.1, 82.7] | 80.4 [76.1, 84.3] |
| Matterport3D | F3Loc mono | Q4 | 159 | 1487 | 9.4 (16) | 83.2 [79.8, 86.5] | 79.8 [75.5, 84.0] |
| Matterport3D | F3Loc mono | Q5 | 167 | 1566 | 9.4 (16) | 83.5 [80.3, 86.5] | 87.3 [83.5, 90.7] |
| Matterport3D | F3Loc mono | all | 713 | 6620 | 9.3 (16) | 80.0 [78.3, 81.8] | 77.9 [75.9, 79.9] |
| Matterport3D | UnLoc | Q1 | 114 | 1040 | 9.1 (16) | 78.4 [73.6, 82.8] | 74.0 [68.7, 78.8] |
| Matterport3D | UnLoc | Q2 | 140 | 1302 | 9.3 (16) | 76.3 [71.9, 80.4] | 72.4 [68.0, 76.5] |
| Matterport3D | UnLoc | Q3 | 149 | 1369 | 9.2 (16) | 81.9 [78.6, 85.0] | 82.3 [78.2, 86.3] |
| Matterport3D | UnLoc | Q4 | 167 | 1559 | 9.3 (16) | 81.6 [78.2, 84.9] | 90.7 [87.6, 93.4] |
| Matterport3D | UnLoc | Q5 | 187 | 1683 | 9.0 (9) | 88.2 [85.4, 90.7] | 97.6 [95.7, 99.2] |
| Matterport3D | UnLoc | all | 757 | 6953 | 9.2 (16) | 81.8 [80.2, 83.3] | 84.9 [83.1, 86.5] |
| Matterport3D | DisCo-FLoc RRP | Q1 | 87 | 790 | 9.1 (16) | 72.0 [66.4, 77.7] | 57.5 [50.3, 64.5] |
| Matterport3D | DisCo-FLoc RRP | Q2 | 126 | 1169 | 9.3 (16) | 83.0 [79.3, 86.6] | 67.2 [62.1, 72.0] |
| Matterport3D | DisCo-FLoc RRP | Q3 | 124 | 1163 | 9.4 (21) | 78.8 [74.8, 82.7] | 70.2 [64.5, 75.3] |
| Matterport3D | DisCo-FLoc RRP | Q4 | 137 | 1282 | 9.4 (16) | 78.0 [73.6, 82.0] | 74.0 [68.7, 79.0] |
| Matterport3D | DisCo-FLoc RRP | Q5 | 152 | 1457 | 9.6 (21) | 82.8 [79.3, 86.0] | 83.4 [79.3, 87.3] |
| Matterport3D | DisCo-FLoc RRP | all | 626 | 5861 | 9.4 (21) | 79.5 [77.6, 81.3] | 71.9 [69.4, 74.2] |
| Structured3D | F3Loc mono | Q1 | 88 | 806 | 9.2 (16) | 94.1 [91.7, 96.2] | 56.3 [50.0, 62.6] |
| Structured3D | F3Loc mono | Q2 | 75 | 689 | 9.2 (16) | 86.3 [81.1, 91.0] | 58.5 [51.9, 64.8] |
| Structured3D | F3Loc mono | Q3 | 83 | 768 | 9.3 (16) | 90.4 [86.1, 94.2] | 72.1 [66.2, 77.9] |
| Structured3D | F3Loc mono | Q4 | 74 | 687 | 9.3 (16) | 90.6 [86.3, 94.2] | 75.1 [68.2, 81.8] |
| Structured3D | F3Loc mono | Q5 | 96 | 906 | 9.4 (16) | 93.1 [90.0, 95.7] | 82.1 [77.0, 86.9] |
| Structured3D | F3Loc mono | all | 416 | 3856 | 9.3 (16) | 91.1 [89.4, 92.7] | 69.2 [66.3, 72.0] |
| Structured3D | UnLoc | Q1 | 87 | 804 | 9.2 (16) | 94.0 [91.5, 96.2] | 50.2 [44.1, 56.3] |
| Structured3D | UnLoc | Q2 | 85 | 772 | 9.1 (16) | 93.9 [91.4, 96.1] | 68.2 [61.8, 74.4] |
| Structured3D | UnLoc | Q3 | 88 | 813 | 9.2 (16) | 91.0 [87.8, 93.9] | 70.4 [64.3, 76.8] |
| Structured3D | UnLoc | Q4 | 95 | 890 | 9.4 (16) | 90.2 [86.7, 93.5] | 71.7 [65.3, 77.8] |
| Structured3D | UnLoc | Q5 | 107 | 1012 | 9.5 (16) | 93.2 [89.9, 96.0] | 91.1 [87.2, 94.4] |
| Structured3D | UnLoc | all | 462 | 4291 | 9.3 (16) | 92.4 [91.1, 93.8] | 71.3 [68.4, 74.0] |
| Structured3D | DisCo-FLoc RRP | Q1 | 76 | 691 | 9.1 (16) | 93.1 [90.4, 95.8] | 59.0 [53.2, 64.8] |
| Structured3D | DisCo-FLoc RRP | Q2 | 79 | 746 | 9.4 (16) | 90.2 [86.4, 93.6] | 62.6 [56.5, 68.6] |
| Structured3D | DisCo-FLoc RRP | Q3 | 77 | 700 | 9.1 (16) | 95.2 [92.5, 97.5] | 66.7 [59.6, 73.4] |
| Structured3D | DisCo-FLoc RRP | Q4 | 79 | 746 | 9.4 (16) | 85.6 [80.7, 90.2] | 73.4 [67.4, 78.9] |
| Structured3D | DisCo-FLoc RRP | Q5 | 96 | 920 | 9.6 (16) | 89.4 [85.9, 92.7] | 83.2 [78.6, 87.8] |
| Structured3D | DisCo-FLoc RRP | all | 407 | 3803 | 9.3 (16) | 90.6 [89.0, 92.2] | 69.7 [66.9, 72.3] |

## Table 4b. The earlier protocol, for continuity: only the two strongest hypotheses, only samples where exactly one is within 1 m

| benchmark | backbone | decidable samples | acoustic right | visual right |
|---|---|---|---|---|
| Replica | F3Loc mono | 312 | 80.8 | 73.4 |
| Replica | UnLoc | 389 | 82.5 | 77.9 |
| Replica | DisCo-FLoc RRP | 319 | 82.1 | 75.2 |
| Matterport3D | F3Loc mono | 426 | 80.5 | 75.1 |
| Matterport3D | UnLoc | 551 | 81.1 | 78.8 |
| Matterport3D | DisCo-FLoc RRP | 333 | 75.4 | 64.6 |
| Structured3D | F3Loc mono | 208 | 86.5 | 56.7 |
| Structured3D | UnLoc | 252 | 89.3 | 61.1 |
| Structured3D | DisCo-FLoc RRP | 192 | 89.1 | 53.6 |

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
the headline protocol (projection trained on the benchmark's own training
rooms, fixed structure, three scalars chosen on validation rooms, test rooms
evaluated once). This section is written by hand.

### Definitions used throughout

**Test samples.** Every query of the test rooms with both motion collections
pooled: Replica 3 rooms × 2 collections × 100 poses = 600, Matterport3D
12 × 2 × 40 = 960, Structured3D 30 rooms × up to 20 poses = 558. The
manuscript's earlier "300 queries on Replica" was the collection-split
protocol (fit on `replica_f`, report on `replica_g`); the paper now pools the
collections and holds out rooms, so every count here and in the headline
table is 600. One sample is one query; every interval in this document is a
bootstrap over samples.

**Hypotheses.** $h_1, \dots, h_K$ with $K = 10$: the spatial hypotheses
after non-maximum suppression at 1.5 m on $\pi(c) = \max_o p_v(c, o)$, the
backbone's posterior over cells with the heading collapsed by max. Orientation
plays no part in any quantity below. $v_k = \log \pi(h_k)$ is the log
posterior at the centre cell of hypothesis $k$; $\alpha_k$ is the acoustic
evidence of hypothesis $k$ alone, the $0.9$ quantile of the projected
acoustic score over the cells within 0.5 m of it.

**Visual ambiguity.** Exactly the quantity the gate reads,
$m_v = v_{(1)} - v_{(2)}$, the log-odds between the two strongest hypotheses
after NMS. It is computed once per sample and nowhere re-defined. The
normalised entropy of $\mathrm{softmax}(v)$ over the $K$ hypotheses,
$u_v$, is reported beside it in Table 3 for reference: it sits at 0.96–1.00
on every benchmark because the posterior is nearly flat across ten separated
places even when its top two differ by a decisive margin, so it does not
resolve the samples that $m_v$ does. The method uses $m_v$; if the manuscript
describes the gate through $u_v$, that text has to change to $m_v$.

**Quintiles.** Per benchmark and backbone, every sample gets its $m_v$, the
samples are sorted by it (a stable sort, so ties keep their order) and split
into five groups of equal count. Q1 is the most ambiguous 20 %, Q5 the most
confident 20 %. The boundaries in Table 3 are read off the test distribution
and are analysis-only; the method never sees them. Structured3D's Q1 consists
entirely of samples with $m_v = 0$, that is, two hypotheses with identical
posterior (112 of 558 samples; UnLoc's posterior saturates on that benchmark's
renders). That is why an earlier version of this table, which cut at
quantile *values* rather than ranks, had no Q1 there.

### Table 3 is a check on the gate, not evidence about sound

The final method suppresses the acoustic term where $m_v$ is large, so a gain
concentrated in the ambiguous quintiles is what the design predicts. Table 3
confirms that it happens: on Replica the gain is +15.0, +13.3, +15.8 in
Q1–Q3 and 0.0 in Q5, where vision is at 93 %; on Matterport3D +9.9 in Q1 and
0.0 in Q5. Read it as "the gate opens where it should and closes where it
should", nothing more. Structured3D is different because the validation rooms
chose a gate that is open on every query ($\tau_v = 0.2$, above every
$m_v$); there the gain reaches even Q5.

### Table 4 is the evidence about sound, with the gate removed

The claim that sound itself separates the right place from the wrong ones
has to be made without the gate or the fusion in the loop, and that is what
Table 4 does. For each sample, $C$ is the set of hypotheses within 1 m of the
truth and $I$ the set at 1 m or beyond, among the same $K = 10$. Every pair
$(c, i) \in C \times I$ is one trial, so a sample yields up to
$|C| \cdot |I| \le 25$ pairs (9.3 on average, 21 at most in practice) and a
sample with no correct or no incorrect hypothesis yields none (Replica UnLoc:
557 of 600 samples contribute). Two orderings are scored on identical pairs:

* *acoustic right*: $\alpha_c > \alpha_i$, the acoustic evidence of each
  hypothesis alone. No visual prior, no gate, no fused score enters.
* *visual right*: $v_c > v_i$, the backbone's own posterior.

Every sample carries equal weight, its pairs sharing $1/(|C||I|)$, and the
intervals are bootstraps over samples, since pairs from one sample are not
independent. The strata are the same five quintiles as Table 3, so the two
tables line up sample by sample.

Read this way: the acoustic ordering is right 80–88 % of the time on the
furnished benchmarks and 90–92 % on Structured3D, and it is flat across
quintiles. The visual ordering is right 89–97 % in Q5 and 56–74 % in Q1. So
sound is informative on the decision the method makes, and it is informative
*independently of how ambiguous vision is*, whereas vision's own ordering is
the thing that degrades. The gate is the consequence: it hands the decision
to sound exactly where the visual ordering has become the weaker of the two.

Table 4b keeps the earlier protocol for continuity. It used only the two
strongest hypotheses and only the samples where exactly one of them is within
1 m (the "decidable" pairs, 193 on Replica under the old collection split,
312–389 here), one pair per sample. Its numbers are lower for both orderings
because the pair it keeps is by construction the hardest one, the two places
vision could not tell apart; the direction is the same.

### Where the gain is (Tables 1–2, `figs/A_gain_per_scene.png`)

Every Replica and Matterport3D room gains under every backbone except two
Matterport3D rooms at −1 point for one backbone each, noise at 80 queries.
The gain is largest where vision is weakest: on Matterport3D it is 10–21
points in rooms where UnLoc or DisCo are below 30 % and 3–5 points above
55 %. Sound repairs far more queries than it breaks (Replica 74–103 against
19–32 per backbone; Matterport3D 51–139 against 6–58; Structured3D 156–166
against 12–20). The largest group everywhere is "both wrong": samples whose
truth is not among the ten hypotheses or where sound cannot separate it, and
that group, not the regressions, caps the method.

### Sound cannot search, it can help choose (Table 5, `figs/E_candidate_set.png`)

Asked to choose alone among the top-$K$ hypotheses, the acoustic score gets
worse as $K$ grows on the furnished benchmarks (Matterport3D 45.3 → 33.2 from
$K = 1$ to 10) and reaches 36.0 % and 15.0 % over the whole grid, while the
gated fusion over the same $K$ rises to 60.0 and 50.0. On Structured3D the
acoustic score alone is competitive at every $K$ and reaches 59.3 % over the
whole grid, which is why the cell-product rule
(`../../feasible/results/CELL_product.md`) gains most there.

### Samples (`figs/samples/`)

Rows drawn under the same protocol: the view, the truth, the fused pick on
the hypothesis discs, the acoustic score over every cell, the visual
posterior. They are the largest repairs per room with a view that has content
in it, favourable by construction; Table 1 says how common a repair is.
