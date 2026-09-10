# Tables

Query recordings are floorplan-only geometry (`floorplan_closed`), matching the candidates; acoustic candidates are rendered from the floorplan alone. The acoustic feature, the fusion policy and both thresholds are selected once on `replica_f` and shared by every backbone; all reported numbers are on the held-out `replica_g`, 300 queries. Intervals are paired bootstraps over queries.


Shared structure: visual evidence `centre`, acoustic evidence `max`, continuous gate. Scalars tuned per backbone on `replica_f`:


| backbone | weight | sigmoid scale | visual threshold | acoustic threshold |
|---|---|---|---|---|
| F3Loc mono | 2 | 0.02 | 0.1 | -inf |
| UnLoc | 2 | 0.02 | 0.2 | -inf |
| DisCo-FLoc RRP | 2 | 0.05 | 0.2 | 0 |

## Table 1. Reproduction of the published baselines

| method | dataset | 0.1 m | 0.5 m | 1 m | 1 m 30 deg | source |
|---|---|---|---|---|---|---|
| F3Loc mono | Gibson-f | 4.7 | 28.6 | 36.6 | 35.1 | published |
| | | 4.7 | 28.5 | 36.5 | 35.0 | ours |
| F3Loc multi-view | Gibson-f | 13.2 | 40.9 | 45.2 | 43.7 | published |
| | | 13.2 | 40.8 | 45.1 | 43.6 | ours |
| F3Loc complementary | Gibson-g | 12.2 | 39.4 | 44.5 | 43.2 | published |
| | | 12.2 | 39.3 | 44.4 | 43.1 | ours |
| UnLoc | Gibson-t | 19.7 | 61.1 | 64.7 | 63.8 | published |
| | | 19.7 | 61.1 | 64.6 | 63.7 | ours |
| DisCo-FLoc (RRP only) | Gibson-f | 12.0 | 45.8 | 50.6 | 49.2 | published |
| | | 11.8 | 45.0 | 49.6 | 48.3 | ours |
| DisCo-FLoc (full) | Gibson-f | 13.1 | 50.9 | 56.7 | 55.4 | published |
| | | 13.8 | 50.2 | 56.5 | 55.6 | ours |

## Table 2b. UnLoc-style layout

| method | audio | 0.1 m | 0.5 m | 1 m | 1 m 30 deg | 2 m | 5 m | 10 m |
|---|---|---|---|---|---|---|---|---|
| F3Loc mono |   | 5.3 | 29.7 | 37.7 | 34.0 | 48.7 | 87.7 | 99.7 |
|  | **ours** | 7.3 | 52.7 | 67.0 | 59.3 | 72.3 | 92.7 | 100.0 |
| UnLoc |   | 9.0 | 44.7 | 49.7 | 49.0 | 53.3 | 83.7 | 100.0 |
|  | **ours** | 14.3 | 73.0 | 79.7 | 77.7 | 82.0 | 92.3 | 100.0 |
| DisCo-FLoc RRP |   | 5.3 | 31.0 | 41.0 | 39.3 | 46.7 | 83.0 | 100.0 |
|  | **ours** | 8.7 | 55.7 | 71.7 | 69.0 | 77.7 | 92.3 | 99.0 |

## Table 2. Single-frame localization on Replica

| visual backbone | acoustic | 0.1 m | 0.5 m | 1 m | 1 m 30 deg | 2 m | 5 m | median | RMSE | gain @1 m |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | none | 5.3 | 29.7 | 37.7 | 34.0 | 48.7 | 87.7 | 2.17 | 3.17 |  |
|  | **ours** | 7.3 | 52.7 | 67.0 | 59.3 | 72.3 | 92.7 | 0.47 | 2.51 | **+29.3** [+24.0, +34.7] |
| UnLoc | none | 9.0 | 44.7 | 49.7 | 49.0 | 53.3 | 83.7 | 1.12 | 3.28 |  |
|  | **ours** | 14.3 | 73.0 | 79.7 | 77.7 | 82.0 | 92.3 | 0.26 | 2.26 | **+30.0** [+24.3, +35.7] |
| DisCo-FLoc RRP | none | 5.3 | 31.0 | 41.0 | 39.3 | 46.7 | 83.0 | 2.39 | 3.48 |  |
|  | **ours** | 8.7 | 55.7 | 71.7 | 69.0 | 77.7 | 92.3 | 0.43 | 2.46 | **+30.7** [+25.0, +36.3] |

## Table 3. Per scene, recall at 1 m

| visual backbone | acoustic | apartment_2 | frl_apartment_5 | office_4 | all |
|---|---|---|---|---|---|
| F3Loc mono | none | 15.0 | 76.0 | 22.0 | 37.7 |
|  | **ours** | 54.0 | 94.0 | 53.0 | 67.0 |
| UnLoc | none | 24.0 | 76.0 | 49.0 | 49.7 |
|  | **ours** | 67.0 | 92.0 | 80.0 | 79.7 |
| DisCo-FLoc RRP | none | 18.0 | 68.0 | 37.0 | 41.0 |
|  | **ours** | 53.0 | 87.0 | 75.0 | 71.7 |

## Table 4. Ablation of the fusion rule (UnLoc)

| level | rule | 0.5 m | 1 m | 1 m 30 deg | median | gain @1 m | 95% CI |
|---|---|---|---|---|---|---|---|
| \textendash | vision only | 44.7 | 49.7 | 49.0 | 1.12 | +0.0 | [+0.0, +0.0] |
| cell | acoustic alone | 93.7 | 94.3 | - | 0.05 | +44.7 | [+38.7, +50.3] |
| cell | rerank vision top-50 | 75.7 | 77.3 | 75.7 | 0.08 | +27.7 | [+22.0, +33.3] |
| cell | log-rank fusion | 72.3 | 73.3 | 72.3 | 0.14 | +23.7 | [+18.0, +29.3] |
| hypothesis | rerank, unconditional | 76.0 | 83.7 | 80.3 | 0.25 | +34.0 | [+27.7, +40.0] |
| hypothesis | relative evidence | 73.0 | 80.0 | 78.0 | 0.26 | +30.3 | [+24.7, +36.0] |
| hypothesis | gate on visual ambiguity | 75.3 | 83.0 | 79.7 | 0.26 | +33.3 | [+27.3, +39.3] |
| hypothesis | gate on both confidences | 75.3 | 83.0 | 79.7 | 0.26 | +33.3 | [+27.3, +39.3] |
| hypothesis | continuous gate (ours) | 73.0 | 79.7 | 77.7 | 0.26 | +30.0 | [+24.3, +35.7] |
| oracle | best of the ten hypotheses | 78.3 | 93.7 | 89.3 | 0.25 | +44.0 | [+38.3, +49.7] |

## Table 5. Ablation of the acoustic feature, selected on replica_f

| feature | analysis window, ms | acoustic alone @1 m | GT rank | picks the right hypothesis |
|---|---|---|---|---|
| envelope 1 ms | 1.00 | 18.7 | 240 | **29.0** |
| envelope 2 ms | 2.00 | 22.3 | 260 | **29.3** |
| stft nfft 64 hop 16 | 1.33 | 15.7 | 184 | **25.3** |
| stft nfft 64 hop 32 | 1.33 | 17.7 | 210 | **23.3** |
| stft nfft 128 hop 32 | 2.67 | 16.7 | 150 | **29.0** |
| stft nfft 128 fine bands | 2.67 | 16.3 | 156 | **30.0** |
| stft nfft 256 hop 64 (ours) | 5.33 | 21.0 | 186 | **35.3** |
| stft nfft 64 wide bands | 1.33 | 17.3 | 179 | **24.3** |

## Table 6. Upper bounds and the domain gap

| setting | acoustic alone @1 m | GT rank | ours @1 m | oracle @1 m |
|---|---|---|---|---|
| furnished query (the real setting) | 20.7 | 192 | 50.7 | 93.7 |
| matched geometry (upper bound) | 94.3 | 1 | 79.7 | 93.7 |

| K | truth within 1 m of a top-K cell | of a top-K hypothesis |
|---|---|---|
| 1 | 49.7 | 49.7 |
| 3 | 59.0 | 75.7 |
| 5 | 65.7 | 86.3 |
| 10 | 75.7 | 93.7 |
| 20 | 84.7 | - |
| 50 | 93.0 | - |
