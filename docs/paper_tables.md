# Tables

Query recordings are the furnished scan (`raw_scan_open`); acoustic candidates are rendered from the floorplan alone (`floorplan_closed`). The acoustic feature, the fusion policy and both thresholds are selected once on `replica_f` and shared by every backbone; all reported numbers are on the held-out `replica_g`, 300 queries. Intervals are paired bootstraps over queries.


Shared structure: visual evidence `centre`, acoustic evidence `quantile`, continuous gate. Scalars tuned per backbone on `replica_f`:


| backbone | weight | sigmoid scale | visual threshold | acoustic threshold |
|---|---|---|---|---|
| F3Loc mono | 1 | 0.02 | 0.05 | 0 |
| UnLoc | 0.5 | 0.02 | 0.2 | -inf |

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

## Table 2. Single-frame localization on Replica

| visual backbone | acoustic | 0.1 m | 0.5 m | 1 m | 1 m 30 deg | 2 m | 5 m | median | RMSE | gain @1 m |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | none | 5.3 | 29.7 | 37.7 | 34.0 | 48.7 | 87.7 | 2.17 | 3.17 |  |
|  | **ours** | 5.7 | 35.0 | 45.7 | 40.3 | 58.3 | 89.0 | 1.20 | 3.02 | **+8.0** [+3.3, +12.7] |
| UnLoc | none | 9.0 | 44.7 | 49.7 | 49.0 | 53.3 | 83.7 | 1.12 | 3.28 |  |
|  | **ours** | 10.0 | 49.7 | 54.7 | 53.7 | 59.7 | 87.7 | 0.51 | 3.01 | **+5.0** [+0.7, +9.3] |

## Table 3. Per scene, recall at 1 m

| visual backbone | acoustic | apartment_2 | frl_apartment_5 | office_4 | all |
|---|---|---|---|---|---|
| F3Loc mono | none | 15.0 | 76.0 | 22.0 | 37.7 |
|  | **ours** | 28.0 | 73.0 | 36.0 | 45.7 |
| UnLoc | none | 24.0 | 76.0 | 49.0 | 49.7 |
|  | **ours** | 39.0 | 74.0 | 51.0 | 54.7 |

## Table 4. Ablation of the fusion rule (UnLoc)

| level | rule | 0.5 m | 1 m | 1 m 30 deg | median | gain @1 m | 95% CI |
|---|---|---|---|---|---|---|---|
| \textendash | vision only | 44.7 | 49.7 | 49.0 | 1.12 | +0.0 | [+0.0, +0.0] |
| cell | acoustic alone | 13.7 | 20.7 | - | 3.02 | -29.0 | [-36.7, -21.3] |
| cell | rerank vision top-50 | 45.3 | 52.7 | 50.7 | 0.79 | +3.0 | [-3.3, +9.3] |
| cell | log-rank fusion | 47.0 | 54.3 | 52.7 | 0.70 | +4.7 | [-1.3, +10.7] |
| hypothesis | rerank, unconditional | 30.0 | 34.0 | 33.0 | 2.61 | -15.7 | [-23.3, -8.0] |
| hypothesis | relative evidence | 49.3 | 54.3 | 53.3 | 0.52 | +4.7 | [+0.3, +9.0] |
| hypothesis | gate on visual ambiguity | 40.3 | 45.3 | 44.3 | 1.73 | -4.3 | [-11.0, +2.3] |
| hypothesis | gate on both confidences | 40.3 | 45.3 | 44.3 | 1.73 | -4.3 | [-10.7, +2.3] |
| hypothesis | continuous gate (ours) | 49.7 | 54.7 | 53.7 | 0.51 | +5.0 | [+0.7, +9.3] |
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
| furnished query (the real setting) | 20.7 | 192 | 54.7 | 93.7 |
| matched geometry (upper bound) | 94.3 | 1 | 65.7 | 93.7 |

| K | truth within 1 m of a top-K cell | of a top-K hypothesis |
|---|---|---|
| 1 | 49.7 | 49.7 |
| 3 | 59.0 | 75.7 |
| 5 | 65.7 | 86.3 |
| 10 | 75.7 | 93.7 |
| 20 | 84.7 | - |
| 50 | 93.0 | - |
