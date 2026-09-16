# Computational cost

Measured on the machine the results were produced on. Rendering figures come from the logs of the runs that produced the grids in use, not from a re-estimate.


## Offline: the acoustic candidate grid

One impulse response per navigable cell, rendered once per floorplan on wall-only geometry at 48 kHz, reflection depth 50, diffraction to order 10, 20k indirect rays.

| scene | cells | grid on disk | per cell | feature only |
|---|---|---|---|---|
| office_4 | 2734 | 1.66 GB | 0.61 MB | 19.1 MB |
| apartment_2 | 5443 | 2.90 GB | 0.53 MB | 38.0 MB |
| frl_apartment_5 | 5344 | 3.40 GB | 0.64 MB | 37.3 MB |

Total 13521 cells, 8.0 GB of impulse responses. Only the feature is needed at inference, and it is roughly 84x smaller, so a deployment stores megabytes rather than gigabytes.


## Offline: rendering wall time

Read from the render logs, which record the rate directly. Each scene was split round-robin across twelve workers, so wall time is the slowest shard and core time is their sum.

| scene | shards | cells | slowest shard | core time | per cell |
|---|---|---|---|---|---|
| apartment_2 | 12 | 5443 | 30 min | 6.0 core-h | 4.0 s |
| frl_apartment_5 | 12 | 5344 | 30 min | 5.9 core-h | 3.9 s |
| office_4 | 12 | 2734 | 14 min | 2.8 core-h | 3.6 s |

Across the three scenes, 13521 cells for 14.6 core-hours, **3.9 s of single-core simulation per candidate cell**. A 5,000-cell floorplan is about 5 core-hours, or 0.4 hours on twelve workers.


This is the method's real deployment cost, it is paid once per building, and it is the number a reviewer should be given rather than an inference latency that flatters the method.


## Online: per query

The acoustic branch only. Timed in isolation, repeated, median reported.

| stage | median time |
|---|---|
| spectrogram of the query response | 2.11 ms |
| normalisation and shape | 0.07 ms |
| scoring 800 candidate cells (10 discs) | 3.94 ms |
| **total acoustic branch** | **6.12 ms** |

The visual backbone runs a vision transformer over a 640x480 image and an $\ell_1$ match over the whole pose grid, which is two orders of magnitude more. The acoustic branch is not the bottleneck at inference; the offline grid is the only cost that matters.


## Scaling with the number of hypotheses

Scoring is linear in the cells examined, and the method examines only the discs around K hypotheses rather than the whole grid.

| K | cells scored | scoring time | fraction of a full-grid scan |
|---|---|---|---|
| 1 | 80 | 0.18 ms | 1.5% |
| 5 | 400 | 0.82 ms | 7.3% |
| 10 | 800 | 1.58 ms | 14.7% |
| 20 | 1600 | 7.28 ms | 29.4% |

A full-grid cell-wise fusion would score all 5443 cells of the largest scene. Working over hypotheses is not only more accurate (Tab.~\ref{tab:fusion}), it examines a few percent of the cells.

