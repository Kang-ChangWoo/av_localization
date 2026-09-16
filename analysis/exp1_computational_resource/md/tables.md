# Computational resource, av_localization

Measured on one NVIDIA GeForce RTX 3090 and a Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz, PyTorch 2.5.1+cu121. Per-query times are medians over 30 held-out Replica queries per scene after warm-up. (f) front branch: the image encoder, which the acoustic branch does not touch. (b) back branch: everything after it.

## Table 1. Learned parameters

| component | branch | parameters | note |
|---|---|---|---|
| UnLoc encoder | (f) | 304.4 M | ViT-L, frozen |
| UnLoc head | (b) | 1.24 M | features → rays |
| UnLoc depth head | – | 30.9 M | in the checkpoint, never called |
| F3Loc mono encoder | (f) | 23.6 M | ResNet-50, layer 1 |
| F3Loc mono head | (b) | 2.42 M | features → rays |
| DisCo-FLoc RRP encoder | (f) | 22.1 M | ViT-S, frozen |
| DisCo-FLoc RRP head | (b) | 0.60 M | features → rays |
| floorplan match | (b) | 0 | L1 against the DESDF |
| acoustic projection W | (b) | 0.22 M | linear, 1746×128, the branch's only learned part |
| acoustic feature and score | (b) | 0 | banded STFT, L1 |
| hypothesis rule | (b) | 3 | weight, sigmoid scale, visual threshold |
| cell-product rule | (b) | 1 | λ |

## Table 2. Cost per building, paid once

| scene | cells | grid on disk | render wall | core-hours | s / cell | featurise | project | feature kept (raw → projected) | DESDF |
|---|---|---|---|---|---|---|---|---|---|
| office_4 | 2734 | 1.66 GB | 14 min (12 workers) | 2.8 | 3.6 | 20 s | 0.01 s | 18.2 MB → 1.3 MB | 0.6 MB |
| apartment_2 | 5443 | 2.90 GB | 30 min (12 workers) | 6.0 | 4.0 | 70 s | 0.00 s | 36.3 MB → 2.7 MB | 1.4 MB |
| frl_apartment_5 | 5344 | 3.40 GB | 30 min (12 workers) | 5.9 | 3.9 | 62 s | 0.01 s | 35.6 MB → 2.6 MB | 1.4 MB |

All scenes: 13521 cells, 14.6 core-hours, 3.9 s of single-core simulation per cell.

## Table 3. Time per query, ms (UnLoc)

| stage | branch | device | office_4 | apartment_2 | frl_apartment_5 |
|---|---|---|---|---|---|
| image read and normalisation | (f) | CPU | 67 | 90 | 98 |
| UnLoc encoder, 640×480 | (f) | GPU | 181 | 139 | 138 |
| ray extraction from predicted depth | (b) | CPU | 0.6 | 0.5 | 0.5 |
| match against the pose grid, 36 headings | (b) | GPU | 4.4 | 3.8 | 3.7 |
| impulse response read, 6 ch at 48 kHz | (b) | CPU | 69 | 102 | 125 |
| banded STFT of the recording | (b) | CPU | 2.9 | 2.8 | 2.8 |
| projection of the recording, 1746 → 128 | (b) | CPU | 0.1 | 0.1 | 0.1 |
| L1 score of every candidate cell (128-d) | (b) | CPU | 1.1 | 2.2 | 2.2 |
| hypothesis rule: ten hypotheses and disc evidence | (b) | CPU | 7.6 | 9.0 | 8.9 |
| hypothesis rule: gated selection | (b) | CPU | 0.7 | 0.7 | 0.6 |
| cell-product rule: argmax of log π + λ z_a | (b) | CPU | 0.2 | 0.2 | 0.2 |
| **front branch** | (f) | | **247** | **230** | **235** |
| **back branch, visual** | (b) | | **5** | **4** | **4** |
| **added by sound, hypothesis rule** | (b) | | **81** | **117** | **139** |
| **added by sound, cell-product rule** | (b) | | **73** | **107** | **130** |
| **total (hypothesis rule)** | | | **337** | **366** | **380** |

Peak GPU memory 1.7 GB; model load 7.5 s, once. Stage rows are medians and do not sum to the total row, which is the median of the whole. The two rules share every row above them; only their last one or two rows differ.

## Table 4. What the acoustic branch could replace

| backbone | encoder (f) params | head (b) params | encoder (f) ms | head (b) ms | match (b) ms | visual total ms | acoustic branch ms (no I/O) |
|---|---|---|---|---|---|---|---|
| UnLoc | 304.4 M | 1.24 M | 315 | 0.0 | 19.2 | 332 | 5.3 |
| F3Loc mono | 23.6 M | 2.42 M | 13 | 3.0 | 7.6 | 26 | 5.3 |
| DisCo-FLoc RRP | 22.1 M | 0.60 M | 15 | 3.7 | 12.3 | 33 | 5.3 |
