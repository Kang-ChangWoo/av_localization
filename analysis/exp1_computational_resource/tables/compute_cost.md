# Computational cost

Measured on one NVIDIA GeForce RTX 3090 and a Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz, PyTorch 2.5.1+cu121. Per-query times are medians over 30 held-out queries per scene after warm-up.

(f) front branch: the image encoder, which the acoustic branch does not touch. (b) back branch: everything after it, the ray head, the floorplan match, and the acoustic term and fusion that join there.

## Learned parameters

| component | branch | parameters | note |
|---|---|---|---|
| UnLoc encoder | (f) | 304.4 M | ViT-L, frozen |
| UnLoc head | (b) | 1.24 M | conv and attention that turn features into rays |
| UnLoc depth head | - | 30.9 M | in the checkpoint, never called at inference |
| F3Loc mono encoder | (f) | 23.6 M | ResNet-50, layer 1 |
| F3Loc mono head | (b) | 2.42 M | conv and attention that turn features into rays |
| DisCo-FLoc RRP encoder | (f) | 22.1 M | ViT-S, frozen |
| DisCo-FLoc RRP head | (b) | 0.60 M | conv and attention that turn features into rays |
| floorplan match | (b) | 0 | L1 against the DESDF; nothing learned |
| acoustic branch | (b) | 0.22 M | banded STFT, linear projection 1746x128, L1 match |
| fusion rule | (b) | 3 | weight, sigmoid scale, visual threshold (one scalar for the cell-product rule) |

## Cost per building, paid once

| scene | cells | grid on disk | render wall (12 workers) | core-hours | s per cell | feature kept | featurise |
|---|---|---|---|---|---|---|---|
| office_4 | 2734 | 1.66 GB | 14 min | 2.8 | 3.6 | 19.1 MB | 7 s |
| apartment_2 | 5443 | 2.90 GB | 30 min | 6.0 | 4.0 | 38.0 MB | 43 s |
| frl_apartment_5 | 5344 | 3.40 GB | 30 min | 5.9 | 3.9 | 37.3 MB | 12 s |

All three: 13521 cells, 14.6 core-hours, 3.9 s of single-core simulation per cell. The visual side needs only the DESDF, 0.6 MB for office_4, 1.5 MB for apartment_2, 1.4 MB for frl_apartment_5.


## Time per query, ms

| stage | branch | device | office_4 | apartment_2 | frl_apartment_5 |
|---|---|---|---|---|---|
| image read and normalisation | (f) | CPU | 48 | 53 | 35 |
| UnLoc encoder, 640x480 | (f) | GPU | 314 | 319 | 315 |
| ray extraction from predicted depth | (b) | CPU | 0.7 | 1.4 | 0.8 |
| match against the pose grid, 36 headings | (b) | GPU | 19 | 21 | 19 |
| impulse response read, 6 ch at 48 kHz | (b) | CPU | 47 | 43 | 48 |
| banded STFT of the recording | (b) | CPU | 2.0 | 4.7 | 2.4 |
| score every candidate cell | (b) | CPU | 9.8 | 59 | 23 |
| ten hypotheses and their disc evidence | (b) | CPU | 4.6 | 5.1 | 4.4 |
| gated fusion | (b) | CPU | 0.7 | 0.7 | 0.5 |
| **front branch** | (f) | | 361 | 371 | 350 |
| **back branch, visual** | (b) | | 20 | 22 | 20 |
| **back branch, added by sound** | (b) | | 64 | 112 | 79 |
| **total** | | | **410** | **487** | **447** |

Peak GPU memory 1.7 GB; model load 9.1 s, once. Rows are per-stage medians and do not sum to the total row, which is the median of the whole. Image and impulse-response reads come from network storage. Scoring here covers every cell of the grid, which is what the analysis pipeline does to draw the maps; the rule itself needs only the cells inside its ten half-metre discs, about 800, which the main repository timed at 1.6 ms.


## What the acoustic branch could replace

Each backbone is a frozen image encoder, a trained head that turns features into rays, and a non-learned match against the floorplan. Sound could stand in only for the head and the match. Times are medians over the three scenes, ms.

| backbone | encoder (f) params | head (b) params | encoder (f) ms | head (b) ms | match (b) ms | visual total | saved if (b) replaced | acoustic branch ms |
|---|---|---|---|---|---|---|---|---|
| UnLoc | 304.4 M | 1.24 M | 315 | 0.5 | 19.2 | 332 | 19.7 ms, 1.24 M | 8.9 |
| F3Loc mono | 23.6 M | 2.42 M | 13 | 5.3 | 7.6 | 26 | 12.9 ms, 2.42 M | 8.9 |
| DisCo-FLoc RRP | 22.1 M | 0.60 M | 15 | 6.0 | 12.3 | 33 | 18.3 ms, 0.60 M | 8.9 |

The head and the match are 2 to 9% of the parameters and 3 to 30% of the visual time; the encoder is the cost. They cannot simply be dropped: they are what produces the hypotheses the acoustic branch ranks, and sound alone reaches 36.0% at 1 m against 60.0% fused (UnLoc, Replica, final protocol). UnLoc's checkpoint also carries a 30.9 M depth head that inference never calls; deleting it from the file saves 9% of the parameters at no cost. The acoustic column is the branch's own compute, with scoring limited to the ten discs (1.6 ms, main repository); reading the impulse response from network storage adds about 45 ms on this machine.

