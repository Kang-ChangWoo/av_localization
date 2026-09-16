# Notes on the cost numbers

Companion to `tables.md` (generated) and `render_cost.md` (rendering, read from
the shard logs of the runs that produced the grids). Numbers below are from
`../data/cost.json`, measured with the learned acoustic branch as it is now:
the linear projection $W$ (1746 → 128) on both sides of the match, and the
two decision rules side by side.

## What the acoustic branch costs at inference

Per query, the branch's own computation is a few milliseconds:

| stage | ms (median of three scenes) |
|---|---|
| banded STFT of the recording | 2.8 |
| projection of the recording, 1746 → 128 | 0.1 |
| L1 score of every candidate cell, 128-d | 1.1–2.2 |
| hypothesis rule (ten hypotheses, disc evidence, gate) | 8.3–9.6 |
| cell-product rule (one argmax) | 0.2 |

So the branch adds 4–5 ms of computation, plus 8–10 ms if the hypothesis rule
is used and 0.2 ms if the cell-product rule is. The visual backbone's encoder
alone is 140–180 ms on the same GPU. The projection is free: 0.1 ms per query,
and a hundredth of a second once per building for all candidate cells.

What actually dominates the acoustic branch's wall time is reading the
impulse response, 70–125 ms on this machine, because the recordings live on
network storage. That is an artefact of the measurement setup, not of the
method; a microphone on the robot produces the recording in memory.

## What the projection changes offline

The feature kept per building shrinks from 1746 floats per cell to 128, so the
"feature kept" column of Table 2 goes from 19–38 MB to 1.4–2.8 MB per scene
(14× smaller), while the impulse-response grid itself (1.7–3.4 GB per scene)
is only needed once, at featurisation. Scoring in 128 dimensions is also what
makes the per-cell L1 cheaper than it was on the raw feature.

## Rendering, once per building

Unchanged from `render_cost.md`: 3.9 s of single-core simulation per
candidate cell at the paper's settings (48 kHz, 20,000 rays, reflection depth
50, diffraction to order 10), so a 5,000-cell floorplan is about 5 core-hours,
or 25 minutes on twelve workers. This is by far the largest number in the
analysis and it is paid before deployment, not at query time.

## Reading Table 4

Sound could stand in only for the visual head and the floorplan match, which
are 0.6–2.4 M parameters and 4–20 ms; the encoder is the cost and the
acoustic branch cannot replace it. This measurement was made with the
hypothesis rule in mind. Under the cell-product rule the relationship is more
direct: the acoustic score is a second term added to the visual posterior over
the same cells, and the only thing it needs from the visual side is that
posterior.

## Caveats

* The encoder times here (140–180 ms) are lower than an earlier measurement
  on the same card (315 ms, `../data/cost_hypothesis_rule_unprojected.json`),
  which was taken while other tenants shared the GPU. The acoustic-branch rows
  are CPU-bound and were unaffected.
* Times are medians over 30 held-out queries per scene after five warm-up
  queries, single query at a time, no batching.
* The three scenes are Replica's test rooms, 2.7–5.4 k candidate cells. Costs
  that scale with cell count (the L1 score, featurisation, rendering) grow
  linearly with floor area; Matterport3D floors have 10–56 k cells.
