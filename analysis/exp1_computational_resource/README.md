# Exp. 1 — computational resource

Everything the paper says about parameters, per-query time and per-building
cost, in one place. Nothing here depends on `vis/`; the scripts read
`../../track1_core`, the checkpoints and grids under `../../outputs`, and
write only into this folder.

```
analysis/exp1_computational_resource/
  src/measure_cost.py        GPU: parameters per component, per-query stage
                             timings over 30 held-out queries per scene, and
                             per-building featurisation -> data/cost.json
  src/measure_backbones.py   GPU: encoder / head / floorplan-match split per
                             backbone (one process each; DisCo needs the
                             f3loc environment) -> data/backbone_<tag>.json
  src/make_cost_table.py     CPU: the tables -> tables/compute_cost.tex, .md
  data/cost.json             measured on one RTX 3090, Xeon 8268, PyTorch 2.5.1
  data/render_cost.json      candidate-grid rendering: cells, wall time,
                             core-hours, size on disk per test scene
  data/backbone_*.json       per-backbone split
  tables/compute_cost.tex    four booktabs tables, \input-able
  tables/compute_cost.md     the same, for reading
```

## Reproduce

```bash
cd analysis/exp1_computational_resource
/opt/conda/envs/unloc/bin/python src/measure_cost.py --gpu 0            # ~5 min
/opt/conda/envs/unloc/bin/python src/measure_backbones.py --backbone unloc
/opt/conda/envs/f3loc/bin/python src/measure_backbones.py --backbone f3loc_mono
/opt/conda/envs/f3loc/bin/python src/measure_backbones.py --backbone disco_rrp
/opt/conda/envs/f3loc/bin/python src/make_cost_table.py                # seconds
```

## What is measured, and what has moved since

The timings were taken with the hypothesis-verification rule (ten visual
hypotheses, half-metre discs, gated selection) on the unprojected acoustic
feature. Two things changed in the method afterwards and neither changes the
picture:

* the acoustic feature is now multiplied by a learned linear projection
  $W \in \mathbb{R}^{1746 \times 128}$ before the $L_1$ match. Per query that is
  one matrix product of 1746×128 for the recording and, once per building, the
  same for every candidate cell; the per-cell feature kept on disk shrinks from
  1746 to 128 floats (about 14× smaller than the "feature kept" column). The
  parameter table lists $W$ (0.22 M) as the branch's only learned part;
* the rule itself is three scalars, or one for the cell-product variant, not
  four; the "fusion rule" row says so. The cell-product variant scores every
  candidate cell, which is exactly what the "score every candidate cell" row
  already times (10–59 ms on CPU), and drops the hypothesis rows (about 5 ms).

The per-building numbers (rendering, core-hours, grid size) are unchanged.
Impulse-response and image reads come from network storage on this machine
and dominate the acoustic branch's per-query time; on local disk the branch is
the STFT plus the match, about 10 ms.

The numbers quoted in the sentences of `make_cost_table.py` (acoustic alone
36.0 %, fused 60.0 %) are UnLoc on Replica under the final protocol; see
`../../feasible/results/VAL_replica_fixed_indomain.md`.
