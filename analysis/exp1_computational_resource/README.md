# Exp. 1 — computational resource

Parameters, per-query time and per-building cost of the method, measured with
the acoustic branch as it is now (learned projection $W$, both decision
rules). Self-contained: the scripts read `../../track1_core`, the checkpoints
and grids under `../../outputs`, and write only into this folder.

```
src/measure_cost.py         GPU: parameters, per-query stage times (UnLoc, 30
                            held-out queries per scene), featurisation and
                            projection per building         -> data/cost.json
src/measure_backbones.py    GPU: encoder / head / match split per backbone
                            (one process each)               -> data/backbone_<tag>.json
src/measure_render_cost.py  CPU: rendering wall time and core-hours from the
                            shard logs, grid sizes           -> data/render_cost.json
src/make_tables.py          CPU: the four tables, regenerated inside the one
                            markdown file between its markers
data/                       the measurements (cost_hypothesis_rule_unprojected.json
                            is the earlier run without W, kept for the caveat)
md/computational_resource.md  the tables (generated block) and the reading of them
```

```bash
cd analysis/exp1_computational_resource
/opt/conda/envs/unloc/bin/python src/measure_cost.py --gpu 0
/opt/conda/envs/unloc/bin/python src/measure_backbones.py --backbone unloc
/opt/conda/envs/f3loc/bin/python src/measure_backbones.py --backbone f3loc_mono
/opt/conda/envs/f3loc/bin/python src/measure_backbones.py --backbone disco_rrp
/opt/conda/envs/f3loc/bin/python src/measure_render_cost.py
/opt/conda/envs/f3loc/bin/python src/make_tables.py
```
