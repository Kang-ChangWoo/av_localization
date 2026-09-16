# Exp. 6 — robustness to a degrading camera

UnLoc on Replica with projection and scalars frozen, the query image
corrupted at 28 levels; the hypothesis rule, the beyond-shortlist variants,
the cell-product rule, and whether visual collapse can be detected and
switched on. Assembled from `../../feasible/results` (the degradation and
collapse-switch studies); writes only here.

```
src/robustness.py   -> data/degradation.json, data/collapse_switch.json, figs/V_final_unloc.png,
                       the table block of md/robustness.md
md/robustness.md    tables (generated block) and the reading of them
```

```bash
cd analysis/exp6_robustness && /opt/conda/envs/f3loc/bin/python src/robustness.py
```
