# Exp. 2 — does acoustic verification resolve visual ambiguity?

Test samples grouped by how many visually plausible poses vision proposes
(v_(1) − v_k ≤ τ_v, τ_v from validation), and, where the correct pose is among
them, which one vision, sound alone and the fused rule pick. Three benchmarks,
three backbones. Self-contained: reads `../../outputs/analysis` and
`../../feasible/results`, writes only here.

```
src/plausible_poses.py   -> data/plausible_poses.json, figs/plausible_poses.png,
                            the table block of md/plausible_poses.md
md/plausible_poses.md    tables (generated block) and the reading of them
```

```bash
cd analysis/exp2_plausible_poses && /opt/conda/envs/f3loc/bin/python src/plausible_poses.py
```
