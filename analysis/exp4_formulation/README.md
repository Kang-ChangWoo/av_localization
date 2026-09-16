# Exp. 4 — why verification rather than acoustic localization?

The acoustic score choosing alone among the top-K visual hypotheses, the
gated fusion over the same K, and the oracle, for K = 1 … 10 and the whole
grid; three benchmarks, three backbones. Self-contained: reads
`../../outputs/analysis` and `../../feasible/results`, writes only here.

```
src/formulation.py   -> data/formulation.json, figs/formulation.png, the table block of md/formulation.md
md/formulation.md    table (generated block) and the reading of it
```

```bash
cd analysis/exp4_formulation && /opt/conda/envs/f3loc/bin/python src/formulation.py
```
