# Exp. 2 — when does acoustic verification help?

Where the gain comes from, under the paper's protocol (projection trained on
the benchmark's own rooms, fixed structure, three scalars chosen on validation
rooms, test rooms evaluated once). Self-contained: the script reads the test
tables under `../../outputs/analysis` and the chosen scalars under
`../../feasible/results`, and writes only into this folder.

```
src/when_works.py     per scene, per outcome, by visual ambiguity, forced
                      pairwise choice, and the candidate-set curve, for three
                      benchmarks and three backbones
                                          -> data/when_works.json, figs/*.png,
                                             the table block of md/when_works.md
data/when_works.json  every number behind the tables and figures
figs/A_gain_per_scene.png   gain against visual recall, one point per room
figs/C_by_ambiguity.png     recall by visual-ambiguity quintile
figs/D_pairwise.png         forced choice, acoustic against visual ordering
figs/E_candidate_set.png    acoustic alone / gated / oracle as K grows
figs/samples/               qualitative rows from ../../../vis (UnLoc)
md/when_works.md      tables (generated block) and the reading of them
```

```bash
cd analysis/exp2_when_works
/opt/conda/envs/f3loc/bin/python src/when_works.py     # ~2 min, CPU
```
