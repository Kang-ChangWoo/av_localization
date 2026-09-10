# Additional analysis

Self-contained. Every script here reads only the extracted mode tables under
`../outputs/analysis/` and the dataset, writes into `results/`, and can be re-run
in seconds after any change to the pipeline.

```bash
python analysis/run_analysis.py  --backbone unlocSTFT          # A, B, D, E, F, G, I
python analysis/beyond_fov.py    --backbone unlocSTFT          # C, furnished
python analysis/beyond_fov.py    --backbone unlocSTFT --condition floorplan_closed
python analysis/compute_cost.py                                # cost
```

| block | question | result file |
|---|---|---|
| A | when does acoustic verification help | `results/additional_analysis.md`, `results/A_ambiguity.png` |
| B | can sound choose between two visually plausible hypotheses | same |
| C | does the evidence come from beyond the field of view | `results/beyond_fov.md`, `results/beyond_fov_matched.md`, `results/C_beyond_fov*.png` |
| D | is the shortlist or the verifier the bottleneck | `results/additional_analysis.md`, `results/D_oracle.png` |
| E | what limits the acoustic evidence | same |
| F | is the gain actually acoustic | same |
| G | does sound complement visual failure | same, `results/G_complementarity.png` |
| I | does the acoustic score know when it is wrong | same |
| cost | rendering, storage, inference | `results/compute_cost.md` |

## What the blocks establish, in one line each

**B is the paper's core claim, measured directly.** On a forced choice between
vision's two strongest hypotheses, acoustics are correct 67.9% of the time
against a 50% chance, and on the subset where the two are visually close they
reach 66.7% where vision itself falls to 60.4%.

**F is the control that makes B believable.** Permuting which hypothesis each
acoustic score belongs to, leaving the score distribution untouched, turns
+5.0 into -7.5. The gain is pose-specific acoustic consistency, not a bias in
the fusion rule.

**G says the gain is complementary rather than incidental.** Across scenes and
backbones, a scene's visual recall and the acoustic gain there correlate -0.97.

**C does not support the strong claim.** Under the furnished-to-floorplan gap,
the effect of hidden-geometry difference is +8.8, +3.2 and -4.3 on the three
backbones: inconsistent in sign, which a property of the room cannot be. Under
matched geometry it is +7.7, +7.5 and +6.4, consistent across all three. The
information exists; the furniture gap destroys it. The paper should claim that
acoustic consistency discriminates visually plausible hypotheses, and may
motivate but must not assert the beyond-field-of-view mechanism.

**E and D locate the bottleneck.** Matching the query geometry moves acoustic-
only recall from 20.7% to 94.3% and the median ground-truth rank from 192 to 1.
The oracle over the same ten hypotheses reaches 93.7% against our 54.7%, so the
shortlist is not the constraint and acoustic discrimination under the domain gap
is.

**I is a negative result kept because it explains the design.** No acoustic
self-confidence measure exceeds AUROC 0.64, while visual ambiguity predicts an
acoustic mistake at 0.80, which is why the gate leans on the visual side.

**Cost.** 3.9 s of single-core simulation per candidate cell, about 5 core-hours
for a 5,000-cell floorplan, paid once per building. Inference adds 6 ms, two
orders of magnitude below the visual backbone.
