# Ablations

<!-- tables:start -->
## Table 1. Projection source (structure and three-scalar rule fixed; scalars chosen on validation), gain at 1 m

| benchmark | backbone | vision | none | own W | pooled W | three W | searched (source, structure, rule on validation) |
|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | 38.3 | +6.2 [+3.5, +8.8] | **+10.5 [+7.0, +14.0]** | +11.0 [+7.5, +14.5] | +9.0 [+5.8, +12.2] | +8.8 [+5.3, +12.2] |
| Replica | UnLoc | 50.7 | +5.2 [+2.5, +7.7] | **+9.3 [+6.3, +12.5]** | +7.5 [+4.7, +10.5] | +8.7 [+5.7, +11.8] | +7.5 [+4.7, +10.5] |
| Replica | DisCo-FLoc RRP | 40.5 | +2.8 [+0.0, +5.8] | **+11.8 [+8.2, +15.5]** | +8.2 [+4.3, +12.0] | +6.8 [+3.7, +10.0] | +9.5 [+6.0, +13.2] |
| Matterport3D | F3Loc mono | 33.3 | +2.0 [-0.5, +4.4] | **+6.8 [+4.1, +9.6]** | +4.7 [+2.0, +7.5] | +5.1 [+2.7, +7.5] | +4.4 [+2.2, +6.7] |
| Matterport3D | UnLoc | 45.3 | +4.3 [+2.4, +6.2] | **+4.7 [+3.2, +6.2]** | +6.8 [+4.7, +8.9] | +4.8 [+3.0, +6.7] | +4.6 [+2.6, +6.6] |
| Matterport3D | DisCo-FLoc RRP | 22.9 | +4.2 [+2.0, +6.4] | **+10.0 [+7.4, +12.7]** | +8.0 [+5.4, +10.7] | +7.3 [+4.6, +10.0] | +8.5 [+5.8, +11.4] |
| Structured3D | F3Loc mono | 24.2 | +22.2 [+18.1, +26.3] | **+24.4 [+20.3, +28.7]** | +24.4 [+20.3, +28.7] | +27.4 [+23.3, +31.7] | +27.2 [+22.9, +31.5] |
| Structured3D | UnLoc | 30.1 | +28.3 [+24.2, +32.4] | **+27.8 [+23.7, +31.9]** | +27.8 [+23.7, +31.9] | +27.6 [+23.7, +31.7] | +31.4 [+27.2, +35.7] |
| Structured3D | DisCo-FLoc RRP | 20.6 | +25.8 [+21.5, +29.9] | **+27.8 [+23.5, +31.9]** | +27.8 [+23.7, +32.1] | +27.4 [+23.3, +31.7] | +31.2 [+26.9, +35.5] |

## Table 2. Rule: three scalars against four, own W, structure fixed

| benchmark | backbone | three-scalar rule | four-scalar rule (acoustic gate + relative evidence) |
|---|---|---|---|
| Replica | F3Loc mono | +8.8 [+5.5, +12.2] | +8.5 [+5.3, +11.7] |
| Replica | UnLoc | +7.5 [+4.7, +10.5] | +6.5 [+3.7, +9.3] |
| Replica | DisCo-FLoc RRP | +9.5 [+6.0, +13.2] | +9.7 [+6.2, +13.3] |
| Matterport3D | F3Loc mono | +4.4 [+2.1, +6.6] | +5.7 [+3.3, +8.1] |
| Matterport3D | UnLoc | +4.6 [+2.7, +6.6] | +4.2 [+2.3, +6.0] |
| Matterport3D | DisCo-FLoc RRP | +8.5 [+5.8, +11.2] | +7.4 [+4.7, +10.1] |
| Structured3D | F3Loc mono | +27.2 [+23.1, +31.5] | +28.0 [+23.7, +32.3] |
| Structured3D | UnLoc | +31.4 [+27.2, +35.3] | +31.9 [+27.8, +36.2] |
| Structured3D | DisCo-FLoc RRP | +27.8 [+23.8, +31.9] | +31.2 [+26.9, +35.7] |

## Table 3. Beyond the visual shortlist (settings chosen on validation), gain at 1 m

| benchmark | backbone | hypothesis rule | A: injected acoustic hypotheses | B: shortlist rejection | C: cell-wise product (16-point grid) |
|---|---|---|---|---|---|
| Replica | F3Loc mono | +10.5 [+7.0, +14.0] | +10.5 [+7.0, +14.0] (m=0 (validation kept the plain shortlist)) | +10.5 [+7.0, +14.0] (τg=inf, used on 0% of test) | +13.7 [+9.5, +17.8] |
| Replica | UnLoc | +9.3 [+6.3, +12.5] | +8.3 [+5.3, +11.3] (m=3 w=0.5 s=0.05 τv=0.05) | +9.3 [+6.3, +12.5] (τg=inf, used on 0% of test) | +9.8 [+5.8, +14.0] |
| Replica | DisCo-FLoc RRP | +11.8 [+8.2, +15.5] | +13.7 [+9.5, +17.7] (m=2 w=2 s=0.02 τv=0.05) | +11.8 [+8.2, +15.5] (τg=inf, used on 0% of test) | +12.2 [+8.0, +16.3] |
| Matterport3D | F3Loc mono | +6.8 [+4.1, +9.6] | +6.6 [+3.6, +9.5] (m=2 w=1 s=0.1 τv=0.2) | +6.8 [+4.1, +9.5] (τg=inf, used on 0% of test) | +4.5 [+1.2, +7.8] |
| Matterport3D | UnLoc | +4.7 [+3.2, +6.2] | +7.2 [+4.8, +9.7] (m=2 w=1 s=0.05 τv=0.005) | +4.7 [+3.2, +6.2] (τg=inf, used on 0% of test) | +8.0 [+5.3, +10.7] |
| Matterport3D | DisCo-FLoc RRP | +10.0 [+7.4, +12.7] | +9.2 [+6.6, +11.9] (m=3 w=1 s=0.02 τv=0.05) | +9.1 [+6.2, +11.9] (τg=2, used on 8% of test) | +8.9 [+5.8, +11.9] |
| Structured3D | F3Loc mono | +24.4 [+20.3, +28.7] | +34.2 [+29.6, +38.9] (m=3 w=2 s=0.02 τv=0.1) | +34.8 [+30.1, +39.2] (τg=0.5, used on 32% of test) | +41.8 [+37.1, +46.4] |
| Structured3D | UnLoc | +27.8 [+23.7, +31.9] | +36.4 [+31.7, +40.9] (m=3 w=2 s=0.02 τv=0.2) | +34.1 [+29.6, +38.5] (τg=0.5, used on 23% of test) | +40.3 [+35.7, +45.0] |
| Structured3D | DisCo-FLoc RRP | +27.8 [+23.5, +31.9] | +38.2 [+33.9, +42.5] (m=3 w=2 s=0.02 τv=0.2) | +38.9 [+34.4, +43.4] (τg=0.5, used on 33% of test) | +44.4 [+40.0, +48.9] |

## Table 4. The cell-product rule, one scalar λ chosen on validation (wide grid), against the hypothesis rule

| benchmark | backbone | λ | val @1m | hypothesis rule | cell product |
|---|---|---|---|---|---|
| Replica | F3Loc mono | 0.0281 | 49.7 | +10.5 [+7.0, +14.0] | +10.3 [+6.7, +13.8] |
| Replica | UnLoc | 0.2238 | 56.7 | +9.3 [+6.3, +12.5] | +12.7 [+8.8, +16.5] |
| Replica | DisCo-FLoc RRP | 0.0562 | 49.3 | +11.8 [+8.2, +15.5] | +12.5 [+8.5, +16.5] |
| Matterport3D | F3Loc mono | 0.0398 | 41.7 | +6.8 [+4.1, +9.6] | +7.9 [+5.2, +10.6] |
| Matterport3D | UnLoc | 0.0398 | 52.3 | +4.7 [+3.2, +6.2] | +7.5 [+5.4, +9.6] |
| Matterport3D | DisCo-FLoc RRP | 0.1584 | 28.5 | +10.0 [+7.4, +12.7] | +6.6 [+3.3, +9.8] |
| Structured3D | F3Loc mono | 0.0794 | 69.4 | +24.4 [+20.3, +28.7] | +41.0 [+36.2, +45.9] |
| Structured3D | UnLoc | 0.2238 | 71.5 | +27.8 [+23.7, +31.9] | +39.6 [+34.9, +44.1] |
| Structured3D | DisCo-FLoc RRP | 0.0794 | 66.3 | +27.8 [+23.5, +31.9] | +43.5 [+38.9, +48.0] |

## Table 5. Projection transfer matrix (test-room CV, three scalars refit per fold; not the headline protocol)

| benchmark | backbone | none | R | M | B | T |
|---|---|---|---|---|---|---|
| Replica | F3Loc mono | +6.5 | +11.5 | – | – | – |
| Replica | UnLoc | +7.5 | +10.3 | – | – | – |
| Replica | DisCo-FLoc RRP | +2.7 | +11.3 | +5.8 | +8.3 | +7.7 |
| Matterport3D | F3Loc mono | +4.1 | +5.7 | +7.0 | +5.6 | +4.7 |
| Matterport3D | UnLoc | +3.9 | +3.5 | +5.3 | +5.2 | +3.4 |
| Matterport3D | DisCo-FLoc RRP | +2.9 | +7.0 | +9.8 | +8.3 | +7.4 |
| Structured3D | F3Loc mono | +22.6 | +26.3 | +26.5 | +26.0 | +27.4 |
| Structured3D | UnLoc | +28.1 | +28.0 | +28.1 | +27.4 | +27.6 |
| Structured3D | DisCo-FLoc RRP | +25.8 | +26.9 | +29.2 | +27.2 | +28.7 |

<!-- tables:end -->

## Reading the tables

Assembled by `src/ablation.py` from the selection runs under
`../../feasible/results`; every number was chosen on validation rooms and
evaluated once on the test rooms unless the table says otherwise. This
section is by hand.

**Projection source (Table 1, `figs/projection_source.png`).** The
projection is the largest single lever: from none to the benchmark's own W
the gain roughly doubles on Replica (+6.2 → +10.5, +5.2 → +9.3, +2.8 → +11.8)
and Matterport3D (+2.0 → +6.8, +4.3 → +4.7, +4.2 → +10.0). One pooled W
trained on Replica and Matterport3D together keeps 70–100 % of that
everywhere, so a deployment can ship one projection. Adding Structured3D's
self-pairs (three W) helps nowhere; searching the source on validation with
the structure and rule is worse than fixing them, because three to six
validation rooms cannot resolve those choices.

**Three scalars against four (Table 2).** The acoustic-margin gate and the
relative-evidence transform buy nothing: the two rules are within one point
in every cell and the sign flips from cell to cell. The paper reports three.

**Beyond the shortlist (Table 3).** Injecting the acoustic field's own peaks
as extra hypotheses (A) helps only on Structured3D; rejecting the whole
shortlist on the acoustic gap (B) is switched off by validation on every
furnished cell; the cell-wise product (C) is the one that matters and is
taken up in Table 4.

**The cell-product rule (Table 4).** argmax over cells of log π(c) + λ z_a(c),
one scalar, no hypotheses and no gate, λ chosen on validation on a wide
log-spaced grid. It matches or beats the hypothesis rule in 8 of 9 cells and
by a wide margin on Structured3D (+40 to +44 against +24 to +28); the one
loss is DisCo-FLoc on Matterport3D (+6.6 against +10.0, intervals
overlapping). Under a degrading camera it is ahead at every level
(`../exp6_robustness`). It is the simplest rule in this document and the
strongest, which is a decision for the paper rather than for this table.

**Transfer matrix (Table 5).** Test-room cross-validation, not the headline
protocol; kept because it is the only view of every source on every target.
A projection trained elsewhere can hurt (Matterport3D's W on Replica UnLoc:
+7.5 → +4.2); the pooled one does not.
