# I. Pairwise acoustic discrimination

For every query with a correct hypothesis (within 1 m) among its ten, each incorrect hypothesis is paired with it and the acoustic score is asked to pick. Chance is 50%. The visual ordering is scored on the same pairs. Intervals are bootstraps over pairs.


## F3Loc mono

600 queries, 4613 pairs. Visual ambiguity is split at its median log-odds (0.01).

| stratum | pairs | acoustic picks correct | 95% CI | visual ordering | 95% CI |
|---|---|---|---|---|---|
| all | 4613 | 76.1% | [74.9, 77.3] | 72.1% | [70.8, 73.4] |
| visually ambiguous | 2304 | 76.9% | [75.1, 78.6] | 66.2% | [64.3, 68.2] |
| visually confident | 2309 | 75.3% | [73.5, 77.1] | 78.0% | [76.3, 79.7] |
| spatially close (<2 m) | 1256 | 74.8% | [72.5, 77.2] | 74.1% | [71.7, 76.5] |
| spatially far (>4 m) | 1561 | 76.9% | [74.8, 78.9] | 68.5% | [66.2, 70.9] |

## UnLoc

600 queries, 4989 pairs. Visual ambiguity is split at its median log-odds (0.06).

| stratum | pairs | acoustic picks correct | 95% CI | visual ordering | 95% CI |
|---|---|---|---|---|---|
| all | 4989 | 79.7% | [78.6, 80.8] | 81.9% | [80.9, 83.0] |
| visually ambiguous | 2490 | 80.0% | [78.4, 81.6] | 74.7% | [72.9, 76.3] |
| visually confident | 2499 | 79.4% | [77.8, 81.0] | 89.2% | [88.0, 90.4] |
| spatially close (<2 m) | 1151 | 81.6% | [79.2, 83.8] | 89.8% | [88.1, 91.6] |
| spatially far (>4 m) | 1843 | 80.4% | [78.6, 82.2] | 76.4% | [74.4, 78.3] |

## DisCo-FLoc RRP

600 queries, 4524 pairs. Visual ambiguity is split at its median log-odds (0.01).

| stratum | pairs | acoustic picks correct | 95% CI | visual ordering | 95% CI |
|---|---|---|---|---|---|
| all | 4524 | 77.7% | [76.5, 79.0] | 77.9% | [76.6, 79.1] |
| visually ambiguous | 2261 | 81.3% | [79.7, 82.9] | 71.6% | [69.7, 73.5] |
| visually confident | 2263 | 74.2% | [72.4, 76.0] | 84.1% | [82.6, 85.6] |
| spatially close (<2 m) | 1180 | 78.6% | [76.2, 80.8] | 83.5% | [81.3, 85.6] |
| spatially far (>4 m) | 1507 | 78.1% | [76.0, 80.2] | 72.3% | [70.0, 74.5] |

What to read off this. Above 50% everywhere means the cue is not empty. Where acoustics beats the visual ordering is where the gate should act; where it does not, the gate should stay closed, and that is what the visual-ambiguity strata test directly.

