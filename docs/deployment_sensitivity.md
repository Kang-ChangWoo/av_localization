# What a deployment can give up

Acoustic score alone, so the effect is not diluted by the visual term. Same queries and same feature as the main tables.


## Query noise

White noise added to the query recording only; the candidate grid is rendered offline and is noiseless by construction. This is not a claim that a real microphone adds white noise, only a measure of the margin.

| query SNR | acoustic @1 m | @2 m | median |
|---|---|---|---|
| clean | 22.2% | 32.5% | 2.90 m |
| 20 dB | 22.8% | 32.8% | 2.90 m |
| 0 dB | 13.6% | 17.2% | 3.87 m |

## Candidate grid resolution

The coarse grids are lattice subsets of the rendered one, which is what a deployment would render instead. Rendering cost falls with the square of the stride.

| grid | cells | render cost | acoustic @1 m | @2 m | median |
|---|---|---|---|---|---|
| 0.1 m | 13521 | 100% | 22.2% | 32.5% | 2.90 m |
| 0.2 m | 3373 | 25% | 20.0% | 31.4% | 2.94 m |
| 0.3 m | 1498 | 11% | 19.2% | 32.8% | 2.96 m |
| 0.5 m | 542 | 4% | 22.5% | 33.9% | 2.96 m |
