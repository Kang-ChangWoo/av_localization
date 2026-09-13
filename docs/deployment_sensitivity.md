# What a deployment can give up

Acoustic score alone, so the effect is not diluted by the visual term. Same queries and same feature as the main tables.


## Query noise

White noise added to the query recording only; the candidate grid is rendered offline and is noiseless by construction. This is not a claim that a real microphone adds white noise, only a measure of the margin.

| query SNR | acoustic @1 m | @2 m | median |
|---|---|---|---|
| clean | 22.0% | 26.0% | 2.76 m |
| 40 dB | 22.0% | 26.0% | 2.76 m |
| 30 dB | 22.0% | 26.0% | 2.76 m |
| 20 dB | 22.0% | 26.0% | 2.84 m |
| 10 dB | 20.0% | 24.0% | 2.81 m |
| 5 dB | 20.0% | 26.0% | 2.80 m |
| 0 dB | 20.0% | 28.0% | 2.93 m |

## Candidate grid resolution

The coarse grids are lattice subsets of the rendered one, which is what a deployment would render instead. Rendering cost falls with the square of the stride.

| grid | cells | render cost | acoustic @1 m | @2 m | median |
|---|---|---|---|---|---|
| 0.1 m | 2734 | 100% | 22.0% | 26.0% | 2.76 m |
| 0.2 m | 683 | 25% | 30.0% | 38.0% | 2.58 m |
| 0.3 m | 304 | 11% | 20.0% | 28.0% | 2.79 m |
| 0.5 m | 110 | 4% | 36.0% | 46.0% | 2.10 m |
