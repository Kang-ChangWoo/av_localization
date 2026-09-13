# Acoustic gain as the camera degrades

The query image is corrupted before the frozen visual backbone; the acoustic side is untouched. Scalars are refit by leave-one-room-out at every level. `audio acts` is the fraction of queries on which the gate lets the acoustic term change the answer.

| degradation | queries | vision @1m | acoustic alone | ours @1m | gain | 95% CI | audio acts | median visual ambiguity |
|---|---|---|---|---|---|---|---|---|
| clean | 600 | 38.3% | 20.8% | 43.3% | +5.0 | [+1.7, +8.3] | 98% | 0.01 |
| blur σ=2 | 600 | 5.5% | 20.8% | 12.8% | +7.3 | [+4.5, +10.2] | 100% | 0.00 |
| blur σ=4 | 600 | 3.5% | 20.8% | 12.2% | +8.7 | [+6.0, +11.3] | 100% | -0.00 |
| blur σ=8 | 600 | 5.0% | 20.8% | 11.8% | +6.8 | [+4.0, +9.7] | 100% | -0.00 |
| dark ×0.5 | 600 | 9.2% | 20.8% | 16.5% | +7.3 | [+4.2, +10.3] | 88% | 0.00 |
| dark ×0.25 | 600 | 5.3% | 20.8% | 17.0% | +11.7 | [+8.7, +14.7] | 100% | 0.00 |
| dark ×0.1 | 600 | 3.2% | 20.8% | 14.7% | +11.5 | [+8.5, +14.5] | 100% | -0.01 |
