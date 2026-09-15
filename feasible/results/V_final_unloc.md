# Visual degradation under the final protocol: UnLoc, Replica

Projection R and the scalars chosen on the clean validation rooms are frozen (ModeFusionConfig(vis_evidence='centre', ac_evidence='quantile', rule='continuous', weight=1.0, tau_v=0.1, tau_a=-2.0, sigmoid_scale=0.1, lam=1.0, ac_transform='standard')); the query image is corrupted at test time and nothing is refit. `acoustic alone` is the projected score's own top-1 over the whole grid; `audio acts` is the fraction of queries where the gate weight exceeds 0.05.

| degradation | queries | vision @1m | acoustic alone | ours @1m | gain | 95% CI | audio acts |
|---|---|---|---|---|---|---|---|
| clean | 600 | 50.7% | 36.0% | 60.0% | +9.3 | [+6.3, +12.5] | 91% |
| blur σ=1 | 600 | 50.3% | 36.0% | 59.2% | +8.8 | [+5.7, +12.2] | 93% |
| blur σ=2 | 600 | 48.0% | 36.0% | 56.8% | +8.8 | [+5.5, +12.3] | 94% |
| blur σ=4 | 600 | 42.8% | 36.0% | 51.3% | +8.5 | [+5.0, +12.0] | 96% |
| blur σ=8 | 600 | 28.0% | 36.0% | 37.8% | +9.8 | [+6.5, +13.2] | 100% |
| dark ×0.75 | 600 | 50.5% | 36.0% | 57.5% | +7.0 | [+3.8, +10.3] | 91% |
| dark ×0.5 | 600 | 47.2% | 36.0% | 55.3% | +8.2 | [+5.0, +11.5] | 92% |
| dark ×0.25 | 600 | 44.2% | 36.0% | 53.0% | +8.8 | [+5.7, +12.0] | 93% |
| dark ×0.1 | 600 | 38.5% | 36.0% | 49.2% | +10.7 | [+7.3, +14.0] | 95% |
| noise σ=10 | 600 | 47.2% | 36.0% | 54.8% | +7.7 | [+4.5, +10.8] | 92% |
| noise σ=25 | 600 | 44.3% | 36.0% | 53.8% | +9.5 | [+6.2, +12.8] | 93% |
| noise σ=50 | 600 | 41.7% | 36.0% | 49.7% | +8.0 | [+4.7, +11.5] | 96% |
| occlude 10% | 600 | 34.3% | 36.0% | 47.0% | +12.7 | [+9.2, +16.3] | 98% |
| occlude 30% | 600 | 15.7% | 36.0% | 22.3% | +6.7 | [+3.8, +9.5] | 100% |
| occlude 50% | 600 | 11.0% | 36.0% | 18.0% | +7.0 | [+4.3, +9.7] | 100% |
| downscale 2× | 600 | 49.5% | 36.0% | 59.2% | +9.7 | [+6.3, +13.0] | 92% |
| downscale 4× | 600 | 47.7% | 36.0% | 54.8% | +7.2 | [+3.8, +10.5] | 93% |
| downscale 8× | 600 | 42.5% | 36.0% | 52.0% | +9.5 | [+6.3, +12.8] | 96% |
