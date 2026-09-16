# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | M | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 38.1 | 33.3 | 40.1 | +6.8 | [+4.1, +9.6] |
| F3Loc mono | three-scalar rule | M | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 38.1 | 33.3 | 40.1 | +6.8 | [+4.1, +9.5] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 35.0 | 33.3 | 35.3 | +2.0 | [-0.5, +4.4] |
| F3Loc mono | A: injected acoustic hypotheses | M | centre/quantile | simple | m=2 w=1 s=0.1 τv=0.2 | 39.2 | 33.3 | 39.9 | +6.6 | [+3.6, +9.5] |
| F3Loc mono | B: shortlist rejection | M | centre/quantile | simple | τg=inf, used on 0% of test | 38.1 | 33.3 | 40.1 | +6.8 | [+4.1, +9.5] |
| F3Loc mono | C: cell-wise product | M | centre/quantile | simple | b2_T5 | 37.1 | 33.3 | 37.8 | +4.5 | [+1.2, +7.8] |
| UnLoc | selected | M | centre/quantile | simple | w=0.5 s=0.1 τv=0.05 | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.2] |
| UnLoc | three-scalar rule | M | centre/quantile | simple | w=0.5 s=0.1 τv=0.05 | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.2] |
| UnLoc | no projection | none | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 47.7 | 45.3 | 49.6 | +4.3 | [+2.4, +6.2] |
| UnLoc | A: injected acoustic hypotheses | M | centre/quantile | simple | m=2 w=1 s=0.05 τv=0.005 | 51.0 | 45.3 | 52.5 | +7.2 | [+4.8, +9.7] |
| UnLoc | B: shortlist rejection | M | centre/quantile | simple | τg=inf, used on 0% of test | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.2] |
| UnLoc | C: cell-wise product | M | centre/quantile | simple | b2_T5 | 51.2 | 45.3 | 53.3 | +8.0 | [+5.3, +10.7] |
| DisCo-FLoc RRP | selected | M | centre/quantile | simple | w=2 s=0.05 τv=0.1 | 29.6 | 22.9 | 32.9 | +10.0 | [+7.4, +12.7] |
| DisCo-FLoc RRP | three-scalar rule | M | centre/quantile | simple | w=2 s=0.05 τv=0.1 | 29.6 | 22.9 | 32.9 | +10.0 | [+7.4, +12.7] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.005 | 22.7 | 22.9 | 27.1 | +4.2 | [+2.0, +6.4] |
| DisCo-FLoc RRP | A: injected acoustic hypotheses | M | centre/quantile | simple | m=3 w=1 s=0.02 τv=0.05 | 32.1 | 22.9 | 32.1 | +9.2 | [+6.6, +11.9] |
| DisCo-FLoc RRP | B: shortlist rejection | M | centre/quantile | simple | τg=2, used on 8% of test | 31.0 | 22.9 | 32.0 | +9.1 | [+6.2, +11.9] |
| DisCo-FLoc RRP | C: cell-wise product | M | centre/quantile | simple | b2_T5 | 29.0 | 22.9 | 31.8 | +8.9 | [+5.8, +11.9] |
