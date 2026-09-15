# Validation-selected configuration, tested once: replica

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | T | centre/quantile | simple | w=1 s=0.02 τv=0.02 | 47.8 | 38.3 | 47.3 | +9.0 | [+5.8, +12.2] |
| F3Loc mono | three-scalar rule | T | centre/quantile | simple | w=1 s=0.02 τv=0.02 | 47.8 | 38.3 | 47.3 | +9.0 | [+5.8, +12.2] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=0.5 s=0.05 τv=0.05 | 43.5 | 38.3 | 44.5 | +6.2 | [+3.5, +8.8] |
| UnLoc | selected | T | centre/quantile | simple | w=1 s=0.02 τv=0.05 | 57.3 | 50.7 | 59.3 | +8.7 | [+5.7, +11.8] |
| UnLoc | three-scalar rule | T | centre/quantile | simple | w=1 s=0.02 τv=0.05 | 57.3 | 50.7 | 59.3 | +8.7 | [+5.7, +11.7] |
| UnLoc | no projection | none | centre/quantile | simple | w=0.5 s=0.02 τv=0.05 | 52.5 | 50.7 | 55.8 | +5.2 | [+2.7, +7.7] |
| DisCo-FLoc RRP | selected | T | centre/quantile | simple | w=1 s=0.02 τv=0.02 | 45.8 | 40.5 | 47.3 | +6.8 | [+3.7, +10.0] |
| DisCo-FLoc RRP | three-scalar rule | T | centre/quantile | simple | w=1 s=0.02 τv=0.02 | 45.8 | 40.5 | 47.3 | +6.8 | [+3.7, +10.0] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=0.5 s=0.05 τv=0.05 | 42.2 | 40.5 | 43.3 | +2.8 | [-0.2, +5.8] |
