# Validation-selected configuration, tested once: gibson

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| UnLoc | selected | G | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 68.5 | 51.7 | 64.0 | +12.3 | [+11.2, +13.4] |
| UnLoc | three-scalar rule | G | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 68.5 | 51.7 | 64.0 | +12.3 | [+11.2, +13.4] |
| UnLoc | no projection | none | centre/quantile | simple | w=2 s=0.05 τv=0.02 | 63.3 | 51.7 | 55.5 | +3.8 | [+2.7, +4.9] |
| F3Loc mono | selected | G | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 67.3 | 47.0 | 59.7 | +12.7 | [+11.5, +13.9] |
| F3Loc mono | three-scalar rule | G | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 67.3 | 47.0 | 59.7 | +12.7 | [+11.5, +13.9] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=1 s=0.1 τv=0.2 | 59.8 | 47.0 | 49.5 | +2.5 | [+1.3, +3.7] |
| DisCo-FLoc RRP | selected | G | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 60.0 | 40.7 | 53.5 | +12.8 | [+11.6, +14.0] |
| DisCo-FLoc RRP | three-scalar rule | G | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 60.0 | 40.7 | 53.5 | +12.8 | [+11.6, +14.0] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=1 s=0.05 τv=0.05 | 54.0 | 40.7 | 43.4 | +2.8 | [+1.7, +3.9] |
