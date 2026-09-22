# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | M | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 38.1 | 33.3 | 40.1 | +6.8 | [+4.1, +9.6] |
| F3Loc mono | three-scalar rule | M | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 38.1 | 33.3 | 40.1 | +6.8 | [+4.1, +9.5] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 35.0 | 33.3 | 35.3 | +2.0 | [-0.5, +4.4] |
| UnLoc | selected | M | centre/quantile | simple | w=0.5 s=0.1 τv=0.05 | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.2] |
| UnLoc | three-scalar rule | M | centre/quantile | simple | w=0.5 s=0.1 τv=0.05 | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.2] |
| UnLoc | no projection | none | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 47.7 | 45.3 | 49.6 | +4.3 | [+2.4, +6.2] |
| DisCo-FLoc RRP | selected | M | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 26.5 | 21.0 | 31.5 | +10.4 | [+7.8, +13.0] |
| DisCo-FLoc RRP | three-scalar rule | M | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 26.5 | 21.0 | 31.5 | +10.4 | [+7.8, +13.0] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 24.8 | 21.0 | 25.2 | +4.2 | [+1.7, +6.8] |
