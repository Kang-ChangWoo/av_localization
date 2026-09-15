# Validation-selected configuration, tested once: replica

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | R | centre/quantile | simple | w=2 s=0.05 τv=0.02 | 48.8 | 38.3 | 48.8 | +10.5 | [+7.0, +14.0] |
| F3Loc mono | three-scalar rule | R | centre/quantile | simple | w=2 s=0.05 τv=0.02 | 48.8 | 38.3 | 48.8 | +10.5 | [+7.0, +14.0] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=0.5 s=0.05 τv=0.05 | 43.5 | 38.3 | 44.5 | +6.2 | [+3.5, +8.8] |
| UnLoc | selected | R | centre/quantile | simple | w=1 s=0.1 τv=0.1 | 55.7 | 50.7 | 60.0 | +9.3 | [+6.3, +12.5] |
| UnLoc | three-scalar rule | R | centre/quantile | simple | w=1 s=0.1 τv=0.1 | 55.7 | 50.7 | 60.0 | +9.3 | [+6.3, +12.5] |
| UnLoc | no projection | none | centre/quantile | simple | w=0.5 s=0.02 τv=0.05 | 52.5 | 50.7 | 55.8 | +5.2 | [+2.7, +7.7] |
| DisCo-FLoc RRP | selected | R | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 47.5 | 40.5 | 52.3 | +11.8 | [+8.3, +15.5] |
| DisCo-FLoc RRP | three-scalar rule | R | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 47.5 | 40.5 | 52.3 | +11.8 | [+8.2, +15.5] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=0.5 s=0.05 τv=0.05 | 42.2 | 40.5 | 43.3 | +2.8 | [-0.2, +5.8] |
