# Validation-selected configuration, tested once: replica

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | B | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 48.0 | 38.3 | 49.3 | +11.0 | [+7.5, +14.5] |
| F3Loc mono | three-scalar rule | B | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 48.0 | 38.3 | 49.3 | +11.0 | [+7.5, +14.7] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=0.5 s=0.05 τv=0.05 | 43.5 | 38.3 | 44.5 | +6.2 | [+3.5, +8.8] |
| UnLoc | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 50.4 | 50.7 | 58.5 | +7.8 | [+5.0, +10.8] |
| UnLoc | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 50.4 | 50.7 | 58.5 | +7.8 | [+5.0, +10.8] |
| UnLoc | no projection | none | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 47.7 | 50.7 | 57.0 | +6.3 | [+3.7, +9.2] |
| DisCo-FLoc RRP | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 45.0 | 40.5 | 48.7 | +8.2 | [+4.3, +12.0] |
| DisCo-FLoc RRP | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 45.0 | 40.5 | 48.7 | +8.2 | [+4.3, +12.0] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=0.5 s=0.05 τv=0.05 | 42.2 | 40.5 | 43.3 | +2.8 | [-0.2, +5.8] |
