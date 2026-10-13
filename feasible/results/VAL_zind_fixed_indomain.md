# Validation-selected configuration, tested once: zind

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 73.5 | 44.0 | 68.1 | +24.1 | [+22.7, +25.5] |
| F3Loc mono | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 73.5 | 44.0 | 68.1 | +24.1 | [+22.7, +25.5] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 73.5 | 44.0 | 69.1 | +25.1 | [+23.7, +26.5] |
| UnLoc | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 74.4 | 48.4 | 71.1 | +22.7 | [+21.3, +24.1] |
| UnLoc | three-scalar rule | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 75.2 | 48.4 | 72.6 | +24.2 | [+22.8, +25.6] |
| UnLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 75.2 | 48.4 | 72.6 | +24.2 | [+22.8, +25.6] |
| DisCo-FLoc RRP | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 71.8 | 39.1 | 65.6 | +26.4 | [+25.0, +27.9] |
| DisCo-FLoc RRP | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 71.8 | 39.1 | 65.6 | +26.4 | [+25.0, +27.9] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 70.9 | 39.1 | 66.1 | +27.0 | [+25.5, +28.4] |
