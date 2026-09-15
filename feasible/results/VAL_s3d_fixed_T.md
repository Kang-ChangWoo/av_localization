# Validation-selected configuration, tested once: s3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | T | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 48.2 | 24.2 | 51.6 | +27.4 | [+23.3, +31.7] |
| F3Loc mono | three-scalar rule | T | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 48.2 | 24.2 | 51.6 | +27.4 | [+23.3, +31.5] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 46.1 | 24.2 | 46.4 | +22.2 | [+18.1, +26.3] |
| UnLoc | selected | T | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 53.4 | 30.1 | 57.7 | +27.6 | [+23.7, +31.7] |
| UnLoc | three-scalar rule | none | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 58.0 | 30.1 | 58.4 | +28.3 | [+24.2, +32.4] |
| UnLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 58.0 | 30.1 | 58.4 | +28.3 | [+24.0, +32.4] |
| DisCo-FLoc RRP | selected | T | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 46.1 | 20.6 | 48.0 | +27.4 | [+23.3, +31.7] |
| DisCo-FLoc RRP | three-scalar rule | T | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 46.1 | 20.6 | 48.0 | +27.4 | [+23.3, +31.5] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 43.0 | 20.6 | 46.4 | +25.8 | [+21.7, +30.1] |
