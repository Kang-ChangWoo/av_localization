# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | T | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 37.7 | 33.3 | 38.4 | +5.1 | [+2.7, +7.5] |
| F3Loc mono | three-scalar rule | T | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 37.7 | 33.3 | 38.4 | +5.1 | [+2.7, +7.6] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 35.0 | 33.3 | 35.3 | +2.0 | [-0.5, +4.4] |
| UnLoc | selected | T | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 50.4 | 45.3 | 50.1 | +4.8 | [+3.0, +6.7] |
| UnLoc | three-scalar rule | T | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 50.4 | 45.3 | 50.1 | +4.8 | [+3.0, +6.7] |
| UnLoc | no projection | none | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 47.7 | 45.3 | 49.6 | +4.3 | [+2.4, +6.2] |
| DisCo-FLoc RRP | selected | T | centre/quantile | simple | w=2 s=0.05 τv=0.1 | 29.0 | 22.9 | 30.2 | +7.3 | [+4.6, +10.0] |
| DisCo-FLoc RRP | three-scalar rule | T | centre/quantile | simple | w=2 s=0.05 τv=0.1 | 29.0 | 22.9 | 30.2 | +7.3 | [+4.6, +10.0] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.005 | 22.7 | 22.9 | 27.1 | +4.2 | [+2.1, +6.4] |
