# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | B | centre/quantile | simple | w=2 s=0.05 τv=0.1 | 37.9 | 33.3 | 38.0 | +4.7 | [+2.0, +7.5] |
| F3Loc mono | three-scalar rule | B | centre/quantile | simple | w=2 s=0.05 τv=0.1 | 37.9 | 33.3 | 38.0 | +4.7 | [+1.9, +7.5] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 35.0 | 33.3 | 35.3 | +2.0 | [-0.5, +4.4] |
| UnLoc | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 50.4 | 45.3 | 52.1 | +6.8 | [+4.7, +8.9] |
| UnLoc | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 50.4 | 45.3 | 52.1 | +6.8 | [+4.7, +8.9] |
| UnLoc | no projection | none | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 47.7 | 45.3 | 49.6 | +4.3 | [+2.4, +6.2] |
| DisCo-FLoc RRP | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 31.2 | 22.9 | 30.9 | +8.0 | [+5.4, +10.7] |
| DisCo-FLoc RRP | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 31.2 | 22.9 | 30.9 | +8.0 | [+5.4, +10.6] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.005 | 22.7 | 22.9 | 27.1 | +4.2 | [+2.1, +6.4] |
