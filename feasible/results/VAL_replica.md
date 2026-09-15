# Validation-selected configuration, tested once: replica

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | R | max/lse | simple | w=2 s=0.05 τv=0.05 | 50.5 | 38.3 | 47.2 | +8.8 | [+5.3, +12.2] |
| F3Loc mono | no projection | none | max/max | simple | w=0.5 s=0.1 τv=0.1 | 44.5 | 38.3 | 43.7 | +5.3 | [+2.8, +7.8] |
| F3Loc mono | three-scalar rule | R | max/lse | simple | w=2 s=0.05 τv=0.05 | 50.5 | 38.3 | 47.2 | +8.8 | [+5.5, +12.2] |
| F3Loc mono | four-scalar rule | R | max/lse | full | w=2 s=0.1 τv=0.1 τa=0.2 | 51.0 | 38.3 | 46.8 | +8.5 | [+5.3, +11.7] |
| UnLoc | selected | B | centre/quantile | simple | w=1 s=0.05 τv=0.05 | 58.2 | 50.7 | 58.2 | +7.5 | [+4.7, +10.5] |
| UnLoc | no projection | none | centre/lse | full | w=1 s=0.02 τv=0.05 τa=0 | 53.2 | 50.7 | 57.2 | +6.5 | [+3.7, +9.5] |
| UnLoc | three-scalar rule | B | centre/quantile | simple | w=1 s=0.05 τv=0.05 | 58.2 | 50.7 | 58.2 | +7.5 | [+4.7, +10.5] |
| UnLoc | four-scalar rule | B | centre/quantile | full | w=1 s=0.05 τv=0.005 τa=-2 | 58.2 | 50.7 | 57.2 | +6.5 | [+3.7, +9.3] |
| DisCo-FLoc RRP | selected | R | centre/max | simple | w=1 s=0.05 τv=0.1 | 49.2 | 40.5 | 50.0 | +9.5 | [+6.0, +13.2] |
| DisCo-FLoc RRP | no projection | none | centre/centre | simple | w=0.5 s=0.02 τv=0.02 | 42.7 | 40.5 | 43.5 | +3.0 | [+0.0, +6.0] |
| DisCo-FLoc RRP | three-scalar rule | R | centre/max | simple | w=1 s=0.05 τv=0.1 | 49.2 | 40.5 | 50.0 | +9.5 | [+6.0, +13.2] |
| DisCo-FLoc RRP | four-scalar rule | R | centre/max | full | w=1 s=0.05 τv=0.05 τa=-2 | 49.2 | 40.5 | 50.2 | +9.7 | [+6.2, +13.3] |
