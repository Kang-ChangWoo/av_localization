# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | R | max/centre | simple | w=2 s=0.02 τv=0.005 | 39.4 | 33.3 | 37.7 | +4.4 | [+2.2, +6.7] |
| F3Loc mono | no projection | none | centre/lse | simple | w=2 s=0.1 τv=0.005 | 35.6 | 33.3 | 36.7 | +3.3 | [+1.1, +5.6] |
| F3Loc mono | three-scalar rule | R | max/centre | simple | w=2 s=0.02 τv=0.005 | 39.4 | 33.3 | 37.7 | +4.4 | [+2.1, +6.6] |
| F3Loc mono | four-scalar rule | R | centre/lse | full | w=2 s=0.02 τv=0.02 τa=-2 | 39.8 | 33.3 | 39.1 | +5.7 | [+3.3, +8.1] |
| UnLoc | selected | T | max/max | simple | w=2 s=0.02 τv=0.005 | 52.1 | 45.3 | 49.9 | +4.6 | [+2.6, +6.6] |
| UnLoc | no projection | none | max/centre | full | w=1 s=0.05 τv=0.005 τa=-2 | 49.0 | 45.3 | 49.3 | +4.0 | [+2.0, +5.9] |
| UnLoc | three-scalar rule | T | max/max | simple | w=2 s=0.02 τv=0.005 | 52.1 | 45.3 | 49.9 | +4.6 | [+2.7, +6.6] |
| UnLoc | four-scalar rule | T | max/max | full | w=1 s=0.02 τv=0.02 τa=-2 | 51.9 | 45.3 | 49.5 | +4.2 | [+2.3, +6.0] |
| DisCo-FLoc RRP | selected | B | max/quantile | simple | w=2 s=0.02 τv=0.1 | 32.7 | 22.9 | 31.5 | +8.5 | [+5.8, +11.4] |
| DisCo-FLoc RRP | no projection | none | max/quantile | full | w=2 s=0.02 τv=0.05 τa=0 | 24.6 | 22.9 | 26.9 | +4.0 | [+1.2, +6.6] |
| DisCo-FLoc RRP | three-scalar rule | B | max/quantile | simple | w=2 s=0.02 τv=0.1 | 32.7 | 22.9 | 31.5 | +8.5 | [+5.8, +11.2] |
| DisCo-FLoc RRP | four-scalar rule | B | max/quantile | full | w=2 s=0.02 τv=0.05 τa=-2 | 32.7 | 22.9 | 30.3 | +7.4 | [+4.7, +10.1] |
