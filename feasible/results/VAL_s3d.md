# Validation-selected configuration, tested once: s3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | B | max/max | simple | w=2 s=0.1 τv=0.2 | 53.9 | 24.2 | 51.4 | +27.2 | [+22.9, +31.5] |
| F3Loc mono | no projection | none | max/max | simple | w=2 s=0.02 τv=0.05 | 50.8 | 24.2 | 50.9 | +26.7 | [+22.6, +30.8] |
| F3Loc mono | three-scalar rule | B | max/max | simple | w=2 s=0.1 τv=0.2 | 53.9 | 24.2 | 51.4 | +27.2 | [+23.1, +31.5] |
| F3Loc mono | four-scalar rule | M | centre/max | full | w=2 s=0.02 τv=0.1 τa=-2 | 54.4 | 24.2 | 52.2 | +28.0 | [+23.7, +32.3] |
| UnLoc | selected | M | centre/max | simple | w=2 s=0.05 τv=0.2 | 60.1 | 30.1 | 61.5 | +31.4 | [+27.2, +35.7] |
| UnLoc | no projection | none | centre/lse | full | w=2 s=0.02 τv=0.2 τa=-2 | 59.1 | 30.1 | 57.5 | +27.4 | [+23.3, +31.5] |
| UnLoc | three-scalar rule | M | centre/max | simple | w=2 s=0.05 τv=0.2 | 60.1 | 30.1 | 61.5 | +31.4 | [+27.2, +35.3] |
| UnLoc | four-scalar rule | M | centre/max | full | w=2 s=0.02 τv=0.2 τa=0 | 60.6 | 30.1 | 62.0 | +31.9 | [+27.8, +36.2] |
| DisCo-FLoc RRP | selected | M | max/max | full | w=2 s=0.02 τv=0.05 τa=-2 | 54.4 | 20.6 | 51.8 | +31.2 | [+26.9, +35.5] |
| DisCo-FLoc RRP | no projection | none | centre/max | full | w=2 s=0.02 τv=0.1 τa=-2 | 51.8 | 20.6 | 50.9 | +30.3 | [+26.0, +34.6] |
| DisCo-FLoc RRP | three-scalar rule | R | centre/max | simple | w=2 s=0.02 τv=0.1 | 52.3 | 20.6 | 48.4 | +27.8 | [+23.8, +31.9] |
| DisCo-FLoc RRP | four-scalar rule | M | max/max | full | w=2 s=0.02 τv=0.05 τa=-2 | 54.4 | 20.6 | 51.8 | +31.2 | [+26.9, +35.7] |
