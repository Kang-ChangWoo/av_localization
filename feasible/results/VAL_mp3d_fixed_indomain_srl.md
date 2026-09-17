# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SemRayLoc | selected | M | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 29.6 | 22.2 | 28.7 | +6.6 | [+4.0, +9.2] |
| SemRayLoc | three-scalar rule | M | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 29.6 | 22.2 | 28.7 | +6.6 | [+4.0, +9.2] |
| SemRayLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.005 | 26.9 | 22.2 | 27.2 | +5.0 | [+2.8, +7.2] |
