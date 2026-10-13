# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SemRayLoc | selected | M | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 29.2 | 24.9 | 31.8 | +6.9 | [+4.5, +9.4] |
| SemRayLoc | three-scalar rule | M | centre/quantile | simple | w=2 s=0.02 τv=0.02 | 29.2 | 24.9 | 31.8 | +6.9 | [+4.5, +9.3] |
| SemRayLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.005 | 27.1 | 24.9 | 29.8 | +4.9 | [+2.8, +7.1] |
