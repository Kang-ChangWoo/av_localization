# Validation-selected configuration, tested once: s3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SemRayLoc | selected | B | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 57.5 | 32.6 | 55.4 | +22.8 | [+18.8, +26.7] |
| SemRayLoc | three-scalar rule | B | centre/quantile | simple | w=2 s=0.05 τv=0.05 | 57.5 | 32.6 | 55.4 | +22.8 | [+18.6, +26.7] |
| SemRayLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 54.9 | 32.6 | 53.6 | +21.0 | [+17.0, +25.1] |
