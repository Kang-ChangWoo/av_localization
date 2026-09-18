# Validation-selected configuration, tested once: replica

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SemRayLoc | selected | R | centre/quantile | simple | w=2 s=0.05 τv=0.02 | 46.7 | 35.0 | 42.3 | +7.3 | [+4.0, +10.5] |
| SemRayLoc | three-scalar rule | R | centre/quantile | simple | w=2 s=0.05 τv=0.02 | 46.7 | 35.0 | 42.3 | +7.3 | [+4.0, +10.7] |
| SemRayLoc | no projection | none | centre/quantile | simple | w=0.5 s=0.05 τv=0.02 | 41.7 | 35.0 | 37.8 | +2.8 | [+0.3, +5.3] |
