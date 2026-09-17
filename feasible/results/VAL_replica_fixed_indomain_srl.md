# Validation-selected configuration, tested once: replica

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SemRayLoc | selected | R | centre/quantile | simple | w=2 s=0.1 τv=0.05 | 41.3 | 29.0 | 38.7 | +9.7 | [+6.3, +13.0] |
| SemRayLoc | three-scalar rule | R | centre/quantile | simple | w=2 s=0.1 τv=0.05 | 41.3 | 29.0 | 38.7 | +9.7 | [+6.3, +13.0] |
| SemRayLoc | no projection | none | centre/quantile | simple | w=0.5 s=0.02 τv=0.1 | 37.2 | 29.0 | 32.5 | +3.5 | [+0.7, +6.3] |
