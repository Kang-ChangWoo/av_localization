# Validation-selected configuration, tested once: zind

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SemRayLoc | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 80.3 | 46.6 | 69.7 | +23.1 | [+21.7, +24.5] |
| SemRayLoc | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 80.3 | 46.6 | 69.7 | +23.1 | [+21.8, +24.5] |
| SemRayLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 77.8 | 46.6 | 69.7 | +23.2 | [+21.8, +24.6] |
