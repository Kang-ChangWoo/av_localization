# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SemRayLoc | selected | M | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 30.0 | 22.8 | 28.6 | +5.8 | [+3.2, +8.4] |
| SemRayLoc | three-scalar rule | M | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 30.0 | 22.8 | 28.6 | +5.8 | [+3.2, +8.4] |
| SemRayLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.005 | 27.3 | 22.8 | 26.2 | +3.4 | [+1.2, +5.6] |
