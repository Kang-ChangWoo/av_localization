# Validation-selected configuration, tested once: s3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| UnLoc | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 53.4 | 30.1 | 57.9 | +27.8 | [+23.7, +31.7] |
| UnLoc | three-scalar rule | none | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 58.0 | 30.1 | 58.4 | +28.3 | [+24.2, +32.6] |
| UnLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 58.0 | 30.1 | 58.4 | +28.3 | [+24.2, +32.4] |
| UnLoc | A: injected acoustic hypotheses | B | centre/quantile | simple | m=3 w=2 s=0.02 τv=0.2 | 64.8 | 30.1 | 66.5 | +36.4 | [+31.9, +40.9] |
| UnLoc | B: shortlist rejection | B | centre/quantile | simple | τg=0.5, used on 23% of test | 63.2 | 30.1 | 64.2 | +34.1 | [+29.6, +38.5] |
| UnLoc | C: cell-wise product | B | centre/quantile | simple | b1_T5 | 70.5 | 30.1 | 70.4 | +40.3 | [+35.7, +45.0] |
