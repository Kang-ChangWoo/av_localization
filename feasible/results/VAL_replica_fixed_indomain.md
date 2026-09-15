# Validation-selected configuration, tested once: replica

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| UnLoc | selected | R | centre/quantile | simple | w=1 s=0.1 τv=0.1 | 55.7 | 50.7 | 60.0 | +9.3 | [+6.3, +12.5] |
| UnLoc | three-scalar rule | R | centre/quantile | simple | w=1 s=0.1 τv=0.1 | 55.7 | 50.7 | 60.0 | +9.3 | [+6.2, +12.5] |
| UnLoc | no projection | none | centre/quantile | simple | w=0.5 s=0.02 τv=0.05 | 52.5 | 50.7 | 55.8 | +5.2 | [+2.7, +7.7] |
| UnLoc | A: injected acoustic hypotheses | R | centre/quantile | simple | m=3 w=0.5 s=0.05 τv=0.05 | 57.7 | 50.7 | 59.0 | +8.3 | [+5.3, +11.3] |
| UnLoc | B: shortlist rejection | R | centre/quantile | simple | τg=inf, used on 0% of test | 55.7 | 50.7 | 60.0 | +9.3 | [+6.3, +12.5] |
| UnLoc | C: cell-wise product | R | centre/quantile | simple | b1_T2 | 56.5 | 50.7 | 60.5 | +9.8 | [+5.7, +13.8] |
