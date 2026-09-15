# Validation-selected configuration, tested once: mp3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| UnLoc | selected | M | centre/quantile | simple | w=0.5 s=0.1 τv=0.05 | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.3] |
| UnLoc | three-scalar rule | M | centre/quantile | simple | w=0.5 s=0.1 τv=0.05 | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.2] |
| UnLoc | no projection | none | centre/quantile | simple | w=1 s=0.05 τv=0.02 | 47.7 | 45.3 | 49.6 | +4.3 | [+2.4, +6.2] |
| UnLoc | A: injected acoustic hypotheses | M | centre/quantile | simple | m=2 w=1 s=0.05 τv=0.005 | 51.0 | 45.3 | 52.5 | +7.2 | [+4.7, +9.7] |
| UnLoc | B: shortlist rejection | M | centre/quantile | simple | τg=inf, used on 0% of test | 49.2 | 45.3 | 50.0 | +4.7 | [+3.2, +6.2] |
| UnLoc | C: cell-wise product | M | centre/quantile | simple | b2_T5 | 51.2 | 45.3 | 53.3 | +8.0 | [+5.3, +10.8] |
