# Validation-selected configuration, tested once: s3d

Every discrete choice and every scalar is chosen on the validation rooms by recall at 1 m; the test rooms are evaluated once with the chosen configuration and nothing is refit on them. `no projection` and `four-scalar rule` are the best configurations under the same validation-only selection with that part fixed, so each part's contribution is measured without the test rooms.

| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| F3Loc mono | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 47.7 | 24.2 | 48.6 | +24.4 | [+20.3, +28.7] |
| F3Loc mono | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.05 | 47.7 | 24.2 | 48.6 | +24.4 | [+20.3, +28.5] |
| F3Loc mono | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 46.1 | 24.2 | 46.4 | +22.2 | [+18.1, +26.3] |
| F3Loc mono | A: injected acoustic hypotheses | B | centre/quantile | simple | m=3 w=2 s=0.02 τv=0.1 | 62.2 | 24.2 | 58.4 | +34.2 | [+29.6, +38.9] |
| F3Loc mono | B: shortlist rejection | B | centre/quantile | simple | τg=0.5, used on 32% of test | 61.7 | 24.2 | 59.0 | +34.8 | [+30.1, +39.2] |
| F3Loc mono | C: cell-wise product | B | centre/quantile | simple | b2_T5 | 68.9 | 24.2 | 65.9 | +41.8 | [+37.1, +46.4] |
| UnLoc | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 53.4 | 30.1 | 57.9 | +27.8 | [+23.7, +31.9] |
| UnLoc | three-scalar rule | none | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 58.0 | 30.1 | 58.4 | +28.3 | [+24.2, +32.4] |
| UnLoc | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 58.0 | 30.1 | 58.4 | +28.3 | [+24.2, +32.4] |
| UnLoc | A: injected acoustic hypotheses | B | centre/quantile | simple | m=3 w=2 s=0.02 τv=0.2 | 64.8 | 30.1 | 66.5 | +36.4 | [+31.7, +40.9] |
| UnLoc | B: shortlist rejection | B | centre/quantile | simple | τg=0.5, used on 23% of test | 63.2 | 30.1 | 64.2 | +34.1 | [+29.6, +38.5] |
| UnLoc | C: cell-wise product | B | centre/quantile | simple | b1_T5 | 70.5 | 30.1 | 70.4 | +40.3 | [+35.7, +45.0] |
| DisCo-FLoc RRP | selected | B | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 49.2 | 20.6 | 48.4 | +27.8 | [+23.5, +31.9] |
| DisCo-FLoc RRP | three-scalar rule | B | centre/quantile | simple | w=2 s=0.02 τv=0.2 | 49.2 | 20.6 | 48.4 | +27.8 | [+23.7, +31.9] |
| DisCo-FLoc RRP | no projection | none | centre/quantile | simple | w=2 s=0.02 τv=0.1 | 43.0 | 20.6 | 46.4 | +25.8 | [+21.5, +29.9] |
| DisCo-FLoc RRP | A: injected acoustic hypotheses | B | centre/quantile | simple | m=3 w=2 s=0.02 τv=0.2 | 61.7 | 20.6 | 58.8 | +38.2 | [+33.9, +42.5] |
| DisCo-FLoc RRP | B: shortlist rejection | B | centre/quantile | simple | τg=0.5, used on 33% of test | 61.1 | 20.6 | 59.5 | +38.9 | [+34.4, +43.4] |
| DisCo-FLoc RRP | C: cell-wise product | B | centre/quantile | simple | b2_T5 | 68.4 | 20.6 | 65.1 | +44.4 | [+40.0, +48.9] |
