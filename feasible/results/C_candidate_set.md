# C. Global acoustic localization against hypothesis verification

Identical queries throughout; only the set of places the acoustic score is asked to choose between changes. `rank` is the position of the cell nearest ground truth in the acoustic ordering over that set, so 0 is a correct first choice.


## F3Loc mono

| candidate set | size | acoustic choice @1 m | gated fusion @1 m | oracle @1 m |
|---|---|---|---|---|
| vision alone (no acoustics) | 1 | 38.3% | 38.3% | – |
| hypotheses, K=1 | 1 | 38.3% | 38.3% | 38.3% |
| hypotheses, K=2 | 2 | 35.8% | 40.2% | 52.2% |
| hypotheses, K=3 | 3 | 34.2% | 43.7% | 62.8% |
| hypotheses, K=5 | 5 | 31.8% | 45.0% | 73.3% |
| hypotheses, K=10 | 10 | 26.3% | 46.8% | 86.2% |
| visual top-50 cells | 50 | 46.0% | – | – |
| all cells, acoustic only | 5344 | 20.8% | – | – |

## UnLoc

| candidate set | size | acoustic choice @1 m | gated fusion @1 m | oracle @1 m |
|---|---|---|---|---|
| vision alone (no acoustics) | 1 | 50.7% | 50.7% | – |
| hypotheses, K=1 | 1 | 50.7% | 50.7% | 50.7% |
| hypotheses, K=2 | 2 | 47.5% | 51.0% | 65.3% |
| hypotheses, K=3 | 3 | 46.0% | 53.2% | 76.3% |
| hypotheses, K=5 | 5 | 41.3% | 54.7% | 85.5% |
| hypotheses, K=10 | 10 | 35.3% | 58.3% | 92.8% |
| visual top-50 cells | 50 | 52.2% | – | – |
| all cells, acoustic only | 5344 | 20.8% | – | – |

## DisCo-FLoc RRP

| candidate set | size | acoustic choice @1 m | gated fusion @1 m | oracle @1 m |
|---|---|---|---|---|
| vision alone (no acoustics) | 1 | 40.5% | 40.5% | – |
| hypotheses, K=1 | 1 | 40.5% | 40.5% | 40.5% |
| hypotheses, K=2 | 2 | 36.7% | 39.7% | 53.7% |
| hypotheses, K=3 | 3 | 35.2% | 42.2% | 66.7% |
| hypotheses, K=5 | 5 | 33.0% | 43.5% | 73.2% |
| hypotheses, K=10 | 10 | 30.8% | 46.5% | 84.5% |
| visual top-50 cells | 50 | 45.8% | – | – |
| all cells, acoustic only | 5344 | 20.8% | – | – |

The curve falls with candidate-set size on every backbone: the same acoustic evidence that cannot find a pose on a map can choose between a handful of them. The distance to the oracle at small K is what better acoustic discrimination would buy without changing anything else.

