# P. Domain-invariant ranking projection

Linear projection 1746 -> 128, trained on the 11 training rooms with an in-room InfoNCE over (furnished query, wall-only twin) pairs, selected on the validation rooms. In-room retrieval recall@1m on validation: identity 17.8%, projected 42.2%.

| test room | queries | acoustic alone @1m, identity | projected | median GT rank, identity | projected |
|---|---|---|---|---|---|
| apartment_2 | 300 | 21.0% | 16.0% | 245 | 260 |
| frl_apartment_5 | 300 | 9.0% | 48.3% | 626 | 79 |
| office_4 | 300 | 33.7% | 35.0% | 50 | 28 |
| **all** | 900 | 21.2% | 33.1% | 168 | 80 |

Acoustic score alone over the whole candidate grid, furnished query, test rooms never seen in training or selection. If the projected column is not above the identity column here, the feature cannot close the gap by a linear change of basis, whatever it does in-room.

