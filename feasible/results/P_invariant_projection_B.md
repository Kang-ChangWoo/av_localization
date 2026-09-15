# P. Domain-invariant ranking projection

Linear projection 1746 -> 128, trained on the 142 training rooms with an in-room InfoNCE over (furnished query, wall-only twin) pairs, selected on the validation rooms. In-room retrieval recall@1m on validation: identity 17.3%, projected 33.9%.

| test room | queries | acoustic alone @1m, identity | projected | median GT rank, identity | projected |
|---|---|---|---|---|---|

Acoustic score alone over the whole candidate grid, furnished query, test rooms never seen in training or selection. If the projected column is not above the identity column here, the feature cannot close the gap by a linear change of basis, whatever it does in-room.

