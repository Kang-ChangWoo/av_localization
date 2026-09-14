# Rooms held out, motion collections pooled

The `_f` and `_g` collections are two motion regimes over the same rooms. They are pooled here into one test set, because they are not spatially separated: on Replica, every general-motion pose has a forward-motion pose within the $1$ m threshold being reported, and the acoustic score depends on position alone. The four scalars are instead chosen by leave-one-room-out cross-validation, so every reported query comes from a room whose scalars were fitted without it.


| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |
|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | 3 | 600 | 38.3% | 45.5% | +7.2 | [+3.8, +10.5] |
| Replica | UnLoc | 3 | 600 | 50.7% | 58.0% | +7.3 | [+4.3, +10.5] |
| Replica | DisCo-FLoc RRP | 3 | 600 | 40.5% | 40.5% | +0.0 | [-2.7, +2.7] |

The structure chosen inside each fold is listed in the JSON. A structure that wins every fold is a property of the method; one that changes fold to fold would mean the summaries are not doing what we claim.

