# Rooms held out, motion collections pooled

The `_f` and `_g` collections are two motion regimes over the same rooms. They are pooled here into one test set, because they are not spatially separated: on Replica, every general-motion pose has a forward-motion pose within the $1$ m threshold being reported, and the acoustic score depends on position alone. The four scalars are instead chosen by leave-one-room-out cross-validation, so every reported query comes from a room whose scalars were fitted without it.


| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |
|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | 3 | 600 | 38.3% | 44.8% | +6.5 | [+3.3, +9.8] |
| Replica | f3loc_mono_proj | 3 | 600 | 38.3% | 49.8% | +11.5 | [+8.0, +15.2] |
| Replica | UnLoc | 3 | 600 | 50.7% | 58.2% | +7.5 | [+4.3, +10.7] |
| Replica | unloc_uproj | 3 | 600 | 50.7% | 61.0% | +10.3 | [+6.7, +14.2] |
| Replica | DisCo-FLoc RRP | 3 | 600 | 40.5% | 43.2% | +2.7 | [-0.3, +5.7] |
| Replica | disco_rrpprojR | 3 | 600 | 40.5% | 51.8% | +11.3 | [+7.7, +15.2] |
| Replica | disco_rrpprojM | 3 | 600 | 40.5% | 46.3% | +5.8 | [+2.2, +9.5] |
| Replica | disco_rrpprojB | 3 | 600 | 40.5% | 48.8% | +8.3 | [+5.0, +11.7] |
| Replica | disco_rrpprojT | 3 | 600 | 40.5% | 48.2% | +7.7 | [+4.0, +11.3] |

The structure chosen inside each fold is listed in the JSON. A structure that wins every fold is a property of the method; one that changes fold to fold would mean the summaries are not doing what we claim.

