# Rooms held out, motion collections pooled

The `_f` and `_g` collections are two motion regimes over the same rooms. They are pooled here into one test set, because they are not spatially separated: on Replica, every general-motion pose has a forward-motion pose within the $1$ m threshold being reported, and the acoustic score depends on position alone. The four scalars are instead chosen by leave-one-room-out cross-validation, so every reported query comes from a room whose scalars were fitted without it.


| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |
|---|---|---|---|---|---|---|---|
| Matterport3D | f3loc_mono_mp3d12 | 12 | 960 | 33.3% | 37.4% | +4.1 | [+2.1, +6.1] |
| Matterport3D | f3loc_mono_mp3d12proj | 12 | 960 | 33.3% | 40.3% | +7.0 | [+4.4, +9.6] |
| Matterport3D | unloc_mp3d12 | 12 | 960 | 45.3% | 49.2% | +3.9 | [+2.0, +5.7] |
| Matterport3D | unloc_mp3d12proj | 12 | 960 | 45.3% | 50.6% | +5.3 | [+3.1, +7.6] |

The structure chosen inside each fold is listed in the JSON. A structure that wins every fold is a property of the method; one that changes fold to fold would mean the summaries are not doing what we claim.

