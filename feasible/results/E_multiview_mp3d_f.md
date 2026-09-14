# Rooms held out, motion collections pooled

The `_f` and `_g` collections are two motion regimes over the same rooms. They are pooled here into one test set, because they are not spatially separated: on Replica, every general-motion pose has a forward-motion pose within the $1$ m threshold being reported, and the acoustic score depends on position alone. The four scalars are instead chosen by leave-one-room-out cross-validation, so every reported query comes from a room whose scalars were fitted without it.


| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |
|---|---|---|---|---|---|---|---|
| Matterport3D | f3loc_mono_mp3d12 | 12 | 480 | 31.0% | 33.8% | +2.7 | [+0.2, +5.2] |
| Matterport3D | f3loc_mv_mp3d12 | 12 | 480 | 43.3% | 49.0% | +5.6 | [+2.9, +8.5] |
| Matterport3D | f3loc_comp_mp3d12 | 12 | 480 | 31.7% | 34.2% | +2.5 | [-1.0, +5.8] |

The structure chosen inside each fold is listed in the JSON. A structure that wins every fold is a property of the method; one that changes fold to fold would mean the summaries are not doing what we claim.

