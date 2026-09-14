# Rooms held out, motion collections pooled

The `_f` and `_g` collections are two motion regimes over the same rooms. They are pooled here into one test set, because they are not spatially separated: on Replica, every general-motion pose has a forward-motion pose within the $1$ m threshold being reported, and the acoustic score depends on position alone. The four scalars are instead chosen by leave-one-room-out cross-validation, so every reported query comes from a room whose scalars were fitted without it.


| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |
|---|---|---|---|---|---|---|---|
| Matterport3D | f3loc_mono_mp3d12 | 12 | 960 | 33.3% | 37.4% | +4.1 | [+2.1, +6.1] |
| Matterport3D | f3loc_mv_mp3d12 | 12 | 960 | 32.0% | 37.5% | +5.5 | [+3.1, +7.9] |
| Matterport3D | f3loc_comp_mp3d12 | 12 | 960 | 34.0% | 37.6% | +3.6 | [+1.1, +6.0] |
| Matterport3D | unloc_mp3d12 | 12 | 960 | 45.3% | 49.2% | +3.9 | [+2.0, +5.7] |
| Matterport3D | disco_rrp_mp3d12 | 12 | 960 | 22.9% | 25.8% | +2.9 | [+0.5, +5.2] |

The structure chosen inside each fold is listed in the JSON. A structure that wins every fold is a property of the method; one that changes fold to fold would mean the summaries are not doing what we claim.

