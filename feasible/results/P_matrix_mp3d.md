# Rooms held out, motion collections pooled

The `_f` and `_g` collections are two motion regimes over the same rooms. They are pooled here into one test set, because they are not spatially separated: on Replica, every general-motion pose has a forward-motion pose within the $1$ m threshold being reported, and the acoustic score depends on position alone. The four scalars are instead chosen by leave-one-room-out cross-validation, so every reported query comes from a room whose scalars were fitted without it.


| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |
|---|---|---|---|---|---|---|---|
| Matterport3D | f3loc_mono_mp3d12 | 12 | 960 | 33.3% | 37.4% | +4.1 | [+2.1, +6.1] |
| Matterport3D | f3loc_mono_mp3d12projR | 12 | 960 | 33.3% | 39.1% | +5.7 | [+3.3, +8.1] |
| Matterport3D | f3loc_mono_mp3d12proj | 12 | 960 | 33.3% | 40.3% | +7.0 | [+4.3, +9.6] |
| Matterport3D | f3loc_mono_mp3d12projB | 12 | 960 | 33.3% | 39.0% | +5.6 | [+3.3, +7.9] |
| Matterport3D | f3loc_mono_mp3d12projT | 12 | 960 | 33.3% | 38.0% | +4.7 | [+2.4, +7.0] |
| Matterport3D | unloc_mp3d12 | 12 | 960 | 45.3% | 49.2% | +3.9 | [+2.0, +5.7] |
| Matterport3D | unloc_mp3d12projR | 12 | 960 | 45.3% | 48.9% | +3.5 | [+1.2, +5.7] |
| Matterport3D | unloc_mp3d12proj | 12 | 960 | 45.3% | 50.6% | +5.3 | [+3.1, +7.5] |
| Matterport3D | unloc_mp3d12projB | 12 | 960 | 45.3% | 50.5% | +5.2 | [+3.2, +7.3] |
| Matterport3D | unloc_mp3d12projT | 12 | 960 | 45.3% | 48.8% | +3.4 | [+1.4, +5.6] |
| Matterport3D | disco_rrp_mp3d12 | 12 | 960 | 22.9% | 25.8% | +2.9 | [+0.5, +5.3] |
| Matterport3D | disco_rrp_mp3d12projR | 12 | 960 | 22.9% | 29.9% | +7.0 | [+4.1, +9.8] |
| Matterport3D | disco_rrp_mp3d12proj | 12 | 960 | 22.9% | 32.7% | +9.8 | [+7.1, +12.6] |
| Matterport3D | disco_rrp_mp3d12projB | 12 | 960 | 22.9% | 31.2% | +8.3 | [+5.6, +10.9] |
| Matterport3D | disco_rrp_mp3d12projT | 12 | 960 | 22.9% | 30.3% | +7.4 | [+4.7, +10.2] |

The structure chosen inside each fold is listed in the JSON. A structure that wins every fold is a property of the method; one that changes fold to fold would mean the summaries are not doing what we claim.

