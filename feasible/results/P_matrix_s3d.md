# Rooms held out, motion collections pooled

The `_f` and `_g` collections are two motion regimes over the same rooms. They are pooled here into one test set, because they are not spatially separated: on Replica, every general-motion pose has a forward-motion pose within the $1$ m threshold being reported, and the acoustic score depends on position alone. The four scalars are instead chosen by leave-one-room-out cross-validation, so every reported query comes from a room whose scalars were fitted without it.


| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |
|---|---|---|---|---|---|---|---|
| Structured3D | F3Loc mono | 30 | 558 | 24.2% | 46.8% | +22.6 | [+18.5, +26.7] |
| Replica | f3loc_mono_s3dprojR | 30 | 558 | 24.2% | 50.5% | +26.3 | [+22.2, +30.5] |
| Replica | f3loc_mono_s3dprojM | 30 | 558 | 24.2% | 50.7% | +26.5 | [+22.4, +30.6] |
| Replica | f3loc_mono_s3dprojB | 30 | 558 | 24.2% | 50.2% | +26.0 | [+21.9, +30.3] |
| Replica | f3loc_mono_s3dprojT | 30 | 558 | 24.2% | 51.6% | +27.4 | [+23.3, +31.5] |
| Structured3D | UnLoc | 30 | 558 | 30.1% | 58.2% | +28.1 | [+24.0, +32.3] |
| Replica | unloc_s3dprojR | 30 | 558 | 30.1% | 58.1% | +28.0 | [+24.0, +32.1] |
| Replica | unloc_s3dprojM | 30 | 558 | 30.1% | 58.2% | +28.1 | [+24.2, +32.3] |
| Replica | unloc_s3dprojB | 30 | 558 | 30.1% | 57.5% | +27.4 | [+23.3, +31.5] |
| Replica | unloc_s3dprojT | 30 | 558 | 30.1% | 57.7% | +27.6 | [+23.5, +31.7] |
| Structured3D | DisCo-FLoc RRP | 30 | 558 | 20.6% | 46.4% | +25.8 | [+21.7, +30.1] |
| Replica | disco_rrp_s3dprojR | 30 | 558 | 20.6% | 47.5% | +26.9 | [+22.8, +31.2] |
| Replica | disco_rrp_s3dprojM | 30 | 558 | 20.6% | 49.8% | +29.2 | [+25.1, +33.3] |
| Replica | disco_rrp_s3dprojB | 30 | 558 | 20.6% | 47.8% | +27.2 | [+23.1, +31.4] |
| Replica | disco_rrp_s3dprojT | 30 | 558 | 20.6% | 49.3% | +28.7 | [+24.6, +32.8] |

The structure chosen inside each fold is listed in the JSON. A structure that wins every fold is a property of the method; one that changes fold to fold would mean the summaries are not doing what we claim.

