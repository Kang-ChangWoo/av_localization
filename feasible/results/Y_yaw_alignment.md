# Y. Aligning the query ring to the candidate frame

Acoustic score alone over the whole grid. `none` compares channels as recorded (query ring rotates with the camera, candidates fixed at world yaw 0). `gt_yaw` rolls the query channels by round(yaw/60°) so both rings share a frame. Same queries, same grid, nothing else changes.

| condition | queries | alignment | recall @1 m | recall @2 m | median GT rank | mean GT rank |
|---|---|---|---|---|---|---|
| raw_scan_open | 360 | none | 22.2% | 32.5% | 162 | 542.3 |
| raw_scan_open | 360 | gt_yaw | 22.5% | 35.6% | 151 | 528.4 |

raw_scan_open: the GT rank is identical with and without alignment on 20.6% of queries.

| floorplan_closed | 360 | none | 94.2% | 94.4% | 0 | 1.5 |
| floorplan_closed | 360 | gt_yaw | 97.5% | 98.3% | 0 | 1.4 |

floorplan_closed: the GT rank is identical with and without alignment on 70.8% of queries.

