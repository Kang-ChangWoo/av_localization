# Vision plus binaural, against vision plus the ring

Same queries, same visual posterior, same fusion rule; only the acoustic candidate changes. 240 queries over 3 room(s), condition `raw_scan_open`. Scalars are fitted by leave-one-room-out where more than one room is present, and on the queries themselves otherwise, which is stated because it flatters every variant equally.


| acoustic candidate | calls/cell | grid | alone @1m | 0.1 m | 0.5 m | 1 m | 1m/30deg | 2 m | 5 m | median | RMSE | gain @1m | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vision only | – | – | – | 4.6 | 25.0 | 32.9 | 30.8 | 45.8 | 87.9 | 2.30 | 3.34 | – | – |
| ring | 1 | 8.0 GB | 23.3 | 2.9 | 18.8 | 29.2 | 24.6 | 41.7 | 82.9 | 2.81 | 3.67 | -3.8 | [-11.2, +3.8] |
| bin_best | 36 | 93.2 GB | 28.3 | 3.3 | 20.8 | 32.9 | 28.7 | 47.1 | 84.2 | 2.16 | 3.53 | +0.0 | [-7.5, +7.5] |
| bin_soft | 36 | 93.2 GB | 23.3 | 3.8 | 23.3 | 35.4 | 30.0 | 48.8 | 85.4 | 2.10 | 3.43 | +2.5 | [-5.0, +10.0] |

`bin_best` takes the best of the 36 headings at each cell, which needs no heading but is optimistic; `bin_soft` sums over them, which is what a deployment with no heading prior would do. `ring` is the shipped six-mic candidate and is the number in the main tables.

