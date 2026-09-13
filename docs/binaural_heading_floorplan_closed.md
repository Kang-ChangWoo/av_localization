# Does binaural give heading

The ring cannot: its channels are near-identical, so the acoustic score is flat across the yaw bins and orientation comes from vision. A binaural candidate grid is rendered once per (cell, heading), which is six times the engine calls per cell, so the question is what that buys.


## Oracle heading at the ground-truth cell

The binaural score is evaluated at the true cell across all 36 headings and the best one is taken. Nothing downstream can beat this.


| queries | within 10 deg | within 30 deg | within 45 deg | median error |
|---|---|---|---|---|
| 120 | 4.2% | 19.2% | 30.0% | 99.3 deg |

A uniformly random heading lands within 30 deg 16.7% of the time, so the number above is the one to compare against that.

