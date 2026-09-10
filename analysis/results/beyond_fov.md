# C. Does the acoustic evidence come from beyond the field of view?

Condition `raw_scan_open`. Decidable pairs only: vision's two strongest hypotheses, exactly one of which is within 1 m of the true pose, so chance is 50%. Geometry is read from the same DESDF the localiser uses, split into the 11 yaw bins the camera spans and the 25 it does not.


## Marginal effect of each quantity (unlocSTFT, 193 pairs)

| quantity | tercile | n | acoustic correct | 95% CI | vision correct |
|---|---|---|---|---|---|
| visible similarity | low | 65 | **70.8%** | [58.8, 80.4] | 89.2% |
| visible similarity | mid | 64 | **60.9%** | [48.7, 71.9] | 76.6% |
| visible similarity | high | 64 | **71.9%** | [59.9, 81.4] | 65.6% |
| hidden difference | low | 65 | **61.5%** | [49.4, 72.4] | 73.8% |
| hidden difference | mid | 64 | **71.9%** | [59.9, 81.4] | 82.8% |
| hidden difference | high | 64 | **70.3%** | [58.2, 80.1] | 75.0% |

## The joint statement

Rows are hidden difference, columns visible similarity. The cell of interest is the top right: the camera sees the same thing at both places and the rest of the room differs. If acoustic accuracy is highest there, the acoustic evidence is coming from geometry the camera cannot see.

| hidden diff \ visible sim | low | mid | high |
|---|---|---|---|
| low | **54%** (n=13) | **48%** (n=21) | **74%** (n=31) |
| mid | **67%** (n=27) | **73%** (n=26) | **82%** (n=11) |
| high | **84%** (n=25) | **59%** (n=17) | **64%** (n=22) |

### The decisive contrast

Both groups below have *visually similar* competing hypotheses, so vision has little to work with in either. They differ only in whether the unseen part of the room differs.

| visible similarity | hidden difference | n | acoustic correct | 95% CI |
|---|---|---|---|---|
| high | high | 22 | **63.6%** | [43.0, 80.3] |
| high | low | 31 | **74.2%** | [56.8, 86.3] |

Difference: **-10.6 points**. A clearly positive value is the evidence the strong claim needs; a value near zero would mean the paper should claim only that acoustic consistency discriminates visually plausible hypotheses, without asserting where the information comes from.


## Replication across backbones

The geometry is a property of the room, so the effect should not depend on which visual model proposed the pair.

| backbone | pairs | hidden diff low | hidden diff high | difference |
|---|---|---|---|---|
| unlocSTFT | 193 | 61.5% | 70.3% | **+8.8** |
| f3STFT | 154 | 63.5% | 66.7% | **+3.2** |
| discoID | 162 | 70.4% | 75.9% | **+5.6** |
