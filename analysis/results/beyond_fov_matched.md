# C. Does the acoustic evidence come from beyond the field of view?

Condition `floorplan_closed`. Decidable pairs only: vision's two strongest hypotheses, exactly one of which is within 1 m of the true pose, so chance is 50%. Geometry is read from the same DESDF the localiser uses, split into the 11 yaw bins the camera spans and the 25 it does not.


## Marginal effect of each quantity (unlocSTFT, 193 pairs)

| quantity | tercile | n | acoustic correct | 95% CI | vision correct |
|---|---|---|---|---|---|
| visible similarity | low | 65 | **93.8%** | [85.2, 97.6] | 89.2% |
| visible similarity | mid | 64 | **90.6%** | [81.0, 95.6] | 76.6% |
| visible similarity | high | 64 | **98.4%** | [91.7, 99.7] | 65.6% |
| hidden difference | low | 65 | **90.8%** | [81.3, 95.7] | 73.8% |
| hidden difference | mid | 64 | **93.8%** | [85.0, 97.5] | 82.8% |
| hidden difference | high | 64 | **98.4%** | [91.7, 99.7] | 75.0% |

## The joint statement

Rows are hidden difference, columns visible similarity. The cell of interest is the top right: the camera sees the same thing at both places and the rest of the room differs. If acoustic accuracy is highest there, the acoustic evidence is coming from geometry the camera cannot see.

| hidden diff \ visible sim | low | mid | high |
|---|---|---|---|
| low | **77%** (n=13) | **86%** (n=21) | **100%** (n=31) |
| mid | **96%** (n=27) | **92%** (n=26) | **91%** (n=11) |
| high | **100%** (n=25) | **94%** (n=17) | **100%** (n=22) |

### The decisive contrast

Both groups below have *visually similar* competing hypotheses, so vision has little to work with in either. They differ only in whether the unseen part of the room differs.

| visible similarity | hidden difference | n | acoustic correct | 95% CI |
|---|---|---|---|---|
| high | high | 22 | **100.0%** | [85.1, 100.0] |
| high | low | 31 | **100.0%** | [89.0, 100.0] |

Difference: **+0.0 points**. A clearly positive value is the evidence the strong claim needs; a value near zero would mean the paper should claim only that acoustic consistency discriminates visually plausible hypotheses, without asserting where the information comes from.


## Replication across backbones

The geometry is a property of the room, so the effect should not depend on which visual model proposed the pair.

| backbone | pairs | hidden diff low | hidden diff high | difference |
|---|---|---|---|---|
| unlocSTFT | 193 | 90.8% | 98.4% | **+7.7** |
| f3STFT | 154 | 82.7% | 90.2% | **+7.5** |
| discoSTFT | 141 | 83.0% | 89.4% | **+6.4** |
