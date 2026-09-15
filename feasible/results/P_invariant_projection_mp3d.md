# P. Domain-invariant ranking projection

Linear projection 1746 -> 128, trained on the 131 training rooms with an in-room InfoNCE over (furnished query, wall-only twin) pairs, selected on the validation rooms. In-room retrieval recall@1m on validation: identity 16.9%, projected 31.3%.

| test room | queries | acoustic alone @1m, identity | projected | median GT rank, identity | projected |
|---|---|---|---|---|---|
| 8WUmhLawc2A_f0 | 151 | 12.6% | 17.9% | 670 | 455 |
| EDJbREhghzL_f0 | 80 | 21.2% | 27.5% | 606 | 334 |
| EDJbREhghzL_f1 | 78 | 2.6% | 7.7% | 1202 | 965 |
| Z6MFQCViBuw_f0 | 160 | 2.5% | 4.4% | 4124 | 3314 |
| gTV8FGcVJC9_f0 | 77 | 7.8% | 11.7% | 1352 | 997 |
| gTV8FGcVJC9_f2 | 65 | 9.2% | 12.3% | 768 | 443 |
| gTV8FGcVJC9_f4 | 142 | 4.2% | 4.2% | 1420 | 1414 |
| gTV8FGcVJC9_f5 | 53 | 17.0% | 18.9% | 660 | 563 |
| pLe4wQe7qrG_f0 | 40 | 30.0% | 32.5% | 80 | 125 |
| q9vSo1VnCiC_f0 | 160 | 6.9% | 14.4% | 706 | 386 |
| sT4fr6TAbpF_f0 | 160 | 15.0% | 21.2% | 564 | 356 |
| uNb9QFRL6hY_f1 | 137 | 3.6% | 1.5% | 2682 | 1262 |
| **all** | 1303 | 9.3% | 12.8% | 1061 | 780 |

Acoustic score alone over the whole candidate grid, furnished query, test rooms never seen in training or selection. If the projected column is not above the identity column here, the feature cannot close the gap by a linear change of basis, whatever it does in-room.

