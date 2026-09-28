# Stage 1 runtime scaling

`|D|`=60, seed 0, 20.0 s limit, synthetic grid. Naive backtracking is the *identical recursion* with every bound removed, so node counts are directly comparable.

| |C| | |D| | naive nodes | B&B nodes | nodes saved | exact k$ | greedy k$ | greedy gap |
|---|---|---|---|---|---|---|---|
| 8 | 44 | 511 | 22 | 23x | 67.26 | 67.26 | 0.0% |
| 12 | 60 | 8,191 | 38 | 216x | 76.61 | 84.72 | 10.6% |
| 16 | 57 | 127,935 | 85 | 1505x | 77.4 | 91.16 | 17.8% |
| 20 | 60 | 1,946,815 | 456 | 4269x | 80.93 | 110.24 | 36.2% |
| 24 | 60 | 30,799,999 | 676 | 45562x | 77.31 | 85.41 | 10.5% |
| 28 | 60 | 46,284,591 (timed out) | 572 | - | 74.14 | 96.24 | 29.8% |

A row marked *(timed out)* is not an optimum: the search was cut off.
