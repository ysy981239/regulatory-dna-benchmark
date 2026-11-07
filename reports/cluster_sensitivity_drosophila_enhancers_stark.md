# Homology clustering: how much does `k` matter? (drosophila_enhancers_stark)

5184 training rows, width 3237. "Family" = shares at least one exact k-mer; components are single-linkage.

| k | clusters | largest | largest % | singletons | direct pairs | closure pairs | amplification | val rows |
|---|---|---|---|---|---|---|---|---|
| 12 | 1 | 5184 | 100.0% | 0 | 2,690,020 | 13,434,336 | 5.0x | 5184 |
| 15 | 1 | 5184 | 100.0% | 0 | 223,700 | 13,434,336 | 60.1x | 5184 |
| 18 | 268 | 4905 | 94.6% | 257 | 34,396 | 12,027,074 | 349.7x | 4930 |
| 20 | 1309 | 3723 | 71.8% | 1189 | 18,401 | 6,928,705 | 376.5x | 778 |
| 22 | 2246 | 2608 | 50.3% | 1986 | 12,430 | 3,399,965 | 273.5x | 778 |
| 25 | 3158 | 1430 | 27.6% | 2727 | 8,927 | 1,022,593 | 114.6x | 778 |
| 30 | 3793 | 511 | 9.9% | 3220 | 7,158 | 132,682 | 18.5x | 779 |

## What holds the largest component together (k = 20, 3723 rows)

| k-mer | rows in component | % of component | rows in dataset | % of dataset |
|---|---|---|---|---|
| `ATTTATATATTTATATATTT` | 71 | 1.9% | 71 | 1.4% |
| `ATATTTATATATTTATATAT` | 71 | 1.9% | 71 | 1.4% |
| `TATATTTATATATTTATATA` | 71 | 1.9% | 71 | 1.4% |
| `ATATATTTATATATTTATAT` | 71 | 1.9% | 71 | 1.4% |
| `TATATATTTATATATTTATA` | 71 | 1.9% | 71 | 1.4% |
| `TTATATATTTATATATTTAT` | 71 | 1.9% | 71 | 1.4% |
| `TATTTATATATTTATATATT` | 71 | 1.9% | 71 | 1.4% |
| `ATAATAATAATAATAATAAT` | 71 | 1.9% | 1 | 0.0% |

Reading: a component is a *family* only if its members are related by descent. If a single k-mer is present in a large share of the component -- especially a low-complexity one -- the threshold is chaining unrelated sequences through a shared repeat, and the grouped split is holding out orphans rather than families.
