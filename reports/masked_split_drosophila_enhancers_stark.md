# Family-aware split with low-complexity k-mers excluded (drosophila_enhancers_stark)

k = 20, train 5184 rows, test 1730 rows. A k-mer below the entropy threshold may not link two sequences.

| min entropy | clusters | largest | largest % | val rows | val clusters | test sharing | val sharing |
|---|---|---|---|---|---|---|---|
| 0.0 | 1309 | 3723 | 71.8% | 778 | 703 | 77.2% | 0 |
| 1.2 | 1844 | 3054 | 58.9% | 778 | 691 | 69.7% | 0 |
| 1.5 | 2506 | 2099 | 40.5% | 778 | 641 | 59.5% | 0 |
| 1.7 | 3049 | 1371 | 26.5% | 778 | 631 | 51.2% | 0 |

Reading: `val rows` is the statistical power the split actually has; `largest %` is how much of the training set is still one blob; `test sharing` is the part of the benchmark no training-side clustering can repair.
