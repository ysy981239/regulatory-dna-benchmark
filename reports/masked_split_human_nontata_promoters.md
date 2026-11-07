# Family-aware split with low-complexity k-mers excluded (human_nontata_promoters)

k = 20, train 27097 rows, test 9034 rows. A k-mer below the entropy threshold may not link two sequences.

| min entropy | clusters | largest | largest % | val rows | val clusters | test sharing | val sharing |
|---|---|---|---|---|---|---|---|
| 0.0 | 12254 | 8307 | 30.7% | 4065 | 2930 | 57.8% | 0 |
| 1.2 | 13705 | 6283 | 23.2% | 4065 | 3027 | 52.8% | 0 |
| 1.5 | 14344 | 5933 | 21.9% | 4065 | 2815 | 49.9% | 0 |
| 1.7 | 14499 | 5744 | 21.2% | 4065 | 2923 | 48.9% | 0 |

Reading: `val rows` is the statistical power the split actually has; `largest %` is how much of the training set is still one blob; `test sharing` is the part of the benchmark no training-side clustering can repair.
