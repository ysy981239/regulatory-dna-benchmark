# Is the leakage / the giant cluster low-complexity sequence? (drosophila_enhancers_stark)

k = 20, train 5184 rows, test 1730 rows. Entropy is the Shannon entropy of the k-mer's base composition (2 bits = all four bases equally present; <= 1 bit = only A+T or only C+G). In the training tensor 0.3% of windows are A+T only.

## Sharing between test and train

| min entropy (bits) | k-mers kept | test rows sharing >=1 | % of test |
|---|---|---|---|
| 0.0 | 10,157,356 | 1336 | 77.2% |
| 1.2 | 10,037,437 | 1205 | 69.7% |
| 1.5 | 9,406,476 | 1030 | 59.5% |
| 1.7 | 7,894,610 | 885 | 51.2% |
| 1.9 | 3,459,873 | 700 | 40.5% |

## Clusters after dropping low-complexity k-mers

| min entropy (bits) | clusters | largest | largest % |
|---|---|---|---|
| 1.2 | 1844 | 3054 | 58.9% |
| 1.5 | 2506 | 2099 | 40.5% |
| 1.7 | 3049 | 1371 | 26.5% |

Reading: if the sharing collapses once low-complexity k-mers are removed, then most shared 20-mers are repeats and the honest wording is shares a low-complexity 20-mer, not near-duplicate sequence.
