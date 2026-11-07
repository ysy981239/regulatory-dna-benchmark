# Is the leakage / the giant cluster low-complexity sequence? (human_nontata_promoters)

k = 20, train 27097 rows, test 9034 rows. Entropy is the Shannon entropy of the k-mer's base composition (2 bits = all four bases equally present; <= 1 bit = only A+T or only C+G). In the training tensor 0.9% of windows are A+T only.

## Sharing between test and train

| min entropy (bits) | k-mers kept | test rows sharing >=1 | % of test |
|---|---|---|---|
| 0.0 | 3,915,707 | 5225 | 57.8% |
| 1.2 | 3,806,220 | 4770 | 52.8% |
| 1.5 | 3,361,414 | 4510 | 49.9% |
| 1.7 | 2,584,548 | 4418 | 48.9% |
| 1.9 | 946,812 | 4325 | 47.9% |

## Clusters after dropping low-complexity k-mers

| min entropy (bits) | clusters | largest | largest % |
|---|---|---|---|
| 1.2 | 13705 | 6283 | 23.2% |
| 1.5 | 14344 | 5933 | 21.9% |
| 1.7 | 14499 | 5744 | 21.2% |

Reading: if the sharing collapses once low-complexity k-mers are removed, then most shared 20-mers are repeats and the honest wording is shares a low-complexity 20-mer, not near-duplicate sequence.
