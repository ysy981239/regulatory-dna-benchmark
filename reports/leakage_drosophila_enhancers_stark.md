# Leakage & separability audit - `drosophila_enhancers_stark`

- sequence length: 3237 bp
- train: 5184 rows, 50.0% positive
- test:  1730 rows, 50.0% positive

## Baselines that use no learned sequence grammar

| baseline | #features | test AUROC | test AUPRC | ECE |
|---|---|---|---|---|
| GC content only | 1 | 0.5087 | 0.4722 | 0.0805 |
| dinucleotide freq (16) | 16 | 0.6873 | 0.6372 | 0.0542 |
| 5-mer freq (1024) | 1024 | 0.7478 | 0.7144 | 0.0992 |

## Near-duplication between train and test

- 77.2% of probed test rows share an exact 20-mer with train (1730 probed).

## Split policy

- 1309 homology clusters over 5184 train rows (mean size 3.96).

| split | shared clusters between train/val | val rows sharing a k-mer with train |
|---|---|---|
| random_row_split | 42 | 75.4% |
| homology_grouped_split | 0 | 0.0% |
