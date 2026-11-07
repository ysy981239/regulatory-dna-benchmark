# Leakage & separability audit - `human_nontata_promoters`

- sequence length: 251 bp
- train: 27097 rows, 54.4% positive
- test:  9034 rows, 54.4% positive

## Baselines that use no learned sequence grammar

| baseline | #features | test AUROC | test AUPRC | ECE |
|---|---|---|---|---|
| GC content only | 1 | 0.7959 | 0.8394 | 0.0558 |
| dinucleotide freq (16) | 16 | 0.8629 | 0.9011 | 0.0217 |
| 5-mer freq (1024) | 1024 | 0.9178 | 0.9446 | 0.0313 |

## Near-duplication between train and test

- 54.8% of probed test rows share an exact 20-mer with train (2000 probed).

## Split policy

- 12254 homology clusters over 27097 train rows (mean size 2.21).

| split | shared clusters between train/val | val rows sharing a k-mer with train |
|---|---|---|
| random_row_split | 434 | 55.7% |
| homology_grouped_split | 0 | 0.0% |
