| dataset | model | params | split | rc aug | seeds n | test AUPRC | between-seed sd | 95% CI (seed 0) | test AUROC | ECE | best F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| human_nontata_promoters | basset | 476,033 | random | yes | 1 | 0.9638 | - | [0.960, 0.967] | 0.9487 | 0.1204 | 0.8902 |
| human_nontata_promoters | dilated | 930,049 | group | yes | 1 | 0.9634 | - | [0.960, 0.967] | 0.9475 | 0.1429 | 0.8904 |
| human_nontata_promoters | basset | 476,033 | group | no | 1 | 0.9436 | - | [0.938, 0.949] | 0.9185 | 0.1036 | 0.8544 |
| human_nontata_promoters | linear-5mer | 1,024 | random | no | 1 | 0.9433 | - | [0.938, 0.948] | 0.9159 | 0.0334 | 0.8553 |
| human_nontata_promoters | linear-5mer | 1,024 | group | no | 1 | 0.9414 | - | [0.935, 0.946] | 0.9142 | 0.0304 | 0.8547 |
| human_nontata_promoters | basset | 476,033 | group | yes | 3 | 0.9324 | 0.0114 | - | 0.9004 | 0.0292 | 0.8374 |
| human_nontata_promoters | linear-2mer | 16 | group | no | 1 | 0.9015 | - | [0.894, 0.908] | 0.8628 | 0.0218 | 0.8041 |
| human_nontata_promoters | linear-2mer | 16 | random | no | 1 | 0.9012 | - | [0.894, 0.908] | 0.8631 | 0.0235 | 0.8045 |
| human_nontata_promoters | transformer | 826,369 | group | yes | 1 | 0.8943 | - | [0.886, 0.901] | 0.8635 | 0.0377 | 0.8126 |
| drosophila_enhancers_stark | basset | 476,033 | random | yes | 1 | 0.8472 | - | [0.826, 0.869] | 0.8760 | 0.1524 | 0.8193 |
| human_nontata_promoters | linear-gc | 1 | group | no | 1 | 0.8394 | - | [0.830, 0.847] | 0.7959 | 0.0480 | 0.7552 |
| human_nontata_promoters | linear-gc | 1 | random | no | 1 | 0.8394 | - | [0.830, 0.847] | 0.7959 | 0.0560 | 0.7560 |
| drosophila_enhancers_stark | basset | 476,033 | group | yes | 3 | 0.8350 | 0.0083 | - | 0.8783 | 0.1453 | 0.8283 |
| drosophila_enhancers_stark | linear-5mer | 1,024 | group | no | 1 | 0.7114 | - | [0.677, 0.740] | 0.7427 | 0.1265 | 0.7299 |
| drosophila_enhancers_stark | linear-2mer | 16 | group | no | 1 | 0.6367 | - | [0.602, 0.673] | 0.6869 | 0.0471 | 0.7061 |
| drosophila_enhancers_stark | linear-gc | 1 | group | no | 1 | 0.4722 | - | [0.443, 0.499] | 0.5087 | 0.0172 | 0.6845 |

Rows with ``seeds n > 1`` are averaged over independent seeds and the sd column is the spread between them.  **Any difference between two rows smaller than their seed spread is not a difference** - see DEVLOG E6.

## Reference baselines (fitted on the full official training split)

These are fitted on the *whole* official train split, i.e. with the benchmark's own leakage included, so they are a ceiling for the baseline rather than a fair comparison.  The fair, same-split versions are the ``linear-*`` rows in the main table.  Either way: a CNN that does not clear the 5-mer linear model is not evidence of regulatory grammar, only of a working optimiser.

| dataset | baseline | #features | test AUROC | test AUPRC |
|---|---|---|---|---|
| drosophila_enhancers_stark | GC content only | 1 | 0.5087 | 0.4722 |
| drosophila_enhancers_stark | dinucleotide freq (16) | 16 | 0.6873 | 0.6372 |
| drosophila_enhancers_stark | 5-mer freq (1024) | 1024 | 0.7478 | 0.7144 |
| human_nontata_promoters | GC content only | 1 | 0.7959 | 0.8394 |
| human_nontata_promoters | dinucleotide freq (16) | 16 | 0.8629 | 0.9011 |
| human_nontata_promoters | 5-mer freq (1024) | 1024 | 0.9178 | 0.9446 |

## Optimism from a random row split

Two numbers move for different reasons.  The **validation** gap is the leakage you would have believed; the **test** gap is only visible if the benchmark's own test split is homology-free.  When the test gap is ~0 while calibration falls apart, the official split is leaking too.

| dataset | model | val AUPRC grouped | val AUPRC random | val gap | test AUPRC grouped | test AUPRC random | test gap | ECE grouped | ECE random |
|---|---|---|---|---|---|---|---|---|---|
| drosophila_enhancers_stark | basset | 0.7718 | 0.8426 | +0.0708 | 0.8350 | 0.8472 | +0.0122 | 0.1453 | 0.1524 |
| human_nontata_promoters | basset | 0.9406 | 0.9663 | +0.0257 | 0.9324 | 0.9638 | +0.0314 | 0.0292 | 0.1204 |
| human_nontata_promoters | linear-5mer | 0.9521 | 0.9422 | -0.0099 | 0.9414 | 0.9433 | +0.0019 | 0.0304 | 0.0334 |
| human_nontata_promoters | linear-2mer | 0.9389 | 0.8979 | -0.0410 | 0.9015 | 0.9012 | -0.0003 | 0.0218 | 0.0235 |
| human_nontata_promoters | linear-gc | 0.9178 | 0.8452 | -0.0726 | 0.8394 | 0.8394 | +0.0000 | 0.0480 | 0.0560 |

## Calibration (temperature fitted on the validation split)

Temperature scaling is a monotone transform, so AUPRC/AUROC **cannot** move.  If NLL improves while ECE does not, the model's mistakes are not uncertain mistakes - they are confident errors, and no global rescaling fixes them.  The oracle column fits T on test labels and is a diagnostic only: a large gap between it and T_val says the two splits differ in difficulty.

| run | T (val) | val ECE | val ECE (T) | test ECE | test ECE (T) | test NLL | test NLL (T) | T oracle (test) |
|---|---|---|---|---|---|---|---|---|
| drosophila_enhancers_stark__basset__group__s0 | 3.208 | 0.1736 | 0.1173 | 0.1908 | 0.1570 | 0.6760 | 0.5125 | 2.904 |
| drosophila_enhancers_stark__basset__random__s0 | 2.011 | 0.1399 | 0.0887 | 0.1524 | 0.0953 | 0.5767 | 0.4714 | 2.098 |
| human_nontata_promoters__basset-norc__group__s0 | 1.722 | 0.1875 | 0.1707 | 0.1036 | 0.1028 | 0.4010 | 0.4038 | 1.268 |
| human_nontata_promoters__basset__group__s0 | 1.370 | 0.1291 | 0.1180 | 0.0218 | 0.0514 | 0.4077 | 0.4154 | 1.032 |
| human_nontata_promoters__basset__random__s0 | 4.958 | 0.1236 | 0.0949 | 0.1204 | 0.0831 | 0.9920 | 0.3253 | 4.968 |
| human_nontata_promoters__dilated__group__s0 | 3.268 | 0.2059 | 0.1332 | 0.1429 | 0.1339 | 0.4678 | 0.3718 | 2.206 |
| human_nontata_promoters__transformer__group__s0 | 1.072 | 0.0858 | 0.0875 | 0.0377 | 0.0375 | 0.4608 | 0.4593 | 1.132 |

## Order or composition? (per-sequence base permutation of the test set)

Every test sequence keeps its exact base counts but loses its order.  A pure composition detector is **invariant** to this, so whatever AUPRC survives was never about sequence order.  The GC-only baseline below is invariant *by construction*: it scores 0.8394 on promoters and 0.4722 on drosophila either way, which is what makes it the yardstick for the shuffled column.

| run | test AUPRC (real) | test AUPRC (shuffled) | drop | retained | corr(p_real, p_shuffled) |
|---|---|---|---|---|---|
| human_nontata_promoters__basset__group__s0 | 0.9193 | 0.8143 | +0.1050 | 88.6% | 0.706 |
| human_nontata_promoters__transformer__group__s0 | 0.8943 | 0.7862 | +0.1081 | 87.9% | 0.652 |
| human_nontata_promoters__basset__group__s2 | 0.9403 | 0.8006 | +0.1397 | 85.1% | 0.626 |
| human_nontata_promoters__basset-norc__group__s0 | 0.9436 | 0.7989 | +0.1447 | 84.7% | 0.628 |
| human_nontata_promoters__basset__group__s1 | 0.9377 | 0.7841 | +0.1536 | 83.6% | 0.544 |
| human_nontata_promoters__dilated__group__s0 | 0.9634 | 0.7858 | +0.1776 | 81.6% | 0.570 |
| human_nontata_promoters__basset__random__s0 | 0.9638 | 0.7264 | +0.2374 | 75.4% | 0.373 |
| drosophila_enhancers_stark__basset__group__s0 | 0.8255 | 0.4976 | +0.3279 | 60.3% | -0.048 |
| drosophila_enhancers_stark__basset__random__s0 | 0.8472 | 0.5101 | +0.3371 | 60.2% | 0.007 |
| drosophila_enhancers_stark__basset__group__s1 | 0.8395 | 0.5045 | +0.3350 | 60.1% | -0.012 |
| drosophila_enhancers_stark__basset__group__s2 | 0.8401 | 0.5047 | +0.3354 | 60.1% | -0.016 |
