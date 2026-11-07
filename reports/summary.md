| dataset | model | params | split | rc aug | test AUPRC | 95% CI | test AUROC | ECE | best F1 |
|---|---|---|---|---|---|---|---|---|---|
| human_nontata_promoters | basset | 476,033 | random | yes | 0.9638 | [0.960, 0.967] | 0.9487 | 0.1204 | 0.8902 |
| human_nontata_promoters | linear-5mer | 1,024 | random | no | 0.9433 | [0.938, 0.948] | 0.9159 | 0.0334 | 0.8553 |
| human_nontata_promoters | dilated | 930,049 | group | yes | 0.9408 | [0.936, 0.945] | 0.9173 | 0.1199 | 0.8452 |
| human_nontata_promoters | basset | 476,033 | group | no | 0.9359 | [0.930, 0.940] | 0.9095 | 0.1048 | 0.8370 |
| human_nontata_promoters | basset | 476,033 | group | yes | 0.9275 | [0.922, 0.932] | 0.8935 | 0.0228 | 0.8277 |
| human_nontata_promoters | linear-5mer | 1,024 | group | no | 0.9180 | [0.911, 0.923] | 0.8900 | 0.1258 | 0.8268 |
| human_nontata_promoters | linear-2mer | 16 | random | no | 0.9012 | [0.894, 0.908] | 0.8631 | 0.0235 | 0.8045 |
| human_nontata_promoters | linear-2mer | 16 | group | no | 0.9000 | [0.892, 0.906] | 0.8626 | 0.0930 | 0.8018 |
| human_nontata_promoters | transformer | 826,369 | group | yes | 0.8564 | [0.848, 0.864] | 0.8116 | 0.1322 | 0.7626 |
| drosophila_enhancers_stark | basset | 476,033 | random | yes | 0.8472 | [0.826, 0.869] | 0.8760 | 0.1524 | 0.8193 |
| human_nontata_promoters | linear-gc | 1 | group | no | 0.8394 | [0.830, 0.847] | 0.7959 | 0.1125 | 0.7560 |
| human_nontata_promoters | linear-gc | 1 | random | no | 0.8394 | [0.830, 0.847] | 0.7959 | 0.0560 | 0.7560 |
| drosophila_enhancers_stark | basset | 476,033 | group | yes | 0.8255 | [0.794, 0.852] | 0.8748 | 0.1908 | 0.8248 |
| drosophila_enhancers_stark | linear-5mer | 1,024 | group | no | 0.7114 | [0.677, 0.740] | 0.7427 | 0.1265 | 0.7299 |
| drosophila_enhancers_stark | linear-2mer | 16 | group | no | 0.6367 | [0.602, 0.673] | 0.6869 | 0.0471 | 0.7067 |
| drosophila_enhancers_stark | linear-gc | 1 | group | no | 0.4722 | [0.443, 0.499] | 0.5087 | 0.0172 | 0.6851 |

## Reference baselines (no learned sequence grammar, official test split)

A CNN that does not clear the 5-mer linear model is not evidence of regulatory grammar, only of a working optimiser.

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
| human_nontata_promoters | linear-5mer | 0.8652 | 0.9422 | +0.0770 | 0.9180 | 0.9433 | +0.0253 | 0.1258 | 0.0334 |
| drosophila_enhancers_stark | basset | 0.7725 | 0.8426 | +0.0701 | 0.8255 | 0.8472 | +0.0217 | 0.1908 | 0.1524 |
| human_nontata_promoters | basset | 0.9020 | 0.9663 | +0.0643 | 0.9275 | 0.9638 | +0.0363 | 0.0228 | 0.1204 |
| human_nontata_promoters | linear-2mer | 0.8840 | 0.8979 | +0.0139 | 0.9000 | 0.9012 | +0.0012 | 0.0930 | 0.0235 |
| human_nontata_promoters | linear-gc | 0.8425 | 0.8452 | +0.0027 | 0.8394 | 0.8394 | +0.0000 | 0.1125 | 0.0560 |

## Calibration (temperature fitted on the validation split)

Temperature scaling is a monotone transform, so AUPRC/AUROC **cannot** move.  If NLL improves while ECE does not, the model's mistakes are not uncertain mistakes - they are confident errors, and no global rescaling fixes them.  The oracle column fits T on test labels and is a diagnostic only: a large gap between it and T_val says the two splits differ in difficulty.

| run | T (val) | val ECE | val ECE (T) | test ECE | test ECE (T) | test NLL | test NLL (T) | T oracle (test) |
|---|---|---|---|---|---|---|---|---|
| drosophila_enhancers_stark__basset__group__s0 | 3.208 | 0.1736 | 0.1173 | 0.1908 | 0.1570 | 0.6760 | 0.5125 | 2.904 |
| drosophila_enhancers_stark__basset__random__s0 | 2.011 | 0.1399 | 0.0887 | 0.1524 | 0.0953 | 0.5767 | 0.4714 | 2.098 |
| human_nontata_promoters__basset-norc__group__s0 | 3.701 | 0.2775 | 0.2911 | 0.1048 | 0.1324 | 0.4732 | 0.4406 | 2.095 |
| human_nontata_promoters__basset__group__s0 | 0.811 | 0.1828 | 0.1646 | 0.0228 | 0.0208 | 0.3887 | 0.3916 | 0.965 |
| human_nontata_promoters__basset__random__s0 | 4.958 | 0.1236 | 0.0949 | 0.1204 | 0.0831 | 0.9920 | 0.3253 | 4.968 |
| human_nontata_promoters__dilated__group__s0 | 5.338 | 0.3397 | 0.3146 | 0.1199 | 0.1579 | 0.4749 | 0.4579 | 2.230 |
| human_nontata_promoters__transformer__group__s0 | 1.908 | 0.2967 | 0.2909 | 0.1322 | 0.1139 | 0.5723 | 0.5585 | 1.519 |

## Order or composition? (per-sequence base permutation of the test set)

Every test sequence keeps its exact base counts but loses its order.  A pure composition detector is **invariant** to this, so whatever AUPRC survives was never about sequence order.  The GC-only baseline below is invariant *by construction*: it scores 0.8394 on promoters and 0.4722 on drosophila either way, which is what makes it the yardstick for the shuffled column.

| run | test AUPRC (real) | test AUPRC (shuffled) | drop | retained | corr(p_real, p_shuffled) |
|---|---|---|---|---|---|
| human_nontata_promoters__transformer__group__s0 | 0.8564 | 0.8079 | +0.0485 | 94.3% | 0.715 |
| human_nontata_promoters__basset-norc__group__s0 | 0.9359 | 0.8151 | +0.1208 | 87.1% | 0.598 |
| human_nontata_promoters__basset__group__s0 | 0.9275 | 0.7955 | +0.1320 | 85.8% | 0.565 |
| human_nontata_promoters__dilated__group__s0 | 0.9408 | 0.8014 | +0.1394 | 85.2% | 0.386 |
| human_nontata_promoters__basset__random__s0 | 0.9638 | 0.7264 | +0.2374 | 75.4% | 0.373 |
| drosophila_enhancers_stark__basset__group__s0 | 0.8255 | 0.4976 | +0.3279 | 60.3% | -0.048 |
| drosophila_enhancers_stark__basset__random__s0 | 0.8472 | 0.5101 | +0.3371 | 60.2% | 0.007 |
