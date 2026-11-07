| run | dataset | model | test AUPRC (real) | test AUPRC (shifted) | retained | corr |
|---|---|---|---|---|---|---|
| drosophila_enhancers_stark__basset__group__s0 | drosophila_enhancers_stark | basset | 0.8255 | 0.6389 | 77.4% | 0.379 |
| drosophila_enhancers_stark__basset__random__s0 | drosophila_enhancers_stark | basset | 0.8472 | 0.6223 | 73.5% | 0.359 |
| human_nontata_promoters__basset-norc__group__s0 | human_nontata_promoters | basset | 0.9359 | 0.9151 | 97.8% | 0.884 |
| human_nontata_promoters__basset__group__s0 | human_nontata_promoters | basset | 0.9275 | 0.9202 | 99.2% | 0.964 |
| human_nontata_promoters__basset__random__s0 | human_nontata_promoters | basset | 0.9638 | 0.9514 | 98.7% | 0.850 |
| human_nontata_promoters__dilated__group__s0 | human_nontata_promoters | dilated | 0.9408 | 0.9380 | 99.7% | 0.980 |
| human_nontata_promoters__transformer__group__s0 | human_nontata_promoters | transformer | 0.8564 | 0.8098 | 94.6% | 0.715 |

6-mer positional divergence on `human_nontata_promoters`: class median 0.3725 vs null median 0.3291; 109 of 3860 k-mers exceed the null p99 (2.8%).
