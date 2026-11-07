# Motif / attribution report - `drosophila_enhancers_stark__basset__group__s0`

- dataset: `drosophila_enhancers_stark`, model: `basset`, input length 3237 bp
- explained: top 32 test positives (p 0.943-0.984)
- IG steps: 32; ISM window: 0..256 (256 bp around the IG peak at 0)

## Does the explanation survive checking?

- IG vs ISM top-5% Jaccard: **0.1621**
- mean peak concentration of the IG track: 0.0064 (1.0 = one hot base, 1/L = flat)

- learned consensus: `AAAAAAAA` (0.03 bits per column, 100 windows)
- C+G share of those columns: **0.40** (0.50 = no preference, 1.00 = pure GC)

**Verdict:** IG and ISM disagree - at least one is describing noise; do not claim a motif

### Position weight matrix

```
            A      C      G      T
  pos0   0.32   0.22   0.19   0.27   0.03 bits
  pos1   0.32   0.22   0.19   0.28   0.03 bits
  pos2   0.32   0.21   0.19   0.28   0.03 bits
  pos3   0.32   0.21   0.19   0.28   0.03 bits
  pos4   0.32   0.21   0.19   0.28   0.03 bits
  pos5   0.32   0.21   0.19   0.28   0.03 bits
  pos6   0.32   0.20   0.19   0.28   0.03 bits
  pos7   0.32   0.21   0.19   0.28   0.03 bits
```

### Most attributed 6-mers

| 6-mer | count |
|---|---|
| `AAAAAA` | 20 |
| `TTTTTT` | 20 |
| `TTTAAT` | 17 |
| `TGTGTG` | 16 |
| `AAATAT` | 16 |
| `AAAAAT` | 14 |
| `TATATA` | 14 |
| `AGAAAA` | 14 |
| `TATTTA` | 14 |
| `GAAAAA` | 13 |
| `ATTTAA` | 12 |
| `TTTTTA` | 12 |
| `ATATAT` | 11 |
| `ATAAAA` | 11 |
| `GAAAAT` | 11 |

### In-silico variants that increase the predicted activity

The same table would be the deliverable for a variant-effect project. Effect sizes are in logit space (Δp is rounded to 6 decimals and is near zero simply because the model is saturated).

| sequence | position | change | Δlogit | Δp | context |
|---|---|---|---|---|---|
| 22 | 0 | A>G | +0.3607 | +0.014528 | `[A>G]AAGGCGAGGCAA` |
| 24 | 8 | A>C | +0.3295 | +0.013901 | `CAGCGTCC[A>C]ACAAGCTGAAAC` |
| 13 | 11 | A>C | +0.3221 | +0.011002 | `GCGTCTCCAAC[A>C]CAACACAGGACA` |
| 29 | 5 | A>G | +0.3141 | +0.014331 | `GCGCA[A>G]GACACACACAAA` |
| 23 | 7 | T>G | +0.3092 | +0.012888 | `GGTAATC[T>G]AGCCTCTATTTG` |
| 24 | 5 | T>C | +0.3007 | +0.012846 | `CAGCG[T>C]CCAACAAGCTGA` |
| 23 | 14 | T>G | +0.2908 | +0.012216 | `TAATCTAGCCTC[T>G]ATTTGCCACCCC` |
| 24 | 5 | T>G | +0.2893 | +0.012421 | `CAGCG[T>G]CCAACAAGCTGA` |
| 13 | 16 | A>C | +0.2825 | +0.009820 | `CTCCAACACAAC[A>C]CAGGACATAAAT` |
| 17 | 19 | T>C | +0.2803 | +0.010569 | `GAGGAAGGGGTC[T>C]TGATAATAGTGT` |
