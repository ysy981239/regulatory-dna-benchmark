# Motif / attribution report - `human_nontata_promoters__dilated__group__s0`

- dataset: `human_nontata_promoters`, model: `dilated`, input length 251 bp
- explained: top 64 test positives (p 1.000-1.000)
- IG steps: 32; ISM window: 0..251 (251 bp around the IG peak at 154)

## Does the explanation survive checking?

- IG vs ISM top-5% Jaccard: **0.381**
- mean peak concentration of the IG track: 0.0232 (1.0 = one hot base, 1/L = flat)

- learned consensus: `GGGGGGGG` (0.22 bits per column, 100 windows)
- C+G share of those columns: **0.77** (0.50 = no preference, 1.00 = pure GC)

**Verdict:** methods agree but the attributed region is diffuse - no clean motif

### Position weight matrix

```
            A      C      G      T
  pos0   0.11   0.38   0.39   0.12   0.22 bits
  pos1   0.11   0.38   0.39   0.12   0.22 bits
  pos2   0.11   0.38   0.39   0.12   0.22 bits
  pos3   0.11   0.38   0.39   0.12   0.22 bits
  pos4   0.11   0.38   0.40   0.12   0.22 bits
  pos5   0.11   0.38   0.39   0.12   0.23 bits
  pos6   0.11   0.39   0.39   0.12   0.22 bits
  pos7   0.12   0.37   0.39   0.12   0.21 bits
```

### Most attributed 6-mers

| 6-mer | count |
|---|---|
| `GGGCGG` | 93 |
| `CCGCCC` | 74 |
| `CGCCCC` | 74 |
| `CCCCGC` | 73 |
| `GGCGGG` | 73 |
| `GCGGGG` | 72 |
| `CGGGGC` | 69 |
| `CCCGCC` | 68 |
| `GGCGGC` | 67 |
| `GGGGCG` | 66 |
| `CGCCGC` | 66 |
| `GCGGCG` | 63 |
| `GCCGCC` | 61 |
| `CCGCCG` | 59 |
| `GCGCGG` | 55 |

### In-silico variants that increase the predicted activity

The same table would be the deliverable for a variant-effect project. Effect sizes are in logit space (Δp is rounded to 6 decimals and is near zero simply because the model is saturated).

| sequence | position | change | Δlogit | Δp | context |
|---|---|---|---|---|---|
| 57 | 112 | T>G | +10.0188 | +0.000000 | `GCTCGCGGGGGC[T>G]GAGGCGAGAGGA` |
| 10 | 121 | A>C | +9.3086 | +0.000000 | `CGCGCCCGCCTC[A>C]GCGCCCCCACTC` |
| 10 | 183 | A>C | +9.1771 | +0.000000 | `CGCCGCCGCCGC[A>C]GCGCCCTCAGGC` |
| 12 | 87 | T>C | +8.8568 | -0.000000 | `TGCCGCTGCCGC[T>C]GCCGCCGCCGCC` |
| 20 | 156 | A>G | +8.7971 | -0.000000 | `GACTGCGGCGCC[A>G]GCGACGCTCCCG` |
| 12 | 75 | T>C | +8.2648 | -0.000000 | `TGCTGCTGCCGC[T>C]GCCGCTGCCGCT` |
| 45 | 165 | T>C | +8.2608 | -0.000000 | `AAGTCCCGCCCC[T>C]GCCCCGGAGAGG` |
| 10 | 91 | T>C | +8.2586 | -0.000000 | `GCCCTCTCCCGC[T>C]GCCGCCGCCGCC` |
| 13 | 176 | A>C | +8.1919 | -0.000000 | `CGGCGGCGCCGC[A>C]GCGCGCGGCTCC` |
| 60 | 153 | A>G | +8.0644 | -0.000000 | `CGCTCCCGGGCC[A>G]GCGCCTCAGTCG` |
