# Motif / attribution report - `human_nontata_promoters__basset__group__s0`

- dataset: `human_nontata_promoters`, model: `basset`, input length 251 bp
- explained: top 64 test positives (p 1.000-1.000)
- IG steps: 32; ISM window: 0..251 (251 bp around the IG peak at 126)

## Does the explanation survive checking?

- IG vs ISM top-5% Jaccard: **0.266**
- mean peak concentration of the IG track: 0.018 (1.0 = one hot base, 1/L = flat)

- learned consensus: `CCCCCCCC` (0.32 bits per column, 100 windows)
- C+G share of those columns: **0.82** (0.50 = no preference, 1.00 = pure GC)

> The attributed windows are C/G-rich but position-independent: this is **base composition**, not a transcription-factor motif. It is the same signal the GC-only linear model already captures, so the network is not evidence of regulatory grammar here.

**Verdict:** IG and ISM disagree - at least one is describing noise; do not claim a motif

### Position weight matrix

```
            A      C      G      T
  pos0   0.09   0.42   0.40   0.09   0.32 bits
  pos1   0.09   0.42   0.40   0.10   0.31 bits
  pos2   0.09   0.41   0.40   0.09   0.31 bits
  pos3   0.09   0.41   0.40   0.09   0.32 bits
  pos4   0.09   0.42   0.40   0.09   0.32 bits
  pos5   0.09   0.41   0.40   0.09   0.31 bits
  pos6   0.09   0.42   0.40   0.09   0.32 bits
  pos7   0.09   0.42   0.40   0.09   0.32 bits
```

### Most attributed 6-mers

| 6-mer | count |
|---|---|
| `GCGGCG` | 156 |
| `GGCGGC` | 144 |
| `CGGCGG` | 130 |
| `CCCGCC` | 125 |
| `GGGCGG` | 119 |
| `CGCCGC` | 116 |
| `CCGCCC` | 115 |
| `GCGCGC` | 113 |
| `GGCGGG` | 111 |
| `GGGGCG` | 105 |
| `CGCCCC` | 101 |
| `GCCGCC` | 100 |
| `CCGCCG` | 98 |
| `GCGGGG` | 97 |
| `CCCCGC` | 96 |

### In-silico variants that increase the predicted activity

The same table would be the deliverable for a variant-effect project. Effect sizes are in logit space (Δp is rounded to 6 decimals and is near zero simply because the model is saturated).

| sequence | position | change | Δlogit | Δp | context |
|---|---|---|---|---|---|
| 34 | 137 | A>G | +1.3580 | +0.000006 | `TGAGTGACGGCC[A>G]TGCGGACCAATG` |
| 34 | 145 | C>T | +1.2509 | +0.000006 | `GGCCATGCGGAC[C>T]AATGGGGCGGGC` |
| 50 | 97 | A>C | +1.2404 | +0.000007 | `CCCCTCCCCCCG[A>C]GCCTCACCTTTT` |
| 33 | 140 | G>T | +1.2397 | +0.000006 | `CGGGCGGCCAAT[G>T]AGGGCGGCCGGA` |
| 40 | 138 | G>C | +1.2274 | +0.000006 | `TCGCTCGCGCCG[G>C]GCCCCGCGGCCG` |
| 31 | 130 | T>G | +1.1885 | +0.000005 | `GCGGCCTCGGGC[T>G]CGCGCGCAAAGG` |
| 43 | 163 | T>C | +1.1456 | +0.000006 | `AAAGTCCCGCGG[T>C]GGCGGCGGCGGC` |
| 34 | 139 | G>T | +1.1446 | +0.000006 | `AGTGACGGCCAT[G>T]CGGACCAATGGG` |
| 47 | 183 | T>C | +1.1319 | +0.000007 | `CGCGCGGCGCGG[T>C]GCACGCTGGGAT` |
| 48 | 159 | T>C | +1.1198 | +0.000007 | `TCCCCTCCCCCC[T>C]GCCCAGAAAGAC` |
