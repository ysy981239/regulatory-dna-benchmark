# Motif / attribution report - `drosophila_enhancers_stark__basset__group__s0`

- dataset: `drosophila_enhancers_stark`, model: `basset`, input length 3237 bp
- explained: top 32 test positives (p 0.958-0.972)
- IG steps: 32; ISM window: 0..256 (256 bp around the IG peak at 22)

## Does the explanation survive checking?

- IG vs ISM top-5% Jaccard: **0.0676**
- mean peak concentration of the IG track: 0.0082 (1.0 = one hot base, 1/L = flat)

- learned consensus: `AAAAAAAA` (0.03 bits per column, 100 windows)
- C+G share of those columns: **0.41** (0.50 = no preference, 1.00 = pure GC)

**Verdict:** IG and ISM disagree - at least one is describing noise; do not claim a motif

### Position weight matrix

```
            A      C      G      T
  pos0   0.32   0.21   0.21   0.27   0.02 bits
  pos1   0.30   0.21   0.20   0.28   0.02 bits
  pos2   0.30   0.21   0.20   0.29   0.03 bits
  pos3   0.30   0.22   0.19   0.29   0.02 bits
  pos4   0.32   0.21   0.20   0.27   0.03 bits
  pos5   0.32   0.20   0.20   0.28   0.03 bits
  pos6   0.31   0.22   0.20   0.28   0.02 bits
  pos7   0.30   0.20   0.20   0.30   0.03 bits
```

### Most attributed 6-mers

| 6-mer | count |
|---|---|
| `AAAAAA` | 31 |
| `TTTTTT` | 17 |
| `ATTTTT` | 16 |
| `AAATAA` | 16 |
| `GAAAAA` | 15 |
| `AAAATA` | 15 |
| `TTATTT` | 14 |
| `TTTTTA` | 14 |
| `CACACA` | 14 |
| `TTTTGC` | 13 |
| `ATAAAT` | 13 |
| `TTTTAT` | 13 |
| `TTTATT` | 13 |
| `ATAATA` | 12 |
| `TATAAA` | 12 |

### In-silico variants that increase the predicted activity

The same table would be the deliverable for a variant-effect project. Effect sizes are in logit space (Δp is rounded to 6 decimals and is near zero simply because the model is saturated).

| sequence | position | change | Δlogit | Δp | context |
|---|---|---|---|---|---|
| 10 | 21 | A>C | +0.2463 | +0.007547 | `GAGATTCAAGAT[A>C]CAAGATTTTAGT` |
| 10 | 2 | A>C | +0.2411 | +0.007406 | `GG[A>C]GGATGTGAGATT` |
| 10 | 10 | A>C | +0.2411 | +0.007406 | `GGAGGATGTG[A>C]GATTCAAGATAC` |
| 10 | 16 | A>G | +0.2407 | +0.007393 | `GATGTGAGATTC[A>G]AGATACAAGATT` |
| 10 | 14 | T>C | +0.2355 | +0.007250 | `AGGATGTGAGAT[T>C]CAAGATACAAGA` |
| 10 | 21 | A>G | +0.2333 | +0.007190 | `GAGATTCAAGAT[A>G]CAAGATTTTAGT` |
| 10 | 24 | A>G | +0.2319 | +0.007152 | `ATTCAAGATACA[A>G]GATTTTAGTTGA` |
| 10 | 17 | A>C | +0.2318 | +0.007150 | `ATGTGAGATTCA[A>C]GATACAAGATTT` |
| 10 | 12 | A>G | +0.2314 | +0.007138 | `GGAGGATGTGAG[A>G]TTCAAGATACAA` |
| 10 | 19 | A>G | +0.2305 | +0.007113 | `GTGAGATTCAAG[A>G]TACAAGATTTTA` |
