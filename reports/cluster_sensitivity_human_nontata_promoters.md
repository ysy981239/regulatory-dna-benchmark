# Homology clustering: how much does `k` matter? (human_nontata_promoters)

27097 training rows, width 251. "Family" = shares at least one exact k-mer; components are single-linkage.

| k | clusters | largest | largest % | singletons | direct pairs | closure pairs | amplification | val rows |
|---|---|---|---|---|---|---|---|---|
| 12 | 1 | 27097 | 100.0% | 0 | 1,022,386 | 367,110,156 | 359.1x | 27097 |
| 15 | 722 | 26234 | 96.8% | 664 | 148,350 | 344,098,769 | 2319.5x | 26689 |
| 18 | 8847 | 12945 | 47.8% | 7955 | 56,155 | 83,910,123 | 1494.3x | 4093 |
| 20 | 12254 | 8307 | 30.7% | 11378 | 44,753 | 34,664,745 | 774.6x | 9384 |
| 22 | 13489 | 6574 | 24.3% | 12682 | 39,300 | 21,785,035 | 554.3x | 4065 |
| 25 | 14189 | 5533 | 20.4% | 13479 | 34,263 | 15,498,793 | 452.3x | 4065 |

## What holds the largest component together (k = 20, 8307 rows)

| k-mer | rows in component | % of component | rows in dataset | % of dataset |
|---|---|---|---|---|
| `AAAAAAAAAAAAAAAAAAAA` | 38 | 0.5% | 1 | 0.0% |
| `TTTTTTTTTTTTTTTTTTTT` | 30 | 0.4% | 3 | 0.0% |
| `AAGAAGAAGAAGAAGAAGAA` | 28 | 0.3% | 28 | 0.1% |
| `TCTCTCTCTCTCTCTCTCTC` | 27 | 0.3% | 20 | 0.1% |
| `CTCTCTCTCTCTCTCTCTCT` | 27 | 0.3% | 1 | 0.0% |
| `AGAAGAAGAAGAAGAAGAAG` | 27 | 0.3% | 27 | 0.1% |
| `GAAGAAGAAGAAGAAGAAGA` | 27 | 0.3% | 27 | 0.1% |
| `TTTTTTTTTTTTTTTTTTTT` | 24 | 0.3% | 3 | 0.0% |

Reading: a component is a *family* only if its members are related by descent. If a single k-mer is present in a large share of the component -- especially a low-complexity one -- the threshold is chaining unrelated sequences through a shared repeat, and the grouped split is holding out orphans rather than families.
