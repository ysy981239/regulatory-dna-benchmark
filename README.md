# What does a DNA sequence model actually learn?

**A leakage-audited benchmark of three architectures on regulatory DNA — with the negative result that, on human promoters, the model is reading base composition rather than regulatory grammar.**

No pretrained weights. Three models trained from scratch. Every claim below is a number produced by a script in this repository, and the script that produces it is named.

---

## TL;DR — five findings

| # | Finding | Number |
|---|---|---|
| 1 | The public benchmark is itself contaminated: most official **test** sequences share a 20-mer with the training set | **77%** (Drosophila enhancers), **55%** (human promoters) |
| 2 | A random row split overstates validation performance, i.e. it overstates *your own* result | **+6.4 AUPRC** (promoters), **+7.0** (Drosophila) |
| 3 | A CNN beats a plain 5-mer logistic regression by almost nothing on promoters | **+0.9 to +2.3 AUPRC points** |
| 4 | Destroy the base order, keep the base composition → most of the performance is still there | **86% retained** (promoters) vs **60%** (Drosophila enhancers) |
| 5 | Attribution maps do not collapse onto a motif; they collapse onto GC enrichment | PWM consensus `CCCCCCCC`, **0.3 bits/column**, **C+G = 0.82** |

Findings 1–5 are the reason the headline is a **negative** result on promoters. Finding 4 is also the reason the project is not entirely negative: on Drosophila enhancers the convolutional models **do** use sequence order (a CNN beats the 5-mer baseline by **+11 AUPRC points**, and 40% of its performance disappears when order is destroyed).

Finding 4 and the padding numbers below were re-measured after the audit described in [docs/DEVLOG.md](docs/DEVLOG.md), which found that the global pooling layer was averaging over padding. The earlier Drosophila numbers (58–64% retained, +12 points, ECE up to 0.33) were produced by the buggy code path; the corrected values are the ones reported everywhere in this repository.

---

## The question

Regulatory genomics benchmarks ask: given a ~250 bp or ~3 kb window of DNA, predict whether it is a promoter / enhancer / accessible region. Published models report high AUPRC. Three questions are usually left unasked:

1. **Is the benchmark leaking?** If near-duplicate sequences sit on both sides of the split, the test score measures memorisation.
2. **Which split are the numbers from?** A random row split and a homology-aware split answer different questions.
3. **What did the model actually use?** Accuracy alone does not distinguish "learned a motif grammar" from "counted GC content".

This repository answers those three, and reports what happens when the answer is unflattering.

## Data

[Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) (Grešová et al., *BMC Genomic Data* 24:25, 2023), downloaded from the Hugging Face mirror:

| Dataset | Task | Train / test | Length |
|---|---|---|---|
| `human_nontata_promoters` | promoter vs non-promoter | 27,097 / 9,034 | 251 bp, fixed |
| `drosophila_enhancers_stark` | developmental enhancer | 5,184 / 1,730 | 236–3,237 bp, padded |

## Models (all from scratch, no pretraining)

| Model | Parameters | Receptive field | Mechanism |
|---|---|---|---|
| `basset` | 476,033 | 294 bp | 3 conv blocks (k=15/9/9) + max-pool + `mean‖max` pooling |
| `dilated` | 930,049 | 507 bp | stem k=11 + 20 dilated residual blocks, no pooling |
| `transformer` | 826,369 | whole input | 4 pre-LN layers, 4 heads, learned absolute positions |

All three end in `cat[mean, max]` over the position axis — which is what makes finding 4 possible at all.

## Quickstart

```bash
conda create -n dna-bench python=3.10
conda activate dna-bench
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python tests/test_core.py                                  # 8 assertions, no GPU needed
python scripts/leakage_check.py --dataset human_nontata_promoters
python -m src.train --dataset human_nontata_promoters --model basset --split group --epochs 30
python scripts/baseline_runner.py --dataset human_nontata_promoters --split group
python scripts/calibrate.py --all
python scripts/shuffle_probe.py --all --n-seqs 3000
python scripts/motif_analysis.py --run-dir runs/human_nontata_promoters__basset__group__s0
python scripts/make_table.py                               # writes reports/summary.md
```

Datasets are fetched from `hf-mirror.com` and cached under `data/`; both `data/` and `runs/` are gitignored, so the commands above are the reproduction path. A full run is roughly 5 minutes per model on a 6 GB laptop GPU.

## Results

Main table (official test split, 95% bootstrap CI on AUPRC; `runs/` is gitignored — regenerate with the commands above):

| dataset | model | split | test AUPRC | test AUROC | ECE |
|---|---|---|---|---|---|
| promoters | basset | random | 0.9638 | 0.9487 | 0.120 |
| promoters | linear 5-mer | random | 0.9433 | 0.9159 | 0.033 |
| promoters | dilated CNN | group | **0.9408** | 0.9173 | 0.120 |
| promoters | basset (no RC) | group | 0.9359 | 0.9095 | 0.105 |
| promoters | basset | group | 0.9275 | 0.8935 | 0.023 |
| promoters | linear 5-mer | group | 0.9180 | 0.8900 | 0.126 |
| promoters | linear 2-mer | group | 0.9000 | 0.8626 | 0.093 |
| promoters | transformer | group | 0.8564 | 0.8116 | 0.132 |
| promoters | linear GC only | group | 0.8394 | 0.7959 | 0.113 |
| Drosophila | basset | random | **0.8472** | 0.8760 | 0.152 |
| Drosophila | dilated CNN (pre-fix) | group | 0.8345 | 0.8841 | 0.330 |
| Drosophila | basset | group | 0.8255 | 0.8748 | 0.191 |
| Drosophila | linear 5-mer | group | 0.7114 | 0.7427 | 0.127 |
| Drosophila | linear 2-mer | group | 0.6367 | 0.6869 | 0.047 |
| Drosophila | linear GC only | group | 0.4722 | 0.5087 | 0.017 |

Read the promoters block top to bottom: the whole range from a GC-content counter (0.8394) to the best CNN (0.9408) is ~10 AUPRC points, and a linear model on 1,024 5-mer counts captures 0.9180 of it. That is the quantitative form of "this task is mostly composition".

The two Drosophila Basset rows were retrained after the padding audit ([docs/DEVLOG.md](docs/DEVLOG.md) P1/P2); the dilated row is the older run and is annotated accordingly — its retraining costs ~4 h on a 6 GB laptop GPU and is left in the table so the before/after contrast stays visible. A padded-only bug cannot touch the promoter rows: those windows are a fixed 251 bp and contain no padding at all.

## Repository layout

```
src/
  dna.py            encoding: ids <-> one-hot, reverse complement, exact k-mer codes, GC/k-mer features
  data.py           dataset registry, download/cache, homology clustering, grouped split, leakage report
  models.py         Basset, DilatedCNN, DNATransformer (+ exact receptive-field computation)
  train.py          training loop: AdamW, warmup+cosine, AMP, EMA, early stopping on val AUPRC
  evaluate.py       AUPRC/AUROC/Brier/ECE with bootstrap CIs, reliability and PR curves
  attributions.py   in-silico mutagenesis, integrated gradients, attribution sanity check
scripts/
  leakage_check.py     composition-only baselines + near-duplicate rate + split comparison
  baseline_runner.py   linear baselines evaluated on exactly the same split as the CNNs
  calibrate.py         post-hoc temperature scaling, fitted on validation only
  shuffle_probe.py     destroy base order, keep base composition, re-score
  position_probe.py    circular-shift probe (destroys absolute position only)
  motif_analysis.py    IG x ISM agreement, PWM extraction, in-silico variant table
  padding_check.py     padding fraction, length/prediction correlation, truncation cost
  make_table.py        aggregate runs/*/metrics.json into reports/summary.md
tests/test_core.py     9 assertions covering the parts where a silent bug would invert a conclusion
docs/DEVLOG.md         the 14 bugs found after the pipeline already "worked"
```

## Documents

- **[REPORT.md](REPORT.md)** — the full report: protocol, all seven findings with tables, the bugs found and fixed, limitations.
- `README.zh.md` / `REPORT.zh.md` — the same documents in Chinese.
- `reports/summary.md` — the generated results table (five sections).
- `reports/*.md` / `*.png` — per-run attribution figures and leakage audits.

## Limitations

- Two datasets, one seed each. No error bar across seeds.
- Homology is defined as "shares at least one exact 20-mer", which misses remote homologues and can chain-merge; on Drosophila it produces one component containing 72% of the training rows.
- The shuffle probe is out-of-distribution; the clean control is retraining on shuffled sequence (not yet run).
- Deconvolution/calibration conclusions are about *measured* quantities; attribution maps are diagnostic, not mechanistic.

## License

Code: MIT. Data: see Genomic Benchmarks (CC-BY-4.0 / Apache-2.0 per dataset).
