# Does a DNA sequence model learn regulatory grammar, or base composition?

**Technical report.** Code, numbers and figures referenced here are all reproducible from this repository; see [README.md](README.md) for commands.

---

## Abstract

I benchmarked three from-scratch sequence architectures (a Basset-style CNN, a dilated residual CNN, a pre-LN transformer) on two Genomic Benchmarks tasks — human promoters and *Drosophila* developmental enhancers — under both a random row split and a homology-aware split. Seven findings, in decreasing order of how much they should change a reader's behaviour:

1. The benchmark itself is contaminated: **77%** of official test sequences (Drosophila) and **55%** (promoters) share an exact 20-mer with the training set. A homology-aware split cannot fix this, because the test set is fixed.
2. Splitting randomly inflates validation AUPRC by **+6.4 points** (promoters) and **+7.0** (Drosophila) relative to a homology-aware split. This is the size of the lie you tell yourself when you skip the split audit.
3. On promoters, the best CNN beats a **5-mer logistic regression** by **0.9–2.3 AUPRC points**, and a **GC-content-only** logistic regression reaches **0.8394** AUPRC on its own. The task is dominated by composition.
4. Reverse-complement augmentation costs **0.8 AUPRC points** but improves calibration by a factor of four (**ECE 0.105 → 0.023**).
5. Validation-fitted temperature scaling helps badly calibrated models and hurts already-calibrated ones; NLL improved while ECE worsened for two runs, because ECE is not a proper scoring rule.
6. Integrated gradients and in-silico mutagenesis agree only weakly (top-5% Jaccard 0.27–0.38), and the extracted PWM collapses to `CCCCCCCC` at 0.3 bits/column with C+G = 0.82 — base composition, not a transcription-factor motif.
7. Destroying base order while preserving base composition removes only **14%** of promoter AUPRC but **≈40%** of Drosophila enhancer AUPRC. The Drosophila CNN is the one place in this study where sequence order demonstrably matters.

The honest one-line conclusion: **on human promoters this benchmark measures composition, not grammar; on Drosophila enhancers it measures something that genuinely depends on order.**

**Post-hoc audit.** All seven findings were already written up when I asked a simpler question — "what does the padding actually do?" — and found two more bugs: the global pooling averaged over padded positions, and the train and test tensors were not even the same width (236–3,237 bp sequences are right-padded, and the padding length was computed per split). Neither raised an exception. Both were repaired and the Drosophila runs retrained; `docs/DEVLOG.md` lists the 14 issues found after the pipeline "worked", with the before/after numbers. The Drosophila calibration numbers in this report improved enough to change a conclusion (its Brier score is no longer worse than a constant predictor), so the numbers below are the corrected ones and the promoter numbers are untouched — those inputs contain no padding at all.

---

## 1. Protocol

| | |
|---|---|
| Data | Genomic Benchmarks v1 via HF mirror; `human_nontata_promoters` (27,097 / 9,034, 251 bp) and `drosophila_enhancers_stark` (5,184 / 1,730, 236–3,237 bp, right-padded) |
| Splits | `random`: stratified row split (15% validation). `group`: sequences sharing any exact 20-mer are connected, connected components are assigned whole-cluster, so no cluster straddles train/validation |
| Metrics | AUPRC (primary, with 1,000-sample bootstrap CI), AUROC, Brier, ECE (15 bins), NLL, best-F1 threshold |
| Training | AdamW, lr 1e-3, warmup 5% + cosine to zero, weight decay 1e-4, grad clip 1.0, AMP, EMA off, 30 epochs, early stopping disabled so the annealed tail is not truncated |
| Test set | the official benchmark test split, used unchanged; it is the contaminated one, which is the point |

The 20-mer clustering threshold is justified arithmetically: $4^{20} \approx 1.1\times10^{12}$ against a $3.1\times10^9$-position genome makes a chance recurrence of one specific 20-mer unlikely ($\approx 0.003$ per site), while a 3.2 kb window is expected to hit ~9 windows by chance. Single shared k-mers are therefore not evidence, but 77% of sequences sharing one is.

## 2. Finding 1 — the benchmark leaks

| dataset | test rows sharing a 20-mer with train | homology clusters | mean cluster size | max cluster |
|---|---|---|---|---|
| Drosophila enhancers | **77.2%** | 1,309 | 3.96 | 3,723 (72% of train) |
| human promoters | **54.8%** | 12,254 | 2.21 | — |

Two consequences. First, a random split puts ~75% of validation rows in the same family as a training row, so validation measures memorisation. Second, **the official test score cannot be de-leaked** — the fix is to report the number with its contamination attached, not to claim it was removed.

A diagnostic worth keeping: pairwise comparison finds 13,219 directly-sharing pairs, but the 3,723-row component implies 6,928,705 forced-together pairs — a **524× amplification** from transitive closure. Any per-item grouping rule (hashing, greedy assignment) would have missed 99.8% of them.

## 3. Finding 2 — what the random split is worth

| dataset | model | val AUPRC (random) | val AUPRC (group) | val gap | test AUPRC (random) | test AUPRC (group) | test gap |
|---|---|---|---|---|---|---|---|
| promoters | basset | 0.9663 | 0.9020 | **+0.0643** | 0.9638 | 0.9275 | +0.0363 |
| promoters | linear 5-mer | 0.9422 | 0.8652 | **+0.0770** | 0.9433 | 0.9180 | +0.0253 |
| Drosophila | basset | 0.8426 | 0.7725 | **+0.0701** | 0.8472 | 0.8255 | +0.0217 |

Both gaps are positive here, and the test gap is smaller than the validation gap for every model: whatever leakage the random split creates is *partly* neutralised by the fact that the official test split carries the same kind of leakage on both sides. The lesson is not "group splitting reduces scores" — it is that **the gap between the two splits is the size of the measurement error, and the two gaps answer different questions**: the validation gap is what you would have believed, the test gap is what survives contact with a contaminated test set.

## 4. Finding 3 — CNN versus linear baselines

Every baseline below is evaluated on *the same split* as the CNN (same clustering, same seed, same official test set), so the comparison is not confounded by the split choice.

| dataset | model | test AUPRC | test AUROC | ECE | Δ vs 5-mer |
|---|---|---|---|---|---|
| promoters | dilated CNN | 0.9408 | 0.9173 | 0.120 | +0.0228 |
| promoters | basset (no RC) | 0.9359 | 0.9095 | 0.105 | +0.0179 |
| promoters | basset | 0.9275 | 0.8935 | 0.023 | +0.0095 |
| promoters | linear 5-mer | 0.9180 | 0.8900 | 0.126 | — |
| promoters | linear 2-mer | 0.9000 | 0.8626 | 0.093 | −0.0180 |
| promoters | transformer | 0.8564 | 0.8116 | 0.132 | −0.0616 |
| promoters | linear GC only | 0.8394 | 0.7959 | 0.113 | −0.0786 |
| Drosophila | dilated CNN (pre-fix) | 0.8345 | 0.8841 | 0.330 | +0.1231 |
| Drosophila | basset | 0.8255 | 0.8748 | 0.191 | +0.1141 |
| Drosophila | linear 5-mer | 0.7114 | 0.7427 | 0.127 | — |
| Drosophila | linear 2-mer | 0.6367 | 0.6869 | 0.047 | −0.0747 |
| Drosophila | linear GC only | 0.4722 | 0.5087 | 0.017 | −0.2392 |

Two different stories. On Drosophila the CNN is worth **+11 AUPRC points** over a linear model — depth and order genuinely buy something. On promoters the whole headroom above GC content is 10 points and a 1,024-feature linear model takes 8 of them.

The dilated Drosophila row predates the padding fix and is kept only for the before/after contrast; its retraining cost is ~4 h on the 6 GB GPU this was developed on, and the two Basset rows already establish the effect.

The transformer underperforms everything on promoters while being the most expressive model. It is not overfitting — training loss stalls around 0.50 over 30 epochs (best epoch 29 of 30), i.e. it never fits the training set. The reason is the prior budget: the convolution imposes "local, position-invariant, weight-shared" for free, and the transformer must buy it from data with 826k parameters against 23k training sequences and 5,400 updates. The ViT-on-ImageNet-1k-versus-JFT-300M trade-off, at a much smaller scale.

## 5. Finding 4 — reverse-complement augmentation

| promoters, group split | test AUPRC | test AUROC | test ECE |
|---|---|---|---|
| with RC augmentation | 0.9275 | 0.8935 | **0.0228** |
| without | 0.9359 | 0.9095 | 0.1048 |

Augmentation trades **−0.8 AUPRC** for a **4.6× reduction in calibration error**. This is the cleanest demonstration in the study that a single design choice can move accuracy and calibration in opposite directions, and it is the reason both are reported.

## 6. Finding 5 — calibration

Temperature fitted on validation only (never test), applied as $p' = \sigma(z/T)$:

| run | T | val ECE → | test ECE → | test NLL → | oracle T |
|---|---|---|---|---|---|
| Drosophila basset group | 3.21 | 0.174 → 0.117 | 0.191 → 0.157 | 0.676 → 0.513 | 2.90 |
| Drosophila basset random | 2.01 | 0.140 → 0.089 | 0.152 → 0.095 | 0.577 → 0.471 | 2.10 |
| Drosophila dilated group (pre-fix) | 3.26 | 0.250 → 0.174 | 0.330 → 0.229 | 0.949 → 0.583 | 4.59 |
| promoters basset group | 0.81 | 0.183 → 0.165 | 0.023 → 0.021 | 0.389 → 0.392 | 0.97 |
| promoters basset-norc group | 3.70 | 0.278 → 0.291 | 0.105 → **0.132 (worse)** | 0.473 → 0.441 | 2.10 |
| promoters transformer group | 1.91 | 0.297 → 0.291 | 0.132 → 0.114 | 0.572 → 0.559 | 1.52 |

Three observations. (i) The size of the correction tracks how badly calibrated the run was: the worst run (pre-fix dilated, ECE 0.330) gains the most, while the already-calibrated promoter run (0.023) cannot be improved at all — and `basset-norc` actually gets *worse* (0.105 → 0.132). (ii) For two runs NLL improved while ECE worsened — the model's errors are confidently wrong, not uncertain, and ECE rewards the latter. (iii) `T` ranges from 0.8 to 5.3, i.e. most raw scores are far too extreme, which is expected when a CNN is trained to convergence on a small, leaky training set.

The Drosophila models are worth flagging separately, because here the padding audit changed the conclusion. Before the fix, at a 0.5 threshold their recall was ≈1.0 with specificity ≈0.23 and their Brier score (0.27–0.29) was **worse than a constant 0.5 predictor** (0.25) — i.e. unusable as classifiers, and I read that as the model being hopelessly overconfident. After masking the padding out of the pooling, the same architecture reaches Brier **0.175–0.197**, better than the constant baseline, with uncalibrated ECE 0.15–0.19. Roughly two thirds of what temperature scaling used to repair was a padding artefact, not miscalibration.

## 7. Finding 6 — attribution does not survive a sanity check

Method: integrated gradients (32 steps, logit space) versus in-silico mutagenesis (single-base substitution, every position, also logit space — computing ISM in probability space returns zeros once the model saturates at p = 1.000). Agreement is measured as top-5% Jaccard.

| run | IG vs ISM Jaccard | peak concentration (1/L = 0.004) | PWM consensus | bits/column | C+G |
|---|---|---|---|---|---|
| promoters basset group | 0.266 | 0.018 | `CCCCCCCC` | 0.31–0.32 | 0.82 |
| promoters dilated group | 0.381 | 0.023 | `GGGGGGGG` | 0.22 | 0.77 |
| Drosophila basset group | 0.068 | 0.008 | `AAAAAAAA` | 0.03 | 0.41 |

None of the three passes the pre-registered bar (Jaccard ≥ 0.3 *and* mean information ≥ 0.6 bits). The two promoter models agree on the *region* but not on the *bases*, and the extracted pattern is a GC-rich, position-independent smear. The most influential 6-mers in the promoter model are `GCGGCG` (156), `GGCGGC` (144), `CGGCGG` (130) — all GC-rich, none a known core-promoter element.

This is the same conclusion as Finding 3, reached by a completely different route: the model is a composition detector. That convergence is the strongest evidence in the report.

## 8. Finding 7 — order versus composition

Two probes, both non-retraining, both preserving different invariants:

- **shuffle** — permute bases within each sequence: composition preserved exactly, order destroyed entirely.
- **circular shift** — roll each sequence by a random offset: composition and all local motifs preserved, only *absolute position* destroyed.

| model | real AUPRC | shuffled | retained | shifted | retained |
|---|---|---|---|---|---|
| promoters basset (group) | 0.9275 | 0.7955 | **85.8%** | 0.9202 | 99.2% |
| promoters dilated (group) | 0.9408 | 0.8014 | 85.2% | 0.9380 | 99.7% |
| promoters transformer (group) | 0.8564 | 0.8079 | 94.3% | 0.8098 | 94.6% |
| Drosophila basset (group) | 0.8255 | 0.4976 | **60.3%** | 0.6389 | 77.4% |
| Drosophila basset (random) | 0.8472 | 0.5101 | **60.2%** | 0.6223 | 73.5% |

Reading:

- The promoter CNN keeps 86% of its performance with the order destroyed. Whatever it learned, it learned from composition.
- The transformer keeps 94% — but it also *started* at 0.8564, so "loses little" is not a virtue here; it never had much to lose.
- The Drosophila CNN falls to AUPRC ≈0.50, i.e. the prevalence baseline (a coin flip in AUPRC terms), and its real and shuffled predictions correlate at r ≈ 0. It is genuinely reading order.
- The 99% shift-retention of the promoter models is architectural: `cat[mean, max]` pooling is translation-invariant by construction, and those windows contain no padding. The transformer is the only promoter model that loses accuracy under a circular shift (94.6%, AUROC −6.4 points), i.e. the only one using absolute position — and it is still the worst model on this data.
- The Drosophila shift-retention of 73–77% does **not** license the same architectural argument, and I do not claim it does. Rolling a padded, left-aligned sequence preserves the mask but creates one artificial junction between two sequence ends; the residual drop therefore conflates "absolute position matters" with "the model reacts to one novel junction". The natural control is a model whose order sensitivity is known (the promoter CNNs are indifferent, the enhancers are not), and that is the pattern observed. Treat it as an open question rather than evidence.

A data-level check explains why the promoter side looks the way it does: for 6-mers, only **2.8%** (109 of 3,860) have positional distributions that differ between positives and negatives beyond a null built from random class halves. The positional signal that benchmark could teach is small to begin with.

Padding itself was measured rather than assumed (`scripts/padding_check.py`): 34.4% of the Drosophila test tensor is padding, and truncating every sequence to its first 500 bases costs **12 AUPRC points** (0.8255 → 0.7060), so the models are not reading only the start of the window. The correlation between real sequence length and the prediction is ≈0 (Spearman −0.006 / +0.013), which is the check that the masked pooling actually removed the length dependence.

## 9. Bugs found and fixed

Every one of these would have silently inverted a conclusion, which is why they are listed:

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | k-mer sentinel used the row index inside the batch | every padded row collided with row 0 → "99.9% of test shares a 20-mer with train", contradicting a 1,309-cluster result | content-derived 64-bit hash (`blake2b`), batch-independent |
| 2 | LR schedule defined per *step*, stepped per *epoch* | training never left warmup: lr reached 29% of the intended peak, cosine never ran | define the schedule in epochs |
| 3 | `LambdaLR` calls `lr_lambda(0)` at construction | epoch 1 trained at lr = 0 | `step = max(step, 1)` |
| 4 | early stopping (patience 8) truncated the cosine tail | runs stopped at epoch 19 of 30, before annealing | patience defaults to 0 |
| 5 | run directory named only by dataset/model/split/seed | `--no-rc-augment` overwrote the RC run | suffix the tag with `-norc`, `-w{width}` |
| 6 | transformer head was `Linear(d_model, 1)` while the pooling concatenates mean and max | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x256 and 128x1)` | `Linear(2 * d_model, 1)` |
| 7 | AUROC of a constant predictor | sklearn returns 0.5, which reads as "no better than chance" rather than "undefined" | return NaN when labels or scores have one unique value |
| 8 | global pooling averaged over padded positions | a 236 bp sequence inside a 3237 bp tensor drew 92.6% of its mean from a constant; no exception, no warning | `masked_pool` with `valid_mask` / `down_mask`, and `src_key_padding_mask` for the transformer |
| 9 | tensor width computed per split | train 3237 vs test 3061 — the same weights evaluated as a different function, silently | pin `max_len` in the registry and share it across splits |
| 10 | best-F1 threshold selected on the test predictions | the reported F1 is the optimistic one, chosen after seeing test labels | `best_f1_threshold()` on validation, applied to test |

Bugs 8–10 were found *after* the pipeline worked, by asking what the padding actually did and by checking whether any reported number was selected using the data it was reported on. Their full write-up, with before/after numbers and the three further issues found in the same audit, is in [`docs/DEVLOG.md`](docs/DEVLOG.md). The pattern worth noting: none of them raised an exception, and each surfaced only when two independent measurements were compared — for 8 and 9 it was "same weights, different code, different score".

## 10. Limitations

- **Two datasets, one seed.** No seed-to-seed error bars; the 95% CIs are bootstrap over the test set, not over training runs.
- **The headline numbers needed a second pass.** Two padding-handling bugs (section 9, #8–#9) were found after the first version of these results, and the Drosophila rows were retrained; the `dilated` Drosophila row is still the pre-fix model, because retraining it costs ~4 h on a 6 GB laptop GPU. Promoter numbers were unaffected by construction (their windows contain no padding) — which is itself the reason the bug survived so long.
- **20-mer homology is a proxy.** It misses remote homologues (so "0% sharing" is a lower bound) and single-linkage can chain-merge: on Drosophila one component holds 72% of the training rows, which means the effective number of independent training families is far below 5,184.
- **Probes are out-of-distribution.** Shuffle and shift create sequences no real genome produces; the clean control is retraining on shuffled data, which I have not run.
- **Attribution is diagnostic, not mechanistic.** The failure of IG and ISM to agree means *neither* should be used to claim a motif — not that a different attribution method would succeed.
- **One task family.** Promoters and enhancers at 251 bp and 3.2 kb; nothing here speaks to 100 kb-scale regulation (Enformer/Sei regime), where attention and long-range priors are known to matter.

## 11. What I would do next

1. Retrain basset on shuffled sequences as the positive control for Finding 7.
2. Three seeds per configuration; report the seed variance next to the bootstrap CI.
3. Sweep the homology threshold (k = 15/20/30) to show how much of Finding 1 is threshold-dependent.
4. Add `human_ocr_ensembl` (139,804 / 34,952) as a third task with a much larger sample size.
5. Replace the from-scratch transformer with a fine-tuned DNA language model (HyenaDNA / Nucleotide Transformer) — the current result is about *from-scratch* training, not about transformers.
