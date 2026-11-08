# Does a DNA sequence model learn regulatory grammar, or base composition?

**Technical report.** Code, numbers and figures referenced here are all reproducible from this repository; see [README.md](README.md) for commands.

---

## Abstract

I benchmarked three from-scratch sequence architectures (a Basset-style CNN, a dilated residual CNN, a pre-LN transformer) on two Genomic Benchmarks tasks — human promoters and *Drosophila* developmental enhancers — under both a random row split and a homology-aware split. Eight findings, in decreasing order of how much they should change a reader's behaviour:

1. The benchmark itself is contaminated: **77%** of official test sequences (Drosophila) and **55%** (promoters) share an exact 20-mer with the training set. A homology-aware split cannot fix this, because the test set is fixed.
2. Splitting randomly inflates **validation** AUPRC by **+7.0 points** (Drosophila) and **+1.4** (promoters) relative to a homology-aware split, but inflates the **test** score by only **+2.2** and **+4.5**. The two gaps answer different questions — what you would have believed, versus what survives contact with a contaminated test set — and neither is consistently the larger one.
3. On promoters a CNN does **not** beat a **5-mer logistic regression**: across three seeds the Basset CNN averages **0.9324 ± 0.0114** against **0.9414** for the linear model, while a **GC-content-only** regression reaches **0.8394** on its own. The task is dominated by composition, and 1,024 composition features take essentially all of the headroom.
4. Reverse-complement augmentation costs **2.4 AUPRC points** but improves calibration by a factor of 4.7 (**ECE 0.104 → 0.022**).
5. Validation-fitted temperature scaling repairs the badly calibrated runs (Drosophila ECE 0.191 → 0.157, NLL 0.676 → 0.513) and does nothing or harm to the already-calibrated ones (promoters, T ≈ 1.1–1.4, ECE 0.022 → 0.051); for one run NLL improved while ECE worsened, because ECE is not a proper scoring rule.
6. Integrated gradients and in-silico mutagenesis agree only weakly (top-5% Jaccard 0.07–0.38), and the extracted PWM collapses to `CCCCCCCC` at 0.3 bits/column with C+G = 0.82 — base composition, not a transcription-factor motif. (Measured on the pre-audit weights.)
7. Destroying base order while preserving base composition removes only **11–19%** of promoter AUPRC (82–89% retained) but **40%** of Drosophila enhancer AUPRC (60% retained). The Drosophila CNN is the one place in this study where sequence order demonstrably matters.
8. Between-seed spread on promoters is **1.1 AUPRC points** — the same size as the differences earlier versions of this report were ranking models by. Three seeds, two splits and four probes later, the only comparison that survives is the Drosophila CNN-versus-linear gap (+12.4 points, ≈15 seed standard deviations).

The honest one-line conclusion: **on human promoters this benchmark measures composition, not grammar; on Drosophila enhancers it measures something that genuinely depends on order.**

**Post-hoc audit.** All seven findings were already written up when I asked a simpler question — "what does the padding actually do?" — and found two more bugs: the global pooling averaged over padded positions, and the train and test tensors were not even the same width (236–3,237 bp sequences are right-padded, and the padding length was computed per split). Neither raised an exception. The audit then turned on the audit itself: the homology-aware splitter was putting 34.6% of the promoter rows into validation where 15% was asked for, and the size of that set was being decided by a shuffle (E5). Both were repaired, every affected run was retrained, and three seeds were added per configuration — which is what produced finding 8 (E6). `docs/DEVLOG.md` lists the **17** issues found after the pipeline "worked", with the before/after numbers for each.

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

The threshold behind both numbers — "same family" = shares at least one exact 20-mer — was never validated, so it was swept (`scripts/cluster_sensitivity.py`) and then stress-tested against sequence complexity (`scripts/complexity_probe.py`). Two things came out.

**The split only exists for k ≥ 20.** At k = 18 the largest component already swallows 95% of the Drosophila training set, so validation would have to take 4,930 of 5,184 rows; at k = 15 everything collapses into one component and the split has nothing to hold out. k = 20 is not a principled constant — it is the smallest value at which the protocol still functions, which is a fragility, not a justification.

**The largest component is chained by low-complexity repeats, not by homology.** The most widespread 20-mers inside the 3,723-row component are AT-only microsatellites such as `ATTTATATATTTATATATTT` and `ATAATAATAATAATAATAAT` (71 rows each, 1.9% of the component), and in the promoter component they are homopolymers (`AAAAAAAAAAAAAAAAAAAA`, 38 rows). Dropping k-mers whose base-composition entropy is below 1.5 bits (a 20-mer over only A and T has entropy ≤ 1 bit) leaves the sharing at **59.5%** on Drosophila and **49.9%** on promoters, and shrinks the largest component from 72% to 40%.

The two datasets respond differently, which is itself informative: promoter sharing barely moves (57.8% → 47.9% across the whole entropy range) because CpG islands really are similar to each other, while enhancer sharing drops by 37 points because much of it was AT-repeat noise. **77% (enhancer) and 55% (promoter) of test rows share an exact 20-mer with train**; after removing low-complexity windows the enhancer figure falls to 51% and the promoter figure to 48%, which is a lower bound (masking can only remove evidence of sharing, never add it). The residual is genuine sequence similarity rather than junk — but note that the number moves with the threshold, so the unthresholded 77%/55% is the only version worth quoting without a caveat.

## 3. Finding 2 — what the random split is worth

| dataset | model | val AUPRC (random) | val AUPRC (group) | val gap | test AUPRC (random) | test AUPRC (group) | test gap |
|---|---|---|---|---|---|---|---|
| promoters | basset | 0.9663 | 0.9521 | +0.0142 | 0.9638 | 0.9193 | **+0.0445** |
| promoters | linear 5-mer | 0.9422 | 0.9521 | −0.0099 | 0.9433 | 0.9414 | +0.0019 |
| promoters | linear GC only | 0.8452 | 0.9178 | −0.0726 | 0.8394 | 0.8394 | 0.0000 |
| Drosophila | basset | 0.8426 | 0.7725 | **+0.0701** | 0.8472 | 0.8255 | +0.0217 |

The two gaps do not have to agree, and after the splitter fix on promoters they point in opposite directions by model: the CNN shows almost no validation gap (+1.4) but a 4.5-point test gap, while the linear models show *negative* validation gaps (the grouped validation set is drawn from small families and is easier for them) and no test gap at all. The lesson is not "group splitting reduces scores" — it is that **each gap measures something different**: the validation gap is what you would have believed, the test gap is what survives contact with a contaminated test set, and a validation set drawn from a different family mix can make the first one smaller than the truth.

## 4. Finding 3 — CNN versus linear baselines

Every baseline below is evaluated on *the same split* as the CNN (same clustering, same seed, same official test set), so the comparison is not confounded by the split choice.

| dataset | model | test AUPRC | test AUROC | ECE | Δ vs 5-mer |
|---|---|---|---|---|---|
| promoters | dilated CNN | 0.9634 | 0.9475 | 0.143 | +0.0220 |
| promoters | basset (no RC) | 0.9436 | 0.9185 | 0.104 | +0.0022 |
| promoters | linear 5-mer | 0.9414 | 0.9142 | 0.030 | — |
| promoters | basset (3 seeds) | **0.9324 ± 0.0114** | 0.9004 | 0.029 | −0.0090 |
| promoters | linear 2-mer | 0.9015 | 0.8628 | 0.022 | −0.0399 |
| promoters | transformer | 0.8943 | 0.8635 | 0.038 | −0.0471 |
| promoters | linear GC only | 0.8394 | 0.7959 | 0.048 | −0.1020 |
| Drosophila | basset (3 seeds) | **0.8350 ± 0.0083** | 0.8783 | 0.145 | +0.1236 |
| Drosophila | dilated CNN (pre-fix) | 0.8345 | 0.8841 | 0.330 | +0.1231 |
| Drosophila | linear 5-mer | 0.7114 | 0.7427 | 0.127 | — |
| Drosophila | linear 2-mer | 0.6367 | 0.6869 | 0.047 | −0.0747 |
| Drosophila | linear GC only | 0.4722 | 0.5087 | 0.017 | −0.2392 |

Two different stories. On Drosophila the CNN is worth **+12.4 AUPRC points** over a linear model — depth and order genuinely buy something, and the effect is about fifteen seed standard deviations. On promoters nothing clearly beats the 5-mer linear model: the Basset CNN sits **1 point below it** once the noise is measured, and only the dilated CNN is ahead (+2.2 points, on a single seed). A 1,024-feature composition model takes essentially all of the headroom above GC content.

The dilated Drosophila row predates the padding fix and is kept only for the before/after contrast; its retraining cost is ~4 h on the 6 GB GPU this was developed on, and the three Basset seeds already establish the effect.

The transformer is the weakest deep model on promoters, but it is no longer the pathological case it was before the split fix: it now runs all 30 epochs and its validation AUPRC is still climbing at the last one (0.9457), i.e. it was under-trained rather than unable to fit. The residual argument stands — the convolution imposes "local, position-invariant, weight-shared" for free, and the transformer has to buy it from 826k parameters against 23k sequences — but it is now a statement about the prior budget rather than about a model that never converged.

## 5. Finding 4 — reverse-complement augmentation

| promoters, group split | test AUPRC | test AUROC | test ECE |
|---|---|---|---|
| with RC augmentation | 0.9193 | 0.8817 | **0.0218** |
| without | 0.9436 | 0.9185 | 0.1036 |

Augmentation trades **−2.4 AUPRC** for a **4.7× reduction in calibration error**. This is the cleanest demonstration in the study that a single design choice can move accuracy and calibration in opposite directions, and it is the reason both are reported. (The accuracy cost doubled when the splitter was fixed — the same seed now trains on 30% more rows, and the augmented model was the one that benefited least.)

## 6. Finding 5 — calibration

Temperature fitted on validation only (never test), applied as $p' = \sigma(z/T)$:

| run | T | val ECE → | test ECE → | test NLL → | oracle T |
|---|---|---|---|---|---|
| Drosophila basset group | 3.21 | 0.174 → 0.117 | 0.191 → 0.157 | 0.676 → 0.513 | 2.90 |
| Drosophila basset random | 2.01 | 0.140 → 0.089 | 0.152 → 0.095 | 0.577 → 0.471 | 2.10 |
| Drosophila dilated group (pre-fix) | 3.26 | 0.250 → 0.174 | 0.330 → 0.229 | 0.949 → 0.583 | 4.59 |
| promoters basset group | 1.37 | 0.129 → 0.118 | 0.022 → **0.051 (worse)** | 0.408 → 0.415 | 1.03 |
| promoters basset-norc group | 1.72 | 0.188 → 0.171 | 0.104 → 0.103 | 0.401 → 0.404 | 1.27 |
| promoters dilated group | 3.27 | 0.206 → 0.133 | 0.143 → 0.134 | 0.468 → 0.372 | 2.21 |
| promoters transformer group | 1.07 | 0.086 → 0.088 | 0.038 → 0.038 | 0.461 → 0.459 | 1.13 |
| promoters basset random | 4.96 | 0.124 → 0.095 | 0.120 → 0.083 | 0.992 → 0.325 | 4.97 |

Three observations. (i) The size of the correction tracks how badly calibrated the run was: the worst runs gain the most (pre-fix dilated, ECE 0.330 → 0.229; promoters dilated, 0.143 → 0.134), while the already-calibrated promoter Basset run cannot be improved and in fact **gets worse** (0.022 → 0.051). (ii) That failure has a concrete cause worth naming: `T` is fitted on a validation split whose positive rate is **73.5%** against 54.4% in test, so the fitted temperature believes the model is under-confident when it is not. Thresholds and temperatures inherited from a differently-composed validation set are a systematic error, not noise. (iii) `T` ranges from 1.07 to 4.96, i.e. most raw scores are too extreme — expected when a CNN is trained to convergence on a small, leaky training set.

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
| promoters dilated (group) | 0.9634 | 0.7858 | **81.6%** | 0.9578 | 99.4% |
| promoters basset (group, 3 seeds) | 0.9193 / 0.9377 / 0.9403 | 0.8143 / 0.7841 / 0.8006 | **88.6 / 83.6 / 85.1%** | 0.9141 / 0.9290 / 0.9313 | 99.4 / 99.1 / 99.0% |
| promoters basset, no RC (group) | 0.9436 | 0.7989 | 84.7% | 0.9203 | 97.5% |
| promoters transformer (group) | 0.8943 | 0.7862 | 87.9% | 0.7980 | 89.2% |
| promoters basset (random) | 0.9638 | 0.7264 | 75.4% | 0.9514 | 98.7% |
| Drosophila basset (group, 3 seeds) | 0.8255 / 0.8395 / 0.8401 | 0.4976 / 0.5045 / 0.5047 | **60.3 / 60.1 / 60.1%** | 0.6389 / 0.6134 / 0.6343 | 77.4 / 73.1 / 75.5% |
| Drosophila basset (random) | 0.8472 | 0.5101 | **60.2%** | 0.6223 | 73.5% |

Reading:

- The promoter models keep **82–89%** of their performance with the order destroyed, and that range is stable across seeds. Whatever they learned, they learned mostly from composition.
- The Drosophila CNN falls to AUPRC ≈0.50 — exactly its prevalence baseline, i.e. a coin flip in AUPRC terms — and its real and shuffled predictions correlate at r ≈ 0. It is genuinely reading order.
- The 99% shift-retention of the promoter models is architectural: `cat[mean, max]` pooling is translation-invariant by construction, and those windows contain no padding. The transformer is the only promoter model that loses accuracy under a circular shift (89.2%, AUROC −7.6 points), i.e. the only one using absolute position — and it is still the worst model on this data.
- The Drosophila shift-retention of 73–77% does **not** license the same architectural argument, and I do not claim it does. Rolling a padded, left-aligned sequence preserves the mask but creates one artificial junction between two sequence ends; the residual drop therefore conflates "absolute position matters" with "the model reacts to one novel junction". Treat it as an open question rather than evidence.

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
| 11 | validation size decided by the cluster shuffle | promoter runs validated on 34.6% of the data where 15% was requested; Drosophila's val moved between 778 and 2,731 rows across thresholds | a cluster larger than the budget never enters val; overshoot capped at 15% |
| 12 | one seed, reported as if it were an effect | the split fix flipped which model wins, and the between-seed spread (1.1 AUPRC points) is the same size as the differences being ranked | three seeds per configuration; table reports mean ± sd |

Bugs 8–12 were found *after* the pipeline worked, by asking what the padding actually did, whether any reported number was selected using the data it was reported on, and whether the split was doing what its name claims. Their full write-up, with before/after numbers, is in [`docs/DEVLOG.md`](docs/DEVLOG.md), which logs **17** silent failures in total. The pattern worth noting: none of them raised an exception, and each surfaced only when two independent measurements were compared — for 8 and 9 it was "same weights, different code, different score"; for 11 it was "the same seed gives a different validation size at a different threshold"; for 12 it was "fixing the split changed the winner".

## 10. Limitations

- **Three seeds for the Basset rows only.** The dilated, transformer and no-RC promoter runs are still single-seed, so their ordering against neighbouring rows should be read as provisional — the dilated lead over the 5-mer baseline is about two seed standard deviations and is the most likely of them to survive more seeds, or not.
- **The headline numbers needed a third pass.** Padding bugs (#8–#9) were found after the first version of these results; then the splitter bug (#11) forced every `group` row to be retrained, which changed the ranking; then the seed spread (#12) showed the ranking was never resolvable at one seed. Only the `dilated` Drosophila row is still a pre-fix model, because retraining it costs ~4 h on the 6 GB laptop GPU this was developed on.
- **20-mer homology is a proxy.** It misses remote homologues (so "0% sharing" is a lower bound) and single-linkage can chain-merge: on Drosophila one component holds 72% of the training rows, which means the effective number of independent training families is far below 5,184.
- **Probes are out-of-distribution.** Shuffle and shift create sequences no real genome produces; the clean control is retraining on shuffled data, which I have not run.
- **Attribution is diagnostic, not mechanistic.** The failure of IG and ISM to agree means *neither* should be used to claim a motif — not that a different attribution method would succeed.
- **One task family.** Promoters and enhancers at 251 bp and 3.2 kb; nothing here speaks to 100 kb-scale regulation (Enformer/Sei regime), where attention and long-range priors are known to matter.

## 11. What I would do next

1. Retrain basset on shuffled sequences as the positive control for Finding 7.
2. Extend the seed sweep to the dilated, transformer and no-RC rows — with three seeds the promoter ranking is still a coin flip between the CNN and the 5-mer model.
3. Sweep the homology threshold further (k = 22/25/30) — the pattern is already clear (at k ≤ 18 the split collapses to a single component), but the report would be stronger with the whole curve in one place.
4. Add `human_ocr_ensembl` (139,804 / 34,952) as a third task with a much larger sample size.
5. Replace the from-scratch transformer with a fine-tuned DNA language model (HyenaDNA / Nucleotide Transformer) — the current result is about *from-scratch* training, not about transformers.
6. Fix the residual validation-composition problem: the grouped validation set is 73.5% positive against 54.4% in test, which biases the fitted temperature and the threshold. Stratifying at the cluster level is the obvious next step, and it is not obviously the right one — it trades an unbiased operating point for a biased estimate of family-level generalisation.
