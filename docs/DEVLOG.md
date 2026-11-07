# Engineering log

Every bug below was found **after** the first version of the pipeline ran
cleanly and produced plausible numbers.  None of them raised an exception:
each one silently changed a result without breaking anything.  That is the
point of this file — for a benchmark-auditing project, the interesting part is
not that the code runs, it is the list of ways it was wrong before it was
right.

`ID` is stable and referenced from the code comments (`see docs/DEVLOG.md P1`).

For the one-page version — three detectors, and the number each failure moved —
see [AUDIT.md](AUDIT.md).  This file is the long form: root cause, fix and
verification for every entry.

| ID | Area | Symptom in one line | Severity | Found by |
|---|---|---|---|---|
| P2 | data | train and test tensors had different widths (3237 vs 3061) | high | asking "what does padding do" |
| P1 | model | global pooling averaged over padding, so a sequence's score depended on how much padding followed it | high | same question, one step further |
| E1 | metrics | a constant predictor reported AUROC 0.5 instead of "undefined" | medium | unit test |
| E2 | clustering | the leak audit reported 99.9% test/train overlap for *every* split, including the homologous one (which had 0 by construction) | high | the two numbers contradicted each other |
| E3 | metrics | the reported F1 used a threshold tuned on the same predictions it was scored on | medium | writing the DEVLOG, asking "what would a reviewer attack" |
| E4 | clustering | the k=20 "same family" threshold was never validated, and the giant component turns out to be chained by low-complexity repeats | high | asking "does the grouped split actually do anything" |
| E5 | clustering | the size of the validation set was decided by the shuffle: promoters ran with 34.6% val where 15% was asked for | high | sweeping the entropy threshold, then wondering why val moved |
| T1 | schedule | the LR never left the linear warmup: cosine annealing never happened | high | reading the LR trace, not the loss |
| T2 | schedule | epoch 1 trained at lr = 0 | medium | same trace |
| T3 | training | early stopping fired during the warmup plateau and cut the annealed tail | medium | run ended at epoch 19 of 30 |
| N1 | bookkeeping | the `--no-rc-augment` run silently overwrote the augmented run's directory | high | a result disappeared |
| M1 | model | transformer crashed: `mat1 and mat2 shapes cannot be multiplied (128x256 and 128x1)` | blocking | first training attempt |
| A1 | attribution | in-silico mutagenesis returned all-zero effects | medium | every variant looked neutral |
| A2 | attribution | motif reports were written to the repository root | low | stray files after a run |
| A3 | attribution | receptive field was documented as 215 bp, actually 294 bp | medium | deriving it by hand |
| D1 | tooling | the Chinese README was destroyed by a PowerShell encoding round-trip | high | reader failed on the file |

---

## P2 — train and test saw different tensor widths

**Symptom.** `drosophila_enhancers_stark` sequences are 236–3237 bp, so they are
right-padded to a rectangle.  The training tensor came out 3237 wide and the
*test* tensor 3061 wide, because the padding length was computed as
`lengths.max()` **inside each split separately**.

**Why it was invisible.** Convolution and global pooling are length-agnostic, so
nothing crashed.  With a transformer it would have crashed immediately
(positional-embedding table sized from the training length).

**Why it still mattered.** `mean(-1)` divides by `L`.  Training divided by 3237,
testing by 3061: the same weights were evaluated as *two different functions*.
BatchNorm's running statistics were also estimated at the wrong width.

**Fix.** `src/data.py` now resolves one padding length per **dataset**, pinned in
the registry (`max_len=3237`, `max_len=251`) or persisted in
`data/cache/max_len.json` for datasets that have no pinned value, and
`load_split` guarantees train and test share it.  `src/train.py` additionally
pads/truncates the test tensor to the training width as a defensive fallback.

**Verification.** `padding_check.py` and the load path both print the width;
`drosophila` now reports 3237 / 3237 and `promoters` 251 / 251.

## P1 — global pooling averaged over padding

**Symptom.** A 236 bp sequence inside a 3237 bp tensor had ~92.6% of its
`mean`-pooled feature drawn from padding.

**Root cause chain.** Padding is encoded as all-zero one-hot columns →
`Conv1d(bias=False)` maps them to 0 → **BatchNorm maps 0 to a per-channel
constant** (`γβ`) → ReLU keeps it.  Padding is therefore *not* silent: it is a
constant that dominates both `mean` (denominator = padded length) and `max`
(the constant can exceed every real activation), and it also contaminates the
BatchNorm batch statistics.

**Why it was invisible.** `promoters` is 251 bp fixed — **zero padding** — so the
whole mechanism does not exist there.  Every promoter number was safe, which is
exactly why the bug could survive the earlier experiments.

**Fix.** `src/models.py` gained three helpers — `valid_mask` (one-hot sum > 0.5),
`down_mask` (the mask is pooled with **max** semantics: a window is valid if it
contains at least one real base) and `masked_pool` (`mean` divided by the number
of valid positions, `max` via `masked_fill(-inf)`, all-empty windows zeroed).
`Basset.forward` now walks its blocks explicitly so the mask follows the same
pooling as the features; `DilatedCNN` masks its pooling; `DNATransformer` masks
both the attention (`src_key_padding_mask`) and the pooling.

**Verification.** `tests/test_core.py::test_padding_invariance` scores the same
120 bp sequence padded to 160 and to 520 and demands equal logits.  Measured:
max |difference| = 0.000000 with masked pooling, 0.0059 with the old
`mean`/`max` (so the test genuinely fails on the old code).

## E1 — a constant predictor has no AUROC, not AUROC 0.5

**Symptom.** `assert not np.isfinite(m3["auroc"])` failed.

**Root cause.** `sklearn` 1.7.2 returns 0.5 for a constant score vector instead
of raising, so the `try/except ValueError` fallback never fired.

**Fix.** `src/evaluate.py`'s `_safe(..., needs_ranking=True)` returns `nan` when
`unique(y).size < 2` or `unique(p).size < 2`, and also when the result is not
finite.  "0.5 = random performance" is a *false* statement for a model that
cannot rank at all; `nan` is the honest value.

## E2 — the leak audit was measuring its own bug

**Symptom.** The audit reported that **99.9% of test rows share a 20-mer with
training**, and it reported the *same* 99.9% for the homology-grouped split,
which by construction shares nothing.  The clustering step had found only 1309
clusters, which is impossible if 99.9% of rows really shared k-mers.

**Root cause.** Windows containing `N` are given a sentinel k-mer code to keep
them from linking unrelated sequences.  That sentinel was
`4**31 + row_index` — i.e. it depended on the row's position *in the batch*.
Probing one row at a time made every row "row 0", so **every padded row of every
sequence collided with row 0 of the training set**.

**Fix.** `src/dna.py` derives the sentinel from a `blake2b` hash of the row's
bytes (`_row_keys`): content-dependent, batch-independent.  Byte-identical rows
still link (they are genuine duplicates); different rows never do.

**Verification.** Regression assertion in `test_kmer_codes_exact_and_sentinel`:
`kmer_codes(ids[i:i+1], 4)` must equal `kmer_codes(ids, 4)[i:i+1]`, and the padded
tails of two different rows must not intersect.

**Lesson.** The contradictory pair of numbers (99.9% overlap *and* 0 shared
clusters) was the only signal.  Neither number alone looked wrong.

## E4 — the "same family" threshold was never validated

**Symptom.** None. `k = 20` looked like a reasonable default, and both the leak audit and the grouped split inherited it silently.

**What it decides.** The entire hold-out protocol. "Two sequences are the same family" is defined as *shares at least one exact k-mer*, so `k` sets both how much of the test set counts as contaminated and which rows can serve as validation.

**Sweep (`scripts/cluster_sensitivity.py`).** Largest connected component and the resulting validation size:

| k | Drosophila: clusters / largest / val rows | Promoters: clusters / largest / val rows |
|---|---|---|
| 12 | 1 / 5,184 (100%) / 5,184 | 1 / 27,097 (100%) / 27,097 |
| 15 | 1 / 5,184 (100%) / 5,184 | 722 / 26,234 (97%) / 26,689 |
| 18 | 268 / 4,905 (95%) / 4,930 | 8,847 / 12,945 (48%) / 4,093 |
| **20** | **1,309 / 3,723 (72%) / 778** | **12,254 / 8,307 (31%) / 9,384** |
| 22 | 2,246 / 2,608 (50%) / 778 | 13,489 / 6,574 (24%) / 4,065 |
| 25 | 3,158 / 1,430 (28%) / 778 | 14,189 / 5,533 (20%) / 4,065 |
| 30 | 3,793 / 511 (10%) / 779 | — |

`k = 20` is **the smallest value at which the protocol still works at all**: at k ≤ 18 the split would have to put 95–100% of the training set into validation. That is fragility, not justification, and it should have been reported as such from the beginning.

**Second finding — the component is not a family.** The most widespread 20-mers *inside* the largest component are low-complexity repeats, not conserved sequence:

| dataset | top 20-mer in the largest component | rows |
|---|---|---|
| Drosophila | `ATTTATATATTTATATATTT` (AT microsatellite) | 71 (1.9% of the component) |
| Drosophila | `ATAATAATAATAATAATAAT` | 71 |
| promoters | `AAAAAAAAAAAAAAAAAAAA` | 38 (0.5%) |
| promoters | `AAGAAGAAGAAGAAGAAGAA` | 28 |

No single k-mer dominates — the component is chained through *many* weak links of ~1% each, which is exactly the signature of a shared repeat rather than a shared ancestor.

**Quantified (`scripts/complexity_probe.py`).** A 20-mer built only from A and T has base-composition entropy ≤ 1 bit, so a minimum-entropy filter removes repeats while keeping real sequence. Drosophila: 77.2% → 69.7% (≥1.2 bits) → **59.5%** (≥1.5) → 51.2% (≥1.7) → 40.5% (≥1.9), and the largest component shrinks 72% → 59% → 40% → 26%. Promoters barely move: 57.8% → **49.9%** → 48.9% → 47.9%.

**Interpretation.** Enhancer sharing is roughly one third AT-repeat noise, two thirds real similarity. Promoter sharing is genuine CpG-island similarity. So the headline was not wrong, but it was overstated: the defensible statement is *at least* 59% (enhancers) / 50% (promoters), not 77% / 55%.

**Side finding.** `reports/leakage_*.json` reports 54.8% for promoters because `leakage_report` subsamples train to 20,000 rows; the full 27,097-row pool gives 57.8%. The published number is a slight *under*-estimate, which is the harmless direction.

**Lesson.** Any constant that carries a conclusion needs (a) a sweep showing where the conclusion breaks and (b) a mechanism test showing *why* the grouping happened. Both were one script away and neither was written until the split's value was challenged.

## E5 — the size of the validation set was decided by a shuffle

**Symptom.** `grouped_split` walked clusters in shuffled order and stopped at the
first crossing of the target size. With one dominant cluster, whether that
cluster landed before or after the crossing changed the validation set
completely. Measured on the real data (`scripts/masked_split.py`):

| dataset | asked for | actually got | spread over thresholds |
|---|---|---|---|
| drosophila | 778 | 778 / 778 / **2,731** / 778 | 3.5× |
| promoters | 4,065 | **9,384** / 6,381 / 6,199 / 4,065 | 2.3× |

The promoter runs therefore never had the 15% validation set the command line
requested — they had **34.6%**. Nothing raised, and the val curves looked fine,
because a bigger validation set is not visibly broken, just different.

**Why it matters.** Validation size is not cosmetic: it sets the resolution of
early stopping, of the fitted temperature, of every calibration number, and of
the reported bootstrap-free val AUPRC that the baseline comparison used.

**Fix.** Two rules, no new dependency (`src/data.py`): a cluster larger than the
target may never enter validation (it would eat the whole budget), and a cluster
is skipped if adding it would overshoot by more than 15%. After the fix both
datasets land on the requested size at every entropy threshold — 778 and 4,065
exactly, across 0.0 / 1.2 / 1.5 / 1.7 bits.

**Verification.** `tests/test_core.py::test_grouped_split_size_is_stable` builds
300 rows with one 120-row blob, sweeps 8 seeds, and asserts val is exactly 45
rows, that the blob never enters val, and that no family is split. On the old
code the same test fails for every seed where the blob is drawn first.

**Limitation this exposes rather than fixes.** Validation is now a random draw of
*small* families, not a random sample of families: any cluster bigger than 15% of
the data can never be validated on at all. That is a property of the protocol,
not of the function, and it means grouped-split numbers measure "generalisation
to unseen small families" — a weaker claim than "unseen families".

**Prospective, not retroactive.** The `group` rows in the results tables were
produced by the old split. Re-running them changes val selection, hence early
stopping and the threshold, so the numbers would move slightly; they have not
been regenerated and the report says so.

## T1 / T2 / T3 — the learning-rate schedule was never executed

**Symptom (T1).** The LR trace went 9.7e-06 → 2.9e-04 over 30 epochs, i.e. it
was still climbing at the last epoch and reached only ~29% of `--lr`.  The
cosine annealing never happened.

**Root cause.** The schedule was defined in **steps**
(`total_steps = epochs * steps_per_epoch`) but stepped once per **epoch**
(`sched.step()` in the epoch loop), so 30 of 5400 steps were ever consumed.

**Symptom (T2).** Epoch 1 ran at lr = 0.

**Root cause.** `LambdaLR` evaluates `lr_lambda(0)` at construction; with
`warmup = 1` the lambda returned `0/1 = 0`.

**Fix.** Define the schedule in epochs (`total_steps = epochs`) and floor the
step counter (`step = max(step, 1)`).  Verified trace: 5e-4 → 1e-3 at epoch 2 →
3.1e-06 at epoch 29 → 0 at epoch 30.

**Symptom (T3).** Runs stopped at epoch 19/30 with `--patience 8`.

**Root cause.** During warmup the validation metric is flat, so patience expired
before the annealed tail — exactly the epochs where the model converges best.

**Fix.** `--patience` now defaults to `0` (run the full schedule) and the break
condition is guarded by `if args.patience > 0`.

## N1 — run directories collided

`--no-rc-augment` shared the run tag with the augmented run, so the second run
overwrote the first one's `best.pt` and `metrics.json`.  The tag now carries the
variant (`basset-norc`, `-w{width}`), and the affected directory was renamed
after the fact.  Reported numbers for the RC ablation are from the renamed,
intact directory.

## M1 — the transformer head had the wrong width

`self.head = nn.Linear(d_model, 1)` while the forward pass concatenates
mean- and max-pooling, i.e. `2 * d_model`.  It crashed only when a transformer
was actually trained (the CNN heads are shaped correctly).  Fixed to
`nn.Linear(2 * d_model, 1)` and smoke-tested on all three architectures.

## E3 — the F1 threshold was tuned on the test predictions

**Symptom.** None: the number looked fine. `binary_metrics` swept the decision
threshold over the scores it was given and reported the best F1 — and the caller
was the *test* split. So `best_f1 = 0.83` was "the best F1 achievable if you were
allowed to pick the threshold after seeing the test labels", which is not a test
statistic.

**Why it matters here specifically.** The whole project is about numbers that are
optimistic because something was fitted on the wrong side of the split. It would
be inconsistent to ship the same mistake in the metric code.

**Fix.** `src/evaluate.py` now exposes `best_f1_threshold(y, p)` and
`binary_metrics(..., given_threshold=...)`. `src/train.py` picks the threshold on
the **validation** predictions after restoring the best checkpoint and reports
`precision/recall/f1/specificity_at_given` plus `val_best_f1_threshold`. The
on-split optimum is still reported (as `at_best_f1`) but its docstring now says
it is an oracle.

**Effect.** `f1_at_given` (honest) sits between `f1_at_0.5` (arbitrary) and
`f1_at_best_f1` (oracle) — the gap between the last two is the size of the
optimism, typically 1–3 F1 points.

## A1 — in-silico mutagenesis measured nothing

The variant table reported `+0.0000` for every single-base substitution.  The
model was saturated at p ≈ 1.000, and a probability difference at that
saturation is below float32 resolution.  Effects are now computed in **logit**
space (`space="logit"`), and the reported `delta_prob` is back-computed as
`sigmoid(z_ref + dz) - p` for display.  Integrated gradients was already in
logit space, so the IG-vs-ISM agreement statistic now compares two quantities in
the same units.

## A2 — attribution outputs landed in the repository root

`--out-prefix foo` produced `./foo.md`.  A bare filename now resolves to
`reports/`.

## A3 — the receptive field was wrong by 79 bp

`Basset.receptive_field()` returned 215 by treating pooling as pure
downsampling.  Counting the pooled windows too gives
`1 + 14 + 4 + 40 + 15 + 160 + 60 + 1 = 294`, confirmed by backprop (a single
centre feature touches exactly 294 input bases).  Fixed, and
`test_receptive_field` now asserts both 294 (Basset) and 507 (dilated CNN) and
cross-checks the measured support.

## D1 — the README was destroyed by an encoding round-trip

The Chinese README was edited through `Get-Content -Raw | Set-Content -Encoding
utf8`.  This PowerShell is 5.1, whose `Encoding.Default` is **gb2312**: the file
was decoded as gb2312 and re-encoded as UTF-8, which is lossy and irreversible
(326 characters became `?`).  The repository had no git history to restore from,
so the file was rebuilt from scratch.  Policy since then: non-ASCII files are
only touched by tools that read and write UTF-8 explicitly.

---

## Timeline

| stage | commit | what became true |
|---|---|---|
| first version | `8548ec9` | the pipeline ran end to end and produced plausible numbers — and was wrong in all the ways above |
| padding audit | `674b6e1` | P1 / P2 / E3 fixed (masked pooling, one tensor width, threshold on validation); Drosophila retrained; every probe re-measured on the fixed weights |
| threshold audit | `0a4107f` | `k` swept and stress-tested against sequence complexity; the leak figure restated as a range (59–77%) instead of a point |

The point of this table is that the first row is where a normal project stops.

## What this list is actually evidence of

1. **Every bug was silent.** Not one raised an exception, and every one produced
   numbers that looked reasonable. A pipeline that "runs" is not a pipeline that
   is correct.
2. **The bugs that matter were found by cross-checking two measurements**, not
   by reading code: 99.9% overlap vs 0 shared clusters (E2); padding fraction
   34.6% vs a plausible score (P1); an LR trace that never annealed (T1).
3. **The dangerous ones were dataset-dependent.** `promoters` (fixed length, zero
   padding) is immune to P1/P2 entirely. A single-dataset project would never
   have found them.
4. **Negative results are what forced the audit.** The model did not beat a 5-mer
   logistic regression, and that is the only reason to ask what it actually
   learned — which is what led to the padding questions in the first place.

## If I rebuilt this from scratch

- Check the data contract first: dump the tensor shape of every split before
  writing a model, and assert train/test shapes are equal.
- Print the padding fraction and the real-length distribution on every run.
- Assert, don't comment: shape/width/mask invariants belong in `tests/`.
- Log the LR trace in `history.json` (it is what exposed T1/T2 — the loss looked
  fine).
- Put the variant name (augmentation, width, seed) in the output path, not in a
  README table.
