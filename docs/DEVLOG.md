# Engineering log

Every bug below was found **after** the first version of the pipeline ran
cleanly and produced plausible numbers.  None of them raised an exception:
each one silently changed a result without breaking anything.  That is the
point of this file — for a benchmark-auditing project, the interesting part is
not that the code runs, it is the list of ways it was wrong before it was
right.

`ID` is stable and referenced from the code comments (`see docs/DEVLOG.md P1`).

| ID | Area | Symptom in one line | Severity | Found by |
|---|---|---|---|---|
| P2 | data | train and test tensors had different widths (3237 vs 3061) | high | asking "what does padding do" |
| P1 | model | global pooling averaged over padding, so a sequence's score depended on how much padding followed it | high | same question, one step further |
| E1 | metrics | a constant predictor reported AUROC 0.5 instead of "undefined" | medium | unit test |
| E2 | clustering | the leak audit reported 99.9% test/train overlap for *every* split, including the homologous one (which had 0 by construction) | high | the two numbers contradicted each other |
| E3 | metrics | the reported F1 used a threshold tuned on the same predictions it was scored on | medium | writing the DEVLOG, asking "what would a reviewer attack" |
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
