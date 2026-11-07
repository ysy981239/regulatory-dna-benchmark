"""Plain-assert smoke tests (no pytest needed).

    python tests/test_core.py

Covers the parts where a silent bug would quietly invalidate every number in the
writeup: encoding, reverse complement, exact k-mer coding, homology clustering and
the metric helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as gdata  # noqa: E402
from src import dna  # noqa: E402
from src import evaluate as ev  # noqa: E402


def test_encoding():
    ids = dna.seqs_to_ids(["ACGTN", "GGCC"])
    assert ids.shape == (2, 5), ids.shape
    assert ids.tolist() == [[0, 1, 2, 3, dna.UNK], [2, 2, 1, 1, dna.UNK]]
    assert dna.ids_to_seq(ids[0]) == "ACGTN"

    oh = dna.one_hot(ids)
    assert oh.shape == (2, 4, 5)
    assert oh.sum() == 8.0, "ambiguous base must contribute zero channels"
    assert np.all(oh[:, :, -1] == 0), "N column must be all-zero"


def test_reverse_complement():
    ids = dna.seqs_to_ids(["ACGT", "GGCC", "ANNT"])
    rc = dna.reverse_complement(ids)
    assert dna.ids_to_seq(rc[0]) == "ACGT"
    assert dna.ids_to_seq(rc[1]) == "GGCC"
    assert dna.ids_to_seq(rc[2]) == "ANNT"
    # involution
    assert np.array_equal(dna.reverse_complement(rc), ids)


def test_kmer_codes_exact_and_sentinel():
    # two identical k-mers 4 long, plus one unrelated row
    ids = dna.seqs_to_ids(["ACGTACGT", "ACGTACGT", "TTTTTTTT"])
    codes = dna.kmer_codes(ids, 4)
    assert codes.shape == (3, 5)
    assert np.array_equal(codes[0], codes[1])
    assert not np.array_equal(codes[0], codes[2])

    # a window containing N must never link two *different* rows
    ids_n = dna.seqs_to_ids(["NNNA", "NNNC"])
    codes_n = dna.kmer_codes(ids_n, 4)
    assert codes_n.shape == (2, 1)
    assert codes_n[0, 0] != codes_n[1, 0], "ambiguous windows must not be shared"

    # ... but byte-identical rows are genuine duplicates and must still link
    codes_dup = dna.kmer_codes(dna.seqs_to_ids(["NNNN", "NNNN"]), 4)
    assert codes_dup[0, 0] == codes_dup[1, 0]

    # regression: the sentinel must not depend on where a row sits in the batch
    ids_pad = dna.seqs_to_ids(["ACGT", "ACGG", "TTTA"], max_len=8)
    full = dna.kmer_codes(ids_pad, 4)
    for i in range(len(ids_pad)):
        solo = dna.kmer_codes(ids_pad[i : i + 1], 4)
        assert np.array_equal(full[i : i + 1], solo), f"row {i} is batch-dependent"
    # the padded tail of row 1 must not collide with the padded tail of row 0
    pad_0 = full[0][full[0] >= dna.SENTINEL_BASE]
    pad_1 = full[1][full[1] >= dna.SENTINEL_BASE]
    assert pad_0.size and pad_1.size
    assert not np.intersect1d(pad_0, pad_1).size, "padded tails must stay distinct"


def test_composition_features():
    ids = dna.seqs_to_ids(["GCGC", "ATAT"])
    gc = dna.gc_fraction(ids)
    assert np.allclose(gc, [1.0, 0.0]), gc
    din = dna.dinucleotide_frequencies(ids)
    assert din.shape == (2, 16)
    assert np.allclose(din.sum(axis=1), 1.0)
    five = dna.kmer_frequencies(ids, 2)
    assert five.shape == (2, 16)
    assert np.allclose(five.sum(axis=1), 1.0)


def test_homology_split_is_disjoint():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 4, size=(40, 200), dtype=np.uint8)
    # 10 exact duplicates of row 0 must all land in the same split
    ids = np.concatenate([base, np.repeat(base[:1], 10, axis=0)], axis=0)
    clusters = gdata.homology_clusters(ids, k=20)
    assert clusters[0] == clusters[-1], "duplicates must share a cluster"
    tr, va = gdata.grouped_split(np.ones(len(ids)), clusters, val_frac=0.2, seed=0)
    assert len(set(clusters[tr]) & set(clusters[va])) == 0, "clusters must not straddle"
    assert set(tr) | set(va) == set(range(len(ids)))


def test_metrics():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(float)
    p = np.clip(0.7 * y + 0.3 * rng.random(500), 0, 1)
    m = ev.binary_metrics(y, p, n_boot=50)
    assert 0.5 < m["auprc"] <= 1.0
    assert 0.5 < m["auroc"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0
    assert "auprc_ci95" in m and m["auprc_ci95"][0] <= m["auprc_ci95"][1]
    # a perfectly calibrated and shifted model: ECE must be small
    y2 = np.array([0.0] * 100 + [1.0] * 100)
    p2 = np.array([0.1] * 100 + [0.9] * 100)
    assert ev.ece(y2, p2) < 0.11
    # constant predictor must not raise
    m3 = ev.binary_metrics(y2, np.full(200, 0.5), n_boot=0)
    assert not np.isfinite(m3["auroc"])
    assert m3["auprc"] > 0.4


def test_registry():
    assert "human_nontata_promoters" in gdata.DATASETS
    assert gdata.split_url("x", "train").endswith("/v1/x/x_train.csv.gz")


def test_receptive_field():
    # Hand computation for Basset's 15/5 · 9/4 · 9/4 stack, counting the pooling
    # windows: 1 +14 +4 +8*5 +3*5 +8*20 +3*20 = 294.  Cross-checked by
    # backprop: perturbing one centre feature lights up exactly 294 input bases.
    import torch
    from src import models as gm

    m = gm.Basset(251)
    assert m.receptive_field() == 294
    # dilated stack has no pooling, so it is the plain sum of 4*dilation terms
    assert gm.DilatedCNN(3237).receptive_field() == 507
    # and the measured support must match the claim
    m.eval()
    x = torch.randn(1, 4, 251, requires_grad=True)
    h = m.features(x)
    h[0, :, h.shape[-1] // 2].sum().backward()
    support = (x.grad.abs().sum(dim=(0, 1)) > 0).sum().item()
    assert support <= m.receptive_field()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
