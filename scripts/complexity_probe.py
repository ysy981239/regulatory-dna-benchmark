"""Is the "leakage" and the giant cluster just low-complexity sequence?

The k-sweep (``scripts/cluster_sensitivity.py``) showed that the largest
component at k = 20 is held together by AT-only 20-mers -- e.g.
``ATTTATATATTTATATATTT`` -- each shared by ~70 sequences.  Those are repeats,
not homologs.  If one shared repeat is enough to merge two sequences, then the
number "77% of test sequences share a 20-mer with train" may be measuring
AT-microsatellites rather than near-duplicates.

This script separates the two.  For each k-mer we compute the Shannon entropy of
its base composition (20-mers made only of A and T have entropy <= 1 bit);
filtering on a minimum entropy forces the k-mer to use at least three bases, so
repeats and low-complexity runs drop out.  Then it reports, as a function of the
entropy threshold:

* fraction of test rows that still share >= 1 k-mer with train,
* largest connected component of the training set,
* number of clusters.

Writes ``reports/complexity_probe_<dataset>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as gdata  # noqa: E402
from src import dna  # noqa: E402

CHUNK = 2_000_000


def entropy_bits(codes: np.ndarray, k: int) -> np.ndarray:
    """Shannon entropy (bits) of each k-mer's base composition; max is 2 bits.

    Chunked: the counts table is 4 float32 per k-mer, which would be ~270 MB for
    the largest tensor here.
    """
    out = np.empty(codes.size, dtype=np.float32)
    for start in range(0, codes.size, CHUNK):
        block = codes[start : start + CHUNK]
        counts = np.zeros((block.size, 4), dtype=np.float32)
        local = np.arange(block.size)
        for i in range(k):
            digit = ((block >> np.uint64(2 * i)) & np.uint64(3)).astype(np.int64)
            counts[local, digit] += 1.0
        p = counts / float(k)
        nz = np.where(p > 0, p, 1.0)
        out[start : start + block.size] = (-(np.where(p > 0, p * np.log2(nz), 0.0)).sum(axis=1))
    return out


def codes_valid(codes: np.ndarray, k: int) -> np.ndarray:
    """Exclude sentinel windows (they encode "this window had an N/pad")."""
    return codes < np.uint64(4 ** k)


def rows_sharing(test_codes: np.ndarray, pool: np.ndarray, k: int) -> int:
    """How many rows of ``test_codes`` contain at least one code in ``pool``."""
    if pool.size == 0:
        return 0
    ok = codes_valid(test_codes, k)
    flat = test_codes[ok]
    rows = np.repeat(np.arange(test_codes.shape[0], dtype=np.int64), ok.sum(axis=1))
    idx = np.clip(np.searchsorted(pool, flat), 0, pool.size - 1)
    found = pool[idx] == flat
    hit = np.zeros(test_codes.shape[0], dtype=bool)
    hit[rows[found]] = True
    return int(hit.sum())


def largest_component(codes: np.ndarray) -> tuple[int, int]:
    """(n_clusters, size of the largest component) for "shares one code"."""
    n_rows, n_windows = codes.shape
    flat = codes.ravel()
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), n_windows)
    order = np.argsort(flat, kind="stable")
    f, r = flat[order], rows[order]
    same = f[1:] == f[:-1]
    a, b = r[1:][same], r[:-1][same]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pairs = np.unique(lo * n_rows + hi)
    graph = coo_matrix(
        (np.ones(pairs.size, dtype=np.int8), (pairs // n_rows, pairs % n_rows)),
        shape=(n_rows, n_rows),
    )
    n_clusters, labels = connected_components(graph, directed=False)
    return int(n_clusters), int(np.bincount(labels).max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="drosophila_enhancers_stark")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--thresholds", default="0,1.2,1.5,1.7,1.9")
    p.add_argument("--cluster-at", default="1.2,1.5,1.7")
    args = p.parse_args(argv)

    k = args.k
    thresholds = [float(x) for x in args.thresholds.split(",")]
    cluster_at = [float(x) for x in args.cluster_at.split(",")]

    ids_tr, y_tr, _ = gdata.load_split(args.dataset, "train")
    ids_te, _, _ = gdata.load_split(args.dataset, "test")
    codes_tr = dna.kmer_codes(ids_tr, k)
    codes_te = dna.kmer_codes(ids_te, k)
    print(f"[info] {args.dataset}: train {ids_tr.shape[0]}, test {ids_te.shape[0]}, k={k}")
    print("[info] computing k-mer composition entropy ...")
    ent_tr = entropy_bits(codes_tr.ravel(), k).reshape(codes_tr.shape)
    ent_te = entropy_bits(codes_te.ravel(), k).reshape(codes_te.shape)
    print(f"[info] entropy: train median {np.median(ent_tr):.3f} bits, "
          f"{np.mean(ent_tr <= 1.0):.1%} of windows are A+T only")

    rows = []
    for t in thresholds:
        keep_tr = codes_valid(codes_tr, k) & (ent_tr >= t)
        pool = np.unique(codes_tr[keep_tr])
        n_hit = rows_sharing(codes_te, pool, k)
        rows.append(
            {
                "min_entropy_bits": t,
                "train_kmers_kept": int(pool.size),
                "test_rows_sharing": n_hit,
                "frac_test_sharing": round(n_hit / ids_te.shape[0], 4),
            }
        )
        print(
            f"[H>={t:3.1f}] train pool {pool.size:9d} kmers   "
            f"test rows sharing {n_hit:5d} ({n_hit / ids_te.shape[0]:6.1%})"
        )

    clusters = []
    sentinel = np.uint64(4 ** k) + np.arange(
        1, ids_tr.shape[0] + 1, dtype=np.uint64
    )[:, None]
    for t in cluster_at:
        invalid = (~codes_valid(codes_tr, k)) | (ent_tr < t)
        keep = np.where(invalid, sentinel, codes_tr)
        n_clusters, largest = largest_component(keep)
        clusters.append(
            {
                "min_entropy_bits": t,
                "n_clusters": n_clusters,
                "largest": largest,
                "largest_frac": round(largest / ids_tr.shape[0], 4),
            }
        )
        print(
            f"[H>={t:3.1f}] clusters {n_clusters:5d}   largest {largest:5d} "
            f"({largest / ids_tr.shape[0]:5.1%})"
        )

    summary = {
        "dataset": args.dataset,
        "k": k,
        "n_train": int(ids_tr.shape[0]),
        "n_test": int(ids_te.shape[0]),
        "frac_at_only": round(float(np.mean(ent_tr <= 1.0)), 4),
        "leakage": rows,
        "clustering": clusters,
    }
    out = ROOT / "reports" / f"complexity_probe_{args.dataset}"
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        f"# Is the leakage / the giant cluster low-complexity sequence? ({args.dataset})",
        "",
        f"k = {k}, train {ids_tr.shape[0]} rows, test {ids_te.shape[0]} rows. "
        "Entropy is the Shannon entropy of the k-mer's base composition "
        "(2 bits = all four bases equally present; <= 1 bit = only A+T or only C+G). "
        f"In the training tensor {np.mean(ent_tr <= 1.0):.1%} of windows are A+T only.",
        "",
        "## Sharing between test and train",
        "",
        "| min entropy (bits) | k-mers kept | test rows sharing >=1 | % of test |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['min_entropy_bits']:.1f} | {r['train_kmers_kept']:,} | "
            f"{r['test_rows_sharing']} | {r['frac_test_sharing']:.1%} |"
        )
    md += [
        "",
        "## Clusters after dropping low-complexity k-mers",
        "",
        "| min entropy (bits) | clusters | largest | largest % |",
        "|---|---|---|---|",
    ]
    for c in clusters:
        md.append(
            f"| {c['min_entropy_bits']:.1f} | {c['n_clusters']} | {c['largest']} | "
            f"{c['largest_frac']:.1%} |"
        )
    md += [
        "",
        "Reading: if the sharing collapses once low-complexity k-mers are removed, then "
        "most shared 20-mers are repeats and the honest wording is "
        "shares a low-complexity 20-mer, not near-duplicate sequence.",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
