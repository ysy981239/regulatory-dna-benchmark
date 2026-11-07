"""Family-aware split with low-complexity k-mers removed from the criterion.

`homology_clusters` says two sequences are the same family if they share one
exact k-mer.  On AT-rich enhancers that criterion is dominated by repeats: a
single shared microsatellite merges two unrelated sequences.  This script keeps
the criterion but filters the k-mers themselves -- a k-mer whose base-composition
entropy is below a threshold is not allowed to link anything -- and then reports
what the split looks like.

The point is not to make the score go up.  The point is to answer, with numbers:
*how much of the training set is actually independent*, and *how much of the
official test set would still be contaminated under the stricter definition*.

Writes ``reports/masked_split_<dataset>.{json,md}``.
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
        out[start : start + block.size] = -(np.where(p > 0, p * np.log2(nz), 0.0)).sum(axis=1)
    return out


def masked_codes(ids: np.ndarray, k: int, min_ent: float) -> np.ndarray:
    """k-mer codes where invalid / low-complexity windows get a per-row sentinel."""
    codes = dna.kmer_codes(ids, k)
    if min_ent > 0:
        ent = entropy_bits(codes.ravel(), k).reshape(codes.shape)
        keep = (codes < np.uint64(4 ** k)) & (ent >= min_ent)
    else:
        keep = codes < np.uint64(4 ** k)
    sentinel = np.uint64(4 ** k) + np.arange(1, ids.shape[0] + 1, dtype=np.uint64)[:, None]
    return np.where(keep, codes, sentinel)


def cluster_from_codes(codes: np.ndarray) -> tuple[np.ndarray, int, int]:
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
    return labels, int(n_clusters), int(pairs.size)


def rows_sharing(codes_a: np.ndarray, pool_codes: np.ndarray, k: int) -> int:
    pool = np.unique(pool_codes[pool_codes < np.uint64(4 ** k)])
    if pool.size == 0:
        return 0
    ok = codes_a < np.uint64(4 ** k)
    flat = codes_a[ok]
    rows = np.repeat(np.arange(codes_a.shape[0], dtype=np.int64), ok.sum(axis=1))
    idx = np.clip(np.searchsorted(pool, flat), 0, pool.size - 1)
    hit_rows = rows[pool[idx] == flat]
    return int(np.unique(hit_rows).size)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="drosophila_enhancers_stark")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--min-ents", default="0,1.2,1.5,1.7")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ids_tr, y_tr, _ = gdata.load_split(args.dataset, "train")
    ids_te, _, _ = gdata.load_split(args.dataset, "test")
    print(f"[info] {args.dataset}: train {ids_tr.shape[0]}, test {ids_te.shape[0]}, k={args.k}")

    rows = []
    for t in [float(x) for x in args.min_ents.split(",")]:
        c_tr = masked_codes(ids_tr, args.k, t)
        labels, n_clusters, n_edges = cluster_from_codes(c_tr)
        sizes = np.bincount(labels)
        tr, va = gdata.grouped_split(y_tr, labels, val_frac=args.val_frac, seed=args.seed)
        c_te = masked_codes(ids_te, args.k, t)
        share_te = rows_sharing(c_te, c_tr, args.k)
        share_va = rows_sharing(c_tr[va], c_tr[tr], args.k)
        rows.append(
            {
                "min_entropy_bits": t,
                "n_clusters": n_clusters,
                "largest": int(sizes.max()),
                "largest_frac": round(float(sizes.max()) / len(labels), 4),
                "singletons": int((sizes == 1).sum()),
                "edges": n_edges,
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "val_clusters": int(len(np.unique(labels[va]))),
                "test_sharing_full_train": share_te,
                "frac_test_sharing": round(share_te / ids_te.shape[0], 4),
                "val_sharing_with_train": share_va,
            }
        )
        print(
            f"[H>={t:3.1f}] clusters {n_clusters:6d}  largest {sizes.max():6d} "
            f"({sizes.max() / len(labels):5.1%})  val {len(va):5d} rows / "
            f"{rows[-1]['val_clusters']:5d} clusters  "
            f"test sharing {share_te:5d} ({share_te / ids_te.shape[0]:6.1%})  "
            f"val sharing {share_va}"
        )

    out = ROOT / "reports" / f"masked_split_{args.dataset}"
    out.with_suffix(".json").write_text(
        json.dumps({"dataset": args.dataset, "k": args.k, "sweep": rows}, indent=2),
        encoding="utf-8",
    )
    md = [
        f"# Family-aware split with low-complexity k-mers excluded ({args.dataset})",
        "",
        f"k = {args.k}, train {ids_tr.shape[0]} rows, test {ids_te.shape[0]} rows. "
        "A k-mer below the entropy threshold may not link two sequences.",
        "",
        "| min entropy | clusters | largest | largest % | val rows | val clusters | test sharing | val sharing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['min_entropy_bits']:.1f} | {r['n_clusters']} | {r['largest']} | "
            f"{r['largest_frac']:.1%} | {r['val_rows']} | {r['val_clusters']} | "
            f"{r['frac_test_sharing']:.1%} | {r['val_sharing_with_train']} |"
        )
    md += [
        "",
        "Reading: `val rows` is the statistical power the split actually has; "
        "`largest %` is how much of the training set is still one blob; "
        "`test sharing` is the part of the benchmark no training-side clustering can repair.",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
