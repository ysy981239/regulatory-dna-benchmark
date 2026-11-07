"""How much does the homology-clustering threshold matter?

The split audit rests on one arbitrary constant: two sequences are "the same
family" if they share at least one exact k-mer, with k = 20.  That number was
picked, not derived.  This script asks three questions:

1. How do the clusters change with k (12 .. 30)?
2. Does the effective split change with k -- i.e. how many rows still land in
   validation, and does the largest component fall into train?
3. For the largest component, *which* k-mers hold it together -- a shared
   repeat / low-complexity sequence, or genuine homology?

Question 3 is the decisive one.  If one k-mer occurs in ~70% of the training
rows, the component is not a family: the threshold is gluing unrelated
sequences together through a common repeat, and "0% sharing under the grouped
split" is a statement about orphans, not about held-out families.

Writes ``reports/cluster_sensitivity_<dataset>.{json,md}``.
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


def kmer_to_seq(code: int, k: int) -> str:
    """Decode a base-4 k-mer code back to its string (A=0, C=1, G=2, T=3)."""
    out = []
    for i in range(k - 1, -1, -1):
        out.append("ACGT"[int(code // 4 ** i) % 4])
    return "".join(out)


def cluster(ids: np.ndarray, k: int):
    """Connected components of "shares at least one exact k-mer"."""
    codes = dna.kmer_codes(ids, k)
    n_rows, n_windows = codes.shape
    flat = codes.ravel()
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), n_windows)
    order = np.argsort(flat, kind="stable")
    f = flat[order]
    r = rows[order]
    same = f[1:] == f[:-1]
    a, b = r[1:][same], r[:-1][same]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    pairs = np.unique(lo * n_rows + hi)  # unique undirected edges
    graph = coo_matrix(
        (np.ones(pairs.size, dtype=np.int8), (pairs // n_rows, pairs % n_rows)),
        shape=(n_rows, n_rows),
    )
    n_clusters, labels = connected_components(graph, directed=False)
    return labels, n_clusters, int(pairs.size)


def document_frequency(ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(codes, number of rows containing each code), sentinels excluded."""
    codes = dna.kmer_codes(ids, k)
    n_rows = codes.shape[0]
    in_range = codes < np.uint64(4 ** k)
    flat = codes[in_range].astype(np.int64)
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), in_range.sum(axis=1))
    pair = flat * n_rows + rows
    uniq, counts = np.unique(pair, return_counts=True)
    return uniq // n_rows, counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="drosophila_enhancers_stark")
    p.add_argument("--ks", default="12,15,18,20,22,25,30")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-kmers", type=int, default=8)
    args = p.parse_args(argv)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ids, y, meta = gdata.load_split(args.dataset, "train")
    n_rows = ids.shape[0]
    print(f"[info] {args.dataset}: {n_rows} train rows, width {ids.shape[1]}")

    rows = []
    per_k = {}
    for k in ks:
        labels, n_clusters, n_edges = cluster(ids, k)
        sizes = np.bincount(labels, minlength=n_clusters)
        largest = int(sizes.max())
        component_pairs = int((sizes * (sizes - 1) // 2).sum())
        tr, va = gdata.grouped_split(y, labels, val_frac=args.val_frac, seed=args.seed)
        biggest_label = int(np.argmax(sizes))
        in_train = bool(np.isin(biggest_label, labels[tr]))
        rows.append(
            {
                "k": k,
                "n_clusters": int(n_clusters),
                "largest": largest,
                "largest_frac": round(largest / n_rows, 4),
                "mean_size": round(float(sizes.mean()), 2),
                "median_size": int(np.median(sizes)),
                "singletons": int((sizes == 1).sum()),
                "pairs_direct": n_edges,
                "pairs_closure": component_pairs,
                "amplification": round(component_pairs / max(n_edges, 1), 1),
                "val_rows": int(len(va)),
                "train_rows": int(len(tr)),
                "largest_in_train": in_train,
            }
        )
        per_k[k] = (labels, sizes, largest)
        print(
            f"[k={k:2d}] clusters {n_clusters:5d}  largest {largest:5d} "
            f"({largest / n_rows:5.1%})  singletons {rows[-1]['singletons']:5d}  "
            f"direct pairs {n_edges:9d}  closure pairs {component_pairs:11d}  "
            f"val {len(va):5d}"
        )

    # Which k-mers hold the largest component together, at the default k = 20?
    k_ref = 20 if 20 in per_k else ks[len(ks) // 2]
    labels, sizes, largest = per_k[k_ref]
    members = np.flatnonzero(labels == int(np.argmax(sizes)))
    codes_ref, counts_ref = document_frequency(ids, k_ref)
    member_codes, member_counts = document_frequency(ids[members], k_ref)
    order = np.argsort(member_counts)[::-1][: args.top_kmers]
    overall = dict(zip(codes_ref.tolist(), counts_ref.tolist()))
    top = [
        {
            "kmer": kmer_to_seq(int(c), k_ref),
            "rows_in_component": int(member_counts[i]),
            "frac_of_component": round(float(member_counts[i]) / largest, 4),
            "rows_in_dataset": overall.get(int(c), 0),
            "frac_of_dataset": round(overall.get(int(c), 0) / n_rows, 4),
        }
        for c, i in ((member_codes[i], i) for i in order)
    ]
    print(f"\n[component of {largest} rows, k={k_ref}] most widespread {k_ref}-mers:")
    for t in top:
        print(
            f"  {t['kmer']}  in {t['rows_in_component']:5d} rows "
            f"({t['frac_of_component']:6.1%} of the component, "
            f"{t['frac_of_dataset']:6.1%} of the dataset)"
        )

    summary = {
        "dataset": args.dataset,
        "n_rows": int(n_rows),
        "width": int(ids.shape[1]),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "sweep": rows,
        "largest_component_kmers": {"k": k_ref, "size": largest, "top": top},
    }
    out = ROOT / "reports" / f"cluster_sensitivity_{args.dataset}"
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        f"# Homology clustering: how much does `k` matter? ({args.dataset})",
        "",
        f"{n_rows} training rows, width {ids.shape[1]}. "
        '"Family" = shares at least one exact k-mer; components are single-linkage.',
        "",
        "| k | clusters | largest | largest % | singletons | direct pairs | closure pairs | amplification | val rows |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['k']} | {r['n_clusters']} | {r['largest']} | {r['largest_frac']:.1%} | "
            f"{r['singletons']} | {r['pairs_direct']:,} | {r['pairs_closure']:,} | "
            f"{r['amplification']}x | {r['val_rows']} |"
        )
    md += [
        "",
        f"## What holds the largest component together (k = {k_ref}, {largest} rows)",
        "",
        "| k-mer | rows in component | % of component | rows in dataset | % of dataset |",
        "|---|---|---|---|---|",
    ]
    for t in top:
        md.append(
            f"| `{t['kmer']}` | {t['rows_in_component']} | {t['frac_of_component']:.1%} | "
            f"{t['rows_in_dataset']} | {t['frac_of_dataset']:.1%} |"
        )
    md += [
        "",
        "Reading: a component is a *family* only if its members are related by descent. "
        "If a single k-mer is present in a large share of the component -- especially a "
        "low-complexity one -- the threshold is chaining unrelated sequences through a "
        "shared repeat, and the grouped split is holding out orphans rather than families.",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
