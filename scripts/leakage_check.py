"""Are you measuring biology or measuring the benchmark?

This script answers that question *before* you train anything, with four probes
that need no deep learning:

1. **single-feature separability** - how much of the task is explained by GC
   content alone?  If a one-number baseline already reaches AUC ~0.8, your CNN
   is being tested on something much narrower than "regulatory grammar".
2. **composition baselines** - dinucleotide and 5-mer frequencies with a linear
   model.  This is the baseline every paper should report and most do not.
3. **train/test near-duplication** - what fraction of test rows share an exact
   20-mer with train?  Shared 20-mers between unrelated 200 bp sequences are
   vanishingly unlikely, so this number is a homology/duplication estimate.
4. **split-policy gap** - the metric difference between a random row split and a
   homology-clustered split, measured on identical data.

Run it on every dataset before you believe any model number.

Usage
-----
    python scripts/leakage_check.py --dataset human_nontata_promoters
    python scripts/leakage_check.py --dataset drosophila_enhancers_stark --out reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as gdata  # noqa: E402
from src import dna  # noqa: E402
from src import evaluate as ev  # noqa: E402


def linear_baseline(X_tr, y_tr, X_te, y_te, name: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
    )
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    m = ev.binary_metrics(y_te, p, n_boot=0)
    return {"baseline": name, "n_features": int(X_tr.shape[1]),
            "auroc": m["auroc"], "auprc": m["auprc"], "ece": m["ece"]}


def near_duplicate_rate(ids_tr, ids_te, k: int = 20, n_probe: int = 2000,
                        n_train_cap: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tr = ids_tr if len(ids_tr) <= n_train_cap else ids_tr[rng.choice(len(ids_tr), n_train_cap, replace=False)]
    te_idx = rng.choice(len(ids_te), min(n_probe, len(ids_te)), replace=False)
    codes = set(dna.kmer_codes(tr, k).ravel().tolist())
    hits = 0
    for i in te_idx:
        row = dna.kmer_codes(ids_te[i:i + 1], k).ravel().tolist()
        if any(c in codes for c in row):
            hits += 1
    return {"kmer_k": k, "probed": int(len(te_idx)),
            "frac_test_sharing_a_kmer_with_train": round(hits / max(len(te_idx), 1), 4)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="human_nontata_promoters")
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "reports"))
    p.add_argument("--kmer", type=int, default=5)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_tr, y_tr, meta = gdata.load_split(args.dataset, "train", max_len=args.max_len)
    ids_te, y_te, _ = gdata.load_split(args.dataset, "test", max_len=args.max_len)
    L = meta["length"]

    results = {
        "dataset": args.dataset,
        "length": L,
        "train": gdata.class_balance(y_tr),
        "test": gdata.class_balance(y_te),
        "baselines": [
            linear_baseline(dna.gc_fraction(ids_tr).reshape(-1, 1), y_tr,
                            dna.gc_fraction(ids_te).reshape(-1, 1), y_te, "GC content only"),
            linear_baseline(dna.dinucleotide_frequencies(ids_tr), y_tr,
                            dna.dinucleotide_frequencies(ids_te), y_te, "dinucleotide freq (16)"),
            linear_baseline(dna.kmer_frequencies(ids_tr, args.kmer), y_tr,
                            dna.kmer_frequencies(ids_te, args.kmer), y_te,
                            f"{args.kmer}-mer freq ({4 ** args.kmer})"),
        ],
        "near_duplication": near_duplicate_rate(ids_tr, ids_te),
    }

    # same-split comparison: how much does the split policy alone move the metric
    clusters = gdata.homology_clusters(ids_tr, k=20)
    n_comp = len(np.unique(clusters))
    results["homology_clusters"] = {"k": 20, "n_clusters": int(n_comp),
                                    "mean_cluster_size": round(len(y_tr) / n_comp, 2)}
    tr_g, va_g = gdata.grouped_split(y_tr, clusters, 0.15, seed=0)
    tr_r, va_r = gdata.stratified_split(y_tr, 0.15, seed=0)
    results["split_leakage"] = {
        "random_row_split": gdata.leakage_report(tr_r, va_r, clusters, ids_tr),
        "homology_grouped_split": gdata.leakage_report(tr_g, va_g, clusters, ids_tr),
    }

    (out_dir / f"leakage_{args.dataset}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Leakage & separability audit - `{args.dataset}`", "",
             f"- sequence length: {L} bp",
             f"- train: {results['train']['n']} rows, {results['train']['pos_frac']:.1%} positive",
             f"- test:  {results['test']['n']} rows, {results['test']['pos_frac']:.1%} positive",
             "", "## Baselines that use no learned sequence grammar", "",
             "| baseline | #features | test AUROC | test AUPRC | ECE |",
             "|---|---|---|---|---|"]
    for b in results["baselines"]:
        lines.append(f"| {b['baseline']} | {b['n_features']} | {b['auroc']:.4f} | "
                     f"{b['auprc']:.4f} | {b['ece']:.4f} |")

    nd = results["near_duplication"]
    lines += ["", "## Near-duplication between train and test", "",
              f"- {nd['frac_test_sharing_a_kmer_with_train']:.1%} of probed test rows share an "
              f"exact {nd['kmer_k']}-mer with train ({nd['probed']} probed).", "",
              "## Split policy", "",
              f"- {results['homology_clusters']['n_clusters']} homology clusters over "
              f"{results['train']['n']} train rows "
              f"(mean size {results['homology_clusters']['mean_cluster_size']}).", "",
              "| split | shared clusters between train/val | val rows sharing a k-mer with train |",
              "|---|---|---|"]
    for name, r in results["split_leakage"].items():
        lines.append(f"| {name} | {r['shared_homology_clusters']} | "
                     f"{r['val_rows_sharing_a_kmer_with_train']:.1%} |")

    md = "\n".join(lines) + "\n"
    (out_dir / f"leakage_{args.dataset}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[done] {out_dir / f'leakage_{args.dataset}.md'}")
    return results


if __name__ == "__main__":
    main()
