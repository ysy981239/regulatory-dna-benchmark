"""Run the "no learned grammar" baselines through the *same* split as the CNNs.

`scripts/leakage_check.py` fits these on the full official train set, which is
the right number for an audit but the wrong number for a head-to-head: a CNN
trained on a homology-disjoint subset would be compared against a baseline that
got to memorise the near-duplicates.  This script rebuilds the split with the
same function, same k and same seed as ``src.train``, then fits:

    linear-gc      GC content only            (1 feature)
    linear-2mer    dinucleotide frequencies   (16)
    linear-5mer    5-mer frequencies          (1024)

and writes a ``metrics.json`` with the same schema as a CNN run, so
``scripts/make_table.py`` puts them in one table.  If your CNN does not beat
``linear-5mer`` on the same rows, the writeup has to say so.

Usage
-----
    python scripts/baseline_runner.py --dataset human_nontata_promoters --split group
    python scripts/baseline_runner.py --dataset human_nontata_promoters --split random --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as gdata  # noqa: E402
from src import dna  # noqa: E402
from src import evaluate as ev  # noqa: E402


def fit_predict(X_tr, y_tr, X_va, X_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs"))
    clf.fit(X_tr, y_tr)
    return (clf.predict_proba(X_tr)[:, 1],
            clf.predict_proba(X_va)[:, 1],
            clf.predict_proba(X_te)[:, 1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="human_nontata_promoters")
    p.add_argument("--split", choices=["group", "random"], default="group")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--kmer-k", type=int, default=5, help="k for the count features")
    p.add_argument("--cluster-k", type=int, default=20,
                   help="k used by homology clustering; must match src/train.py")
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args(argv)

    ids_tr, y_tr, meta = gdata.load_split(args.dataset, "train", max_len=args.max_len)
    ids_te, y_te, _ = gdata.load_split(args.dataset, "test", max_len=args.max_len)

    if args.split == "group":
        clusters = gdata.homology_clusters(ids_tr, k=args.cluster_k)
        tr_idx, va_idx = gdata.grouped_split(y_tr, clusters, args.val_frac, args.seed)
        note = f"grouped by shared {args.cluster_k}-mer ({len(np.unique(clusters))} clusters)"
    else:
        clusters = None
        tr_idx, va_idx = gdata.stratified_split(y_tr, args.val_frac, args.seed)
        note = "random rows (leaky by construction - this is the control)"

    feats = {
        "linear-gc": (dna.gc_fraction(ids_tr).reshape(-1, 1),
                      dna.gc_fraction(ids_te).reshape(-1, 1)),
        "linear-2mer": (dna.dinucleotide_frequencies(ids_tr),
                        dna.dinucleotide_frequencies(ids_te)),
        f"linear-{args.kmer_k}mer": (dna.kmer_frequencies(ids_tr, args.kmer_k),
                                     dna.kmer_frequencies(ids_te, args.kmer_k)),
    }

    rows = []
    for name, (X_tr, X_te) in feats.items():
        t0 = time.time()
        p_tr, p_va, p_te = fit_predict(X_tr[tr_idx], y_tr[tr_idx],
                                       X_tr[va_idx], X_te)
        m_te = ev.binary_metrics(y_te, p_te, n_boot=200, seed=args.seed)
        m_va = ev.binary_metrics(y_tr[va_idx], p_va, n_boot=0)
        run_dir = Path(args.out_dir) / f"{args.dataset}__{name}__{args.split}__s{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset, "model": name, "params": int(X_tr.shape[1]),
            "receptive_field": 0, "split": args.split, "split_detail": note,
            "best_epoch": -1, "best_val_auprc": m_va["auprc"],
            "test": m_te, "wall_clock_s": round(time.time() - t0, 1),
            "rc_augment": False, "ema_decay": 0.0, "seed": args.seed,
            "val": m_va,
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append((name, X_tr.shape[1], m_va["auprc"], m_te["auroc"], m_te["auprc"],
                     m_te["ece"]))
        print(f"[baseline] {name:16s} val AUPRC {m_va['auprc']:.4f} | "
              f"test AUROC {m_te['auroc']:.4f} AUPRC {m_te['auprc']:.4f} "
              f"ECE {m_te['ece']:.4f}")

    print(f"\n[baseline] {len(rows)} baselines written under {args.out_dir} "
          f"({args.dataset}, {args.split} split)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
