"""Collect every run under ``runs/`` into one comparison table.

This is the artifact the writeup is built on: one row per (dataset, model, split,
seed), with the split policy as a first-class column, because the honest headline
of the project is usually "the same model loses N points when you stop leaking".

Usage
-----
    python scripts/make_table.py --runs runs --out reports/summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="reports/summary.md")
    p.add_argument("--sort", default="auprc")
    args = p.parse_args(argv)

    rows = []
    for f in sorted(Path(args.runs).glob("*/metrics.json")):
        try:
            rows.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if not rows:
        print(f"no metrics.json under {args.runs!r} - train something first")
        return

    rows.sort(key=lambda r: -r["test"].get(args.sort, 0))
    header = ("| dataset | model | params | split | rc aug | test AUPRC | 95% CI | "
              "test AUROC | ECE | best F1 |")
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["test"]
        ci = t.get("auprc_ci95")
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['params']:,} | {r['split']} | "
            f"{'yes' if r.get('rc_augment') else 'no'} | {t['auprc']:.4f} | {ci_s} | "
            f"{t['auroc']:.4f} | {t['ece']:.4f} | {t.get('best_f1', float('nan')):.4f} |"
        )
    md = "\n".join(lines) + "\n"

    # the number every deep model has to beat, kept in the same document so it
    # cannot quietly disappear from the writeup
    audits = []
    for f in sorted(Path("reports").glob("leakage_*.json")):
        try:
            audits.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if audits:
        md += ("\n## Reference baselines (no learned sequence grammar, official test split)\n\n"
               "A CNN that does not clear the 5-mer linear model is not evidence of "
               "regulatory grammar, only of a working optimiser.\n\n"
               "| dataset | baseline | #features | test AUROC | test AUPRC |\n"
               "|---|---|---|---|---|\n")
        for a in audits:
            for b in a.get("baselines", []):
                md += (f"| {a['dataset']} | {b['baseline']} | {b['n_features']} | "
                       f"{b['auroc']:.4f} | {b['auprc']:.4f} |\n")

    # the split-policy gap, if both policies exist for a (dataset, model)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["dataset"], r["model"]), {})[r["split"]] = r
    gaps = []
    for (ds, model), d in by_key.items():
        if "random" in d and "group" in d:
            g, rr = d["group"], d["random"]
            gaps.append(dict(
                dataset=ds, model=model,
                val_group=g.get("best_val_auprc", float("nan")),
                val_random=rr.get("best_val_auprc", float("nan")),
                test_group=g["test"]["auprc"], test_random=rr["test"]["auprc"],
                ece_group=g["test"].get("ece", float("nan")),
                ece_random=rr["test"].get("ece", float("nan")),
                brier_group=g["test"].get("brier", float("nan")),
                brier_random=rr["test"].get("brier", float("nan")),
            ))
    if gaps:
        md += ("\n## Optimism from a random row split\n\n"
               "Two numbers move for different reasons.  The **validation** gap is the "
               "leakage you would have believed; the **test** gap is only visible if the "
               "benchmark's own test split is homology-free.  When the test gap is ~0 "
               "while calibration falls apart, the official split is leaking too.\n\n"
               "| dataset | model | val AUPRC grouped | val AUPRC random | val gap "
               "| test AUPRC grouped | test AUPRC random | test gap "
               "| ECE grouped | ECE random |\n"
               "|---|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(gaps, key=lambda x: -(x["val_random"] - x["val_group"])):
            md += (f"| {r['dataset']} | {r['model']} | {r['val_group']:.4f} | "
                   f"{r['val_random']:.4f} | {r['val_random'] - r['val_group']:+.4f} | "
                   f"{r['test_group']:.4f} | {r['test_random']:.4f} | "
                   f"{r['test_random'] - r['test_group']:+.4f} | "
                   f"{r['ece_group']:.4f} | {r['ece_random']:.4f} |\n")

    # calibration: temperature scaling fitted on the grouped validation split
    cals = []
    for f in sorted(Path(args.runs).glob("*/calibration.json")):
        try:
            cals.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if cals:
        md += ("\n## Calibration (temperature fitted on the validation split)\n\n"
               "Temperature scaling is a monotone transform, so AUPRC/AUROC **cannot** "
               "move.  If NLL improves while ECE does not, the model's mistakes are not "
               "uncertain mistakes - they are confident errors, and no global rescaling "
               "fixes them.  The oracle column fits T on test labels and is a diagnostic "
               "only: a large gap between it and T_val says the two splits differ in "
               "difficulty.\n\n"
               "| run | T (val) | val ECE | val ECE (T) | test ECE | test ECE (T) "
               "| test NLL | test NLL (T) | T oracle (test) |\n"
               "|---|---|---|---|---|---|---|---|---|\n")
        for c in sorted(cals, key=lambda x: x["run"]):
            md += (f"| {c['run']} | {c['temperature']:.3f} | "
                   f"{c['val_ece_before']:.4f} | {c['val_ece_after']:.4f} | "
                   f"{c['ece_before']:.4f} | {c['ece_after']:.4f} | "
                   f"{c['test_nll_before']:.4f} | {c['test_nll_after']:.4f} | "
                   f"{c['oracle_temperature_test']:.3f} |\n")

    # order vs composition: permute each test sequence's bases, keep its counts
    probe_f = Path("reports/shuffle_probe.json")
    if probe_f.exists():
        try:
            probe = json.loads(probe_f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            probe = None
        if probe and probe.get("records"):
            md += ("\n## Order or composition? (per-sequence base permutation of the test set)\n\n"
                   "Every test sequence keeps its exact base counts but loses its order.  A pure "
                   "composition detector is **invariant** to this, so whatever AUPRC survives was "
                   "never about sequence order.  The GC-only baseline below is invariant *by "
                   "construction*: it scores 0.8394 on promoters and 0.4722 on drosophila either "
                   "way, which is what makes it the yardstick for the shuffled column.\n\n"
                   "| run | test AUPRC (real) | test AUPRC (shuffled) | drop | retained | "
                   "corr(p_real, p_shuffled) |\n"
                   "|---|---|---|---|---|---|\n")
            for r in sorted(probe["records"], key=lambda x: -x["retained_auprc_frac"]):
                md += (f"| {r['run']} | {r['real']['auprc']:.4f} | "
                       f"{r['shuffled']['auprc']:.4f} | {r['drop_auprc']:+.4f} | "
                       f"{r['retained_auprc_frac']:.1%} | {r['pred_corr']:.3f} |\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
