"""Binary classification metrics that are honest about imbalance and uncertainty.

Why this module exists: on these benchmarks a single ``accuracy`` or ``auroc``
number hides almost everything.  Positive rates are often near 50% by
construction but the *useful* regime is high specificity, and calibration is
what a biologist downstream actually consumes.  So we always report:

* AUROC (ranking, prevalence-free)
* AUPRC (= average precision) with a bootstrap confidence interval
* Brier score and ECE (calibration)
* sensitivity/specificity/F1 at a chosen threshold, and the best-F1 threshold
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _safe(fn, y, p, default=float("nan"), needs_ranking: bool = False):
    """Call ``fn(y, p)`` but return ``default`` instead of a meaningless number.

    Two distinct failure modes matter here.  sklearn raises ``ValueError`` for
    some degenerate inputs.  Worse, for others it silently returns a plausible
    score: a constant predictor gets AUROC 0.5, which reads as "chance" when the
    honest answer is "undefined" (a model that ranks nothing has no ranking
    metric).  ``needs_ranking`` selects that stricter reading, so a constant
    score or a single-class target yields NaN and can never be quoted as a real
    result in a table.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if needs_ranking and (np.unique(y).size < 2 or np.unique(p).size < 2):
        return default
    try:
        v = float(fn(y, p))
    except ValueError:
        return default
    return v if np.isfinite(v) else default


def reliability_points(y: np.ndarray, p: np.ndarray, n_bins: int = 15):
    """Bin the predicted scores and return (mean confidence, observed frequency)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    conf, acc = [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf.append(float(p[m].mean()))
        acc.append(float(y[m].mean()))
    return conf, acc


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on the predicted score."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def best_f1_threshold(y, p) -> float:
    """Threshold that maximises F1 **on the data it is computed from**.

    This is an oracle: applied to the same split it was tuned on, it is an upper
    bound, not an honest test statistic.  Tune it on validation and pass it to
    ``binary_metrics(..., given_threshold=...)`` (see docs/DEVLOG.md, issue E3).
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(ps[int(np.argmax(f1))])


def binary_metrics(y, p, threshold: float = 0.5, n_boot: int = 200, seed: int = 0,
                   given_threshold: float | None = None) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    y = np.asarray(y, float)
    p = np.asarray(p, float)
    out = {
        "n": int(y.size),
        "pos_frac": round(float(y.mean()), 4),
        "auroc": round(_safe(roc_auc_score, y, p, needs_ranking=True), 4),
        "auprc": round(_safe(average_precision_score, y, p), 4),
        "brier": round(_safe(brier_score_loss, y, p), 4),
        "ece": round(ece(y, p), 4),
    }

    # bootstrap CI for AUPRC: the number that actually has variance on small sets
    if n_boot and out["n"] > 10:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, y.size, y.size)
            v = _safe(average_precision_score, y[idx], p[idx])
            if np.isfinite(v):
                vals.append(v)
        if vals:
            out["auprc_ci95"] = [round(float(np.percentile(vals, 2.5)), 4),
                                 round(float(np.percentile(vals, 97.5)), 4)]

    # best-F1 threshold on *this* split - kept for reproducibility, but it is an
    # oracle: reporting it as a test number is optimistic by construction
    out["best_f1_threshold"] = round(best_f1_threshold(y, p), 4)

    named = [("at_0.5", 0.5), ("at_best_f1", out["best_f1_threshold"])]
    if given_threshold is not None:
        out["given_threshold"] = round(float(given_threshold), 4)
        named.append(("at_given", float(given_threshold)))
    for name, thr in named:
        pred = (p >= thr).astype(int)
        out[f"precision_{name}"] = round(float(precision_score(y, pred, zero_division=0)), 4)
        out[f"recall_{name}"] = round(float(recall_score(y, pred, zero_division=0)), 4)
        out[f"f1_{name}"] = round(float(f1_score(y, pred, zero_division=0)), 4)
        out[f"specificity_{name}"] = round(float(recall_score(1 - y, 1 - pred, zero_division=0)), 4)
    out["best_f1"] = out["f1_at_best_f1"]
    return out


def save_reliability(y, p, path: Path, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 14)
    xs, ys = [], []
    for b in range(15):
        m = idx == b
        if m.sum() >= 5:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", lw=1.5)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title(title or "reliability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pr_curve(y, p, path: Path, label: str = "", baseline: float | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    ax.plot(rec, prec, lw=1.5, label=label or "model")
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="grey", lw=1, label=f"positive rate {baseline:.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("precision-recall")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def dump_json(obj, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
