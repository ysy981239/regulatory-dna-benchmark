"""Regenerate docs/figures/audit.png -- the three-panel summary of the audit.

Panel A  what the padding fix changed (Drosophila Basset, official test)
Panel B  the "same family" threshold sweep: why k = 20 is fragile
Panel C  how much leakage survives once low-complexity k-mers are removed

Panels B and C read the committed JSON reports.  Panel A cannot: the pre-fix
`best.pt` files live in `runs_legacy/`, which is gitignored and was produced on
one laptop.  Those numbers are pinned here from `runs_legacy/*/metrics.json`
(commit 674b6e1 retrained the two post-fix runs), so the figure is reproducible
without shipping 20 MB of checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "docs" / "figures" / "audit.png"

# --- panel A: before/after the padding fix, Drosophila Basset, official test ---
METRICS = ["AUPRC", "ECE", "Brier"]
BEFORE = {  # runs_legacy/, buggy pooling
    "AUPRC": (0.8317, 0.8048),
    "ECE": (0.2946, 0.2956),
    "Brier": (0.2703, 0.2746),
}
AFTER = {  # runs/, masked pooling
    "AUPRC": (0.8255, 0.8472),
    "ECE": (0.1908, 0.1524),
    "Brier": (0.1974, 0.1750),
}

INK = "#2f3b52"
MUTED = "#8b95a7"
ACCENT = "#c1553b"
GREEN = "#3f7d5a"


def panel_a(ax):
    x = np.arange(len(METRICS))
    w = 0.2
    series = [
        ("group, before", [BEFORE[m][0] for m in METRICS], MUTED, 0.55),
        ("group, after", [AFTER[m][0] for m in METRICS], INK, 1.0),
        ("random, before", [BEFORE[m][1] for m in METRICS], "#e3b0a0", 0.55),
        ("random, after", [AFTER[m][1] for m in METRICS], ACCENT, 1.0),
    ]
    for i, (label, vals, colour, alpha) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=colour, alpha=alpha, linewidth=0)
    ax.axhline(0.25, color=GREEN, lw=1, ls="--")
    ax.text(len(METRICS) - 0.45, 0.256, "constant 0.5 predictor (Brier)", color=GREEN,
            fontsize=7, ha="right")
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0, 1.0)
    ax.set_title("A. Pooling over padding\ndrosophila Basset, official test", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    ax.tick_params(labelsize=7)


def panel_b(ax):
    data = json.loads((REPORTS / "cluster_sensitivity_drosophila_enhancers_stark.json").read_text())
    ks = [r["k"] for r in data["sweep"]]
    frac = [100 * r["largest_frac"] for r in data["sweep"]]
    val = [r["val_rows"] for r in data["sweep"]]
    ax.plot(ks, frac, "o-", color=ACCENT, lw=1.6, ms=4, label="largest component (%)")
    ax.set_xlabel("k (shared exact k-mer)", fontsize=7.5)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7)
    ax.set_ylabel("largest component (% of train)", fontsize=7.5, color=ACCENT)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.plot(ks, val, "s--", color=INK, lw=1.3, ms=3.5, label="validation rows")
    ax2.set_ylabel("validation rows", fontsize=7.5, color=INK)
    ax2.tick_params(labelsize=7)
    ax2.set_ylim(0, 5600)
    ax.axvline(20, color=MUTED, lw=0.8, ls=":")
    ax.annotate("k = 20 (chosen)", xy=(20, 72), xytext=(22.4, 88), fontsize=6.5,
                color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_title("B. The split only exists for k >= 20", fontsize=9, loc="left")


def panel_c(ax):
    for name, label, colour, marker in [
        ("drosophila_enhancers_stark", "Drosophila enhancers", ACCENT, "o"),
        ("human_nontata_promoters", "human promoters", INK, "s"),
    ]:
        data = json.loads((REPORTS / f"complexity_probe_{name}.json").read_text())
        xs = [r["min_entropy_bits"] for r in data["leakage"]]
        ys = [100 * r["frac_test_sharing"] for r in data["leakage"]]
        ax.plot(xs, ys, marker=marker, lw=1.6, ms=4, color=colour, label=label)
    ax.set_xlabel("minimum base-composition entropy (bits)", fontsize=7.5)
    ax.set_ylabel("test rows sharing a 20-mer (%)", fontsize=7.5)
    ax.set_ylim(30, 85)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("C. How much leakage is low-complexity repeat", fontsize=9, loc="left")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=200)
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#b8bfcc")
        ax.spines["bottom"].set_color("#b8bfcc")
    fig.suptitle(
        "Sixteen failures that raised no exception — and the comparisons that exposed them",
        fontsize=11, x=0.008, ha="left", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"[done] {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
