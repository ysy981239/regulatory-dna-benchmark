"""Post-hoc temperature scaling, fitted on the validation split.

Why this script exists
----------------------
A CNN trained with plain cross-entropy on 20k sequences comes out *overconfident*:
it puts 0.99 on things that are right 70% of the time.  The reliability curve
looks terrible (ECE ~0.29 on drosophila) even though the ranking is fine.

Temperature scaling divides the logits by a single scalar T fitted on the
validation split (never on test) to minimise the negative log-likelihood.  It
cannot change AUROC or AUPRC -- it is a monotone transform -- so it is a clean
way to separate "the model ranks badly" from "the model reports the wrong
probabilities".

Usage
-----
    python scripts/calibrate.py --run-dir runs/human_nontata_promoters__basset__group__s0
    python scripts/calibrate.py --all          # every run under runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import attributions as gattr  # noqa: E402
from src import data as gdata  # noqa: E402
from src import evaluate as gev  # noqa: E402
from src import train as gtrain  # noqa: E402

LOGIT_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Minimise binary NLL over a single scalar T (LBFGS on log T)."""
    zt = torch.tensor(z, dtype=torch.float64)
    yt = torch.tensor(y.astype(np.float64), dtype=torch.float64)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(zt / torch.exp(log_t), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def rebuild_split(args: dict):
    ids_tr, y_tr, meta = gdata.load_split(args["dataset"], "train", max_len=args.get("max_len"))
    ids_te, y_te, _ = gdata.load_split(args["dataset"], "test", max_len=args.get("max_len"))
    k = args.get("kmer_k", 20)
    if args.get("split") == "group":
        clusters = gdata.homology_clusters(ids_tr, k=k)
        tr_idx, va_idx = gdata.grouped_split(y_tr, clusters, args.get("val_frac", 0.15),
                                             args.get("seed", 0))
    else:
        tr_idx, va_idx = gdata.stratified_split(y_tr, args.get("val_frac", 0.15),
                                                args.get("seed", 0))
    return ids_tr, y_tr, tr_idx, va_idx, ids_te, y_te, meta


def write_figure(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for p, label, color in ((p_raw, "as trained", "#c0392b"),
                            (p_cal, "temperature scaled", "#2c7fb8")):
        conf, acc = gev.reliability_points(y, p)
        axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
        axes[0].plot(conf, acc, "o-", ms=3, color=color, label=label)

    axes[0].set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title("reliability (test split)")
    axes[0].legend(fontsize=8)

    axes[1].hist(p_raw, bins=50, alpha=0.55, label="as trained", color="#c0392b")
    axes[1].hist(p_cal, bins=50, alpha=0.55, label="temperature scaled", color="#2c7fb8")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("predicted probability")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("score histogram")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibrate_run(run_dir: Path, device: str, n_boot: int = 200) -> dict:
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    model, ckpt = gattr.load_model(run_dir, device=device)
    ids_tr, y_tr, tr_idx, va_idx, ids_te, y_te, meta = rebuild_split(args)

    bs = args.get("batch_size", 128)
    va_loader = gtrain.build_loaders(ids_tr[va_idx], y_tr[va_idx], bs, False, 0,
                                     args.get("seed", 0), shuffle=False)
    te_loader = gtrain.build_loaders(ids_te, y_te, bs, False, 0,
                                     args.get("seed", 0), shuffle=False)

    y_va, p_va = gtrain.predict(model, va_loader, device)
    T = fit_temperature(logit(p_va), y_va)
    p_va_cal = sigmoid(logit(p_va) / T)

    y_te, p_te = gtrain.predict(model, te_loader, device)
    p_cal = sigmoid(logit(p_te) / T)

    # diagnostic only: the temperature that *would* be optimal on the test split.
    # It is not usable in practice (it peeks at the labels) but the gap between
    # T_val and T_test measures how much the two splits differ in difficulty.
    T_oracle = fit_temperature(logit(p_te), y_te)
    p_oracle = sigmoid(logit(p_te) / T_oracle)

    before = gev.binary_metrics(y_te, p_te, n_boot=n_boot)
    after = gev.binary_metrics(y_te, p_cal, n_boot=n_boot)
    oracle = gev.binary_metrics(y_te, p_oracle, n_boot=n_boot)
    nll_before = float(-np.mean(y_te * np.log(np.clip(p_te, 1e-12, 1))
                                + (1 - y_te) * np.log(np.clip(1 - p_te, 1e-12, 1))))
    nll_after = float(-np.mean(y_te * np.log(np.clip(p_cal, 1e-12, 1))
                               + (1 - y_te) * np.log(np.clip(1 - p_cal, 1e-12, 1))))

    def _nll(y_, p_):
        return float(-np.mean(y_ * np.log(np.clip(p_, 1e-12, 1))
                              + (1 - y_) * np.log(np.clip(1 - p_, 1e-12, 1))))

    out = {
        "run": run_dir.name,
        "dataset": args.get("dataset"),
        "model": args.get("model"),
        "split": args.get("split"),
        "best_epoch": ckpt.get("epoch"),
        "temperature": T,
        "temperature_effect": "T > 1 softens overconfident scores, T < 1 sharpens them",
        "val_ece_before": gev.ece(y_va, p_va),
        "val_ece_after": gev.ece(y_va, p_va_cal),
        "val_nll_before": _nll(y_va, p_va),
        "val_nll_after": _nll(y_va, p_va_cal),
        "test_nll_before": nll_before,
        "test_nll_after": nll_after,
        "ece_before": before["ece"],
        "ece_after": after["ece"],
        "brier_before": before["brier"],
        "brier_after": after["brier"],
        "auprc_before": before["auprc"],
        "auprc_after": after["auprc"],
        "auroc_before": before["auroc"],
        "auroc_after": after["auroc"],
        "auprc_unchanged": bool(abs(before["auprc"] - after["auprc"]) < 1e-9),
        "best_f1_threshold_before": before["best_f1_threshold"],
        "best_f1_threshold_after": after["best_f1_threshold"],
        "oracle_temperature_test": T_oracle,
        "oracle_ece_test": oracle["ece"],
        "oracle_brier_test": oracle["brier"],
        "metrics_before": before,
        "metrics_after": after,
    }
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_figure(y_te, p_te, p_cal, run_dir / "calibration.png")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--all", action="store_true", help="every run under runs/ with best.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-boot", type=int, default=200)
    args = p.parse_args(argv)

    if args.all:
        runs = sorted(d for d in (ROOT / "runs").iterdir() if (d / "best.pt").exists())
    elif args.run_dir:
        runs = [Path(args.run_dir)]
        if not runs[0].is_absolute():
            runs = [ROOT / runs[0]]
    else:
        p.error("give --run-dir or --all")
        return 2

    rows = []
    for d in runs:
        out = calibrate_run(d, args.device, args.n_boot)
        rows.append(out)
        print(f"[{out['run']}] T={out['temperature']:.3f}  "
              f"val ECE {out['val_ece_before']:.4f} -> {out['val_ece_after']:.4f}  "
              f"test ECE {out['ece_before']:.4f} -> {out['ece_after']:.4f}  "
              f"(oracle T on test {out['oracle_temperature_test']:.3f} -> "
              f"{out['oracle_ece_test']:.4f})  AUPRC {out['auprc_before']:.4f}")

    out_path = ROOT / "reports" / "calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[written] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
