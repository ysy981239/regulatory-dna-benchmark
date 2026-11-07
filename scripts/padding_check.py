"""How much of the input is padding, and does it leak into the prediction?

Variable-length datasets are padded to a rectangle.  If the network pools over
the whole tensor, the padded positions are not "silent" - after BatchNorm the
all-zero one-hot column of padding comes out as a per-channel constant, and a
plain ``mean`` then divides by the *padded* length.  The same sequence then gets
a different score depending on how much padding follows it.

``src/models.py`` now masks the padding out of both the pooling and the
attention (see docs/DEVLOG.md, issue P1).  This script is the audit that shows
the problem was real and that the fix worked:

1. padding fraction and the real-length distribution of the test split;
2. Spearman correlation between real length and the prediction - if padding
   dilutes, longer (less padded) sequences should be scored more extremely;
3. AUPRC per real-length quintile - a model that uses the last base would lose
   accuracy on the short sequences;
4. AUPRC after truncating every sequence to its first N real bases - if the
   score barely moves, the tail contributes nothing.

Usage
-----
    python scripts/padding_check.py --run-dir drosophila_enhancers_stark__basset__group__s0
    python scripts/padding_check.py --all --trunc 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import attributions as gattr  # noqa: E402
from src import evaluate as gev  # noqa: E402
from src import train as gtrain  # noqa: E402
from scripts.calibrate import rebuild_split  # noqa: E402
from scripts.shuffle_probe import score  # noqa: E402


def real_lengths(ids: np.ndarray) -> np.ndarray:
    """Number of real bases per row (anything < 4; 255 marks padding)."""
    return (ids < 4).sum(axis=1)


def truncate_to(ids: np.ndarray, n: int) -> np.ndarray:
    """Keep the first ``n`` *real* bases of each row and pad the rest."""
    out = np.full_like(ids, 255)
    for i in range(ids.shape[0]):
        real = np.flatnonzero(ids[i] < 4)[:n]
        if real.size:
            out[i, : real.size] = ids[i, real]
    return out


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    m = gev.binary_metrics(y, p, n_boot=0)
    return float(m["auprc"])


def check(run_dir: Path, device: str, trunc: int) -> dict:
    model, ckpt = gattr.load_model(run_dir, device=device)
    args = ckpt["args"]
    _, _, _, _, ids_te, y_te, meta = rebuild_split(args)

    L = ids_te.shape[1]
    lens = real_lengths(ids_te)
    pad_frac = float((ids_te == 255).mean())
    p_full = score(model, ids_te, y_te, device)

    rec = {
        "run": run_dir.name,
        "dataset": args["dataset"],
        "model": args["model"],
        "split": args["split"],
        "tensor_len": int(L),
        "padding_fraction": pad_frac,
        "real_len": {
            "min": int(lens.min()), "q25": float(np.quantile(lens, 0.25)),
            "median": float(np.median(lens)), "mean": float(lens.mean()),
            "q75": float(np.quantile(lens, 0.75)), "max": int(lens.max()),
        },
        "auprc_full": _auprc(y_te, p_full),
        "spearman_len_vs_pred": float(spearmanr(lens, p_full).statistic),
        "spearman_len_vs_abspred": float(spearmanr(lens, np.abs(p_full)).statistic),
        "by_quintile": [],
    }

    order = np.argsort(lens)
    for qi, chunk in enumerate(np.array_split(order, 5)):
        if chunk.size < 10:
            continue
        rec["by_quintile"].append({
            "quintile": qi + 1,
            "n": int(chunk.size),
            "mean_real_len": float(lens[chunk].mean()),
            "pos_frac": float(y_te[chunk].mean()),
            "auprc": _auprc(y_te[chunk], p_full[chunk]),
        })

    if trunc and trunc < int(lens.max()):
        p_tr = score(model, truncate_to(ids_te, trunc), y_te, device)
        rec["trunc"] = {
            "keep": int(trunc),
            "auprc": _auprc(y_te, p_tr),
            "delta_auprc": _auprc(y_te, p_tr) - rec["auprc_full"],
            "corr_with_full": float(np.corrcoef(p_full, p_tr)[0, 1]),
        }

    print(f"[pad] {run_dir.name}")
    print(f"      tensor {L} bp, padding {pad_frac:.1%}, real length "
          f"{lens.min()}..{lens.max()} (median {np.median(lens):.0f})")
    print(f"      AUPRC full {rec['auprc_full']:.4f}")
    print(f"      spearman(real length, prediction) = {rec['spearman_len_vs_pred']:+.3f} "
          f"(|prediction| {rec['spearman_len_vs_abspred']:+.3f})")
    print("      AUPRC by real-length quintile: " + "  ".join(
        f"{q['mean_real_len']:.0f}bp:{q['auprc']:.3f}" for q in rec["by_quintile"]))
    if "trunc" in rec:
        print(f"      truncated to {trunc} bp: AUPRC {rec['trunc']['auprc']:.4f} "
              f"({rec['trunc']['delta_auprc']:+.4f}), corr with full "
              f"{rec['trunc']['corr_with_full']:.3f}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--all", action="store_true", help="every CNN run under runs/")
    ap.add_argument("--trunc", type=int, default=500, help="keep only the first N real bases")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "reports" / "padding_check.json"))
    a = ap.parse_args(argv)

    if a.all:
        dirs = sorted(d for d in (ROOT / "runs").iterdir()
                      if d.is_dir() and (d / "best.pt").exists())
    elif a.run_dir:
        dirs = [a.run_dir if a.run_dir.is_absolute() else (ROOT / "runs" / a.run_dir).resolve()]
    else:
        ap.error("pass --run-dir or --all")

    recs = [check(d, a.device, a.trunc) for d in dirs]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gev.dump_json({"records": recs,
                   "note": "padding is excluded from pooling and attention since "
                           "the P1 fix; a large spearman(length, prediction) would "
                           "mean padding still dilutes the score"}, out)

    md = ["| run | tensor | padding | real length (median) | AUPRC full | "
          "spearman(len, pred) | AUPRC @ trunc | corr(full, trunc) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        md.append(
            f"| {r['run']} | {r['tensor_len']} | {r['padding_fraction']:.1%} | "
            f"{r['real_len']['median']:.0f} | {r['auprc_full']:.4f} | "
            f"{r['spearman_len_vs_pred']:+.3f} | "
            + (f"{r['trunc']['auprc']:.4f} | {r['trunc']['corr_with_full']:.3f} |"
               if "trunc" in r else "- | - |"))
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] {out}")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
