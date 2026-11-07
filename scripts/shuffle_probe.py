"""Does the model use base ORDER, or only base COMPOSITION?

Take a trained CNN, keep every test sequence's exact base counts, but destroy
the order by permuting its bases.  Then re-score the same model.

    real test AUPRC  -  shuffled test AUPRC  =  the part of the performance
                                               that genuinely needs sequence order

A model that only reads GC content is *invariant* by construction, so an
AUPRC drop of ~0 means "this network is a composition detector".  A large
drop means the performance really comes from local order (k-mers, motifs).

This is a probe of the *trained* model, not a retrain: it costs seconds and
cannot be confounded by optimisation.

Usage
-----
    python scripts/shuffle_probe.py --run-dir runs/promoters__basset__group__s0
    python scripts/shuffle_probe.py --all --n-seqs 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import attributions as gattr  # noqa: E402
from src import data as gdata  # noqa: E402
from src import evaluate as gev  # noqa: E402
from src import train as gtrain  # noqa: E402
from scripts.calibrate import rebuild_split  # noqa: E402


def shuffle_bases(ids: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute the real bases inside each row, leaving padding untouched.

    Base *counts* per sequence are preserved exactly; every longer-range
    statistic (dinucleotide, 5-mer, motif) is destroyed.
    """
    rng = np.random.default_rng(seed)
    out = ids.copy()
    for i in range(out.shape[0]):
        real = np.flatnonzero(out[i] < 4)  # 0..3 = ACGT, 255 = padding
        if real.size < 2:
            continue
        vals = out[i, real].copy()
        rng.shuffle(vals)
        out[i, real] = vals
    return out


def score(model, ids: np.ndarray, y: np.ndarray, device: str, batch_size: int = 256):
    loader = gtrain.build_loaders(ids, y, batch_size=batch_size, rc_augment=False,
                                  num_workers=0, seed=0, shuffle=False)
    _, p = gtrain.predict(model, loader, device)
    return p


def probe(run_dir: Path, device: str, n_seqs: int, seed: int) -> dict:
    model, ckpt = gattr.load_model(run_dir, device=device)
    args = ckpt["args"]
    ids_tr, y_tr, tr_idx, va_idx, ids_te, y_te, meta = rebuild_split(args)

    if n_seqs and n_seqs < len(y_te):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(y_te), size=n_seqs, replace=False)
        ids_te, y_te = ids_te[sel], y_te[sel]

    p_real = score(model, ids_te, y_te, device)
    p_shuf = score(model, shuffle_bases(ids_te, seed=seed), y_te, device)

    m_real = gev.binary_metrics(y_te, p_real, n_boot=200, seed=seed)
    m_shuf = gev.binary_metrics(y_te, p_shuf, n_boot=200, seed=seed)
    r = float(np.corrcoef(p_real, p_shuf)[0, 1]) if p_real.std() > 0 and p_shuf.std() > 0 else float("nan")

    rec = {
        "run": run_dir.name,
        "dataset": args["dataset"],
        "model": args["model"],
        "split": args["split"],
        "n_seqs": int(len(y_te)),
        "real": {"auprc": m_real["auprc"], "auroc": m_real["auroc"]},
        "shuffled": {"auprc": m_shuf["auprc"], "auroc": m_shuf["auroc"]},
        "drop_auprc": m_real["auprc"] - m_shuf["auprc"],
        "drop_auroc": m_real["auroc"] - m_shuf["auroc"],
        "retained_auprc_frac": (m_shuf["auprc"] / m_real["auprc"]) if m_real["auprc"] else float("nan"),
        "pred_corr": r,
    }
    print(f"[probe] {run_dir.name}")
    print(f"        real     AUPRC {m_real['auprc']:.4f}  AUROC {m_real['auroc']:.4f}")
    print(f"        shuffled AUPRC {m_shuf['auprc']:.4f}  AUROC {m_shuf['auroc']:.4f}")
    print(f"        drop     AUPRC {rec['drop_auprc']:+.4f}  AUROC {rec['drop_auroc']:+.4f}"
          f"  | retained {rec['retained_auprc_frac']:.1%}  | corr(p_real,p_shuf) {r:.3f}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--all", action="store_true", help="every CNN run under runs/")
    ap.add_argument("--n-seqs", type=int, default=0, help="subsample test rows (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "reports" / "shuffle_probe.json"))
    a = ap.parse_args(argv)

    if a.all:
        dirs = sorted(d for d in (ROOT / "runs").iterdir()
                      if d.is_dir() and (d / "best.pt").exists())
    elif a.run_dir:
        dirs = [a.run_dir if a.run_dir.is_absolute() else (ROOT / "runs" / a.run_dir).resolve()]
    else:
        ap.error("pass --run-dir or --all")

    recs = [probe(d, a.device, a.n_seqs, a.seed) for d in dirs]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gev.dump_json({"seed": a.seed, "records": recs,
                   "note": "shuffled = per-sequence base permutation: "
                           "composition preserved, order destroyed"}, out)
    print(f"[done] {out}")

    md = ["| run | dataset | test AUPRC (real) | test AUPRC (shuffled) | drop | retained |",
          "|---|---|---|---|---|---|"]
    for r in recs:
        md.append(f"| {r['run']} | {r['dataset']} | {r['real']['auprc']:.4f} | "
                  f"{r['shuffled']['auprc']:.4f} | {r['drop_auprc']:+.4f} | "
                  f"{r['retained_auprc_frac']:.1%} |")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
