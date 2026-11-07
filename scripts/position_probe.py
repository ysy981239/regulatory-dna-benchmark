"""Does the model use ABSOLUTE POSITION, or only what is present somewhere?

Two independent measurements, chosen so that they can disagree with each other:

1. **circular-shift probe** - roll the *real* bases of each test sequence by a
   random offset (padding layout untouched).  Base composition is preserved
   exactly, every local motif survives except at the wrap point, and the only
   thing destroyed is *where* things sit inside the window.  A model that reads
   "TATA at position 100" must lose accuracy; a bag-of-k-mers model must not.

2. **per-6-mer positional divergence** (data only, no model) - for each 6-mer,
   compare its position histogram in positives with the one in negatives (total
   variation distance), against a null built from two random halves of the same
   class.  This says whether positional information *exists in the data at all*,
   which is the question that has to be answered before the model's answer means
   anything.

A model can be position-sensitive (measurement 1 drops) while the data carries
almost no positional signal (measurement 2 flat) - that combination means the
model is using position as a shortcut, not as biology.

Usage
-----
    python scripts/position_probe.py --all --dataset human_nontata_promoters
    python scripts/position_probe.py --run-dir drosophila_enhancers_stark__basset__group__s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import attributions as gattr  # noqa: E402
from src import data as gdata  # noqa: E402
from src import dna  # noqa: E402
from src import evaluate as gev  # noqa: E402
from src import train as gtrain  # noqa: E402
from scripts.calibrate import rebuild_split  # noqa: E402
from scripts.shuffle_probe import score  # noqa: E402


def circular_shift(ids: np.ndarray, seed: int = 0) -> np.ndarray:
    """Roll the real bases of each row; leave the padding where it is.

    Rolling the whole padded row would drag padding into the middle of the
    sequence, which changes the mask as well as the order.  Only the real part
    is rolled, so the *only* thing that changes is where each base sits.
    """
    rng = np.random.default_rng(seed)
    out = ids.copy()
    for i in range(out.shape[0]):
        real = np.flatnonzero(out[i] < 4)
        if real.size < 2:
            continue
        k = int(rng.integers(1, real.size))
        out[i, real] = np.roll(out[i, real], k)
    return out


def shift_probe(run_dir: Path, device: str, n_seqs: int, seed: int) -> dict:
    model, ckpt = gattr.load_model(run_dir, device=device)
    args = ckpt["args"]
    _, _, _, _, ids_te, y_te, _ = rebuild_split(args)
    if n_seqs and n_seqs < len(y_te):
        sel = np.random.default_rng(seed).choice(len(y_te), size=n_seqs, replace=False)
        ids_te, y_te = ids_te[sel], y_te[sel]

    p_real = score(model, ids_te, y_te, device)
    p_shift = score(model, circular_shift(ids_te, seed=seed), y_te, device)
    m_real = gev.binary_metrics(y_te, p_real, n_boot=200, seed=seed)
    m_shift = gev.binary_metrics(y_te, p_shift, n_boot=200, seed=seed)
    corr = float(np.corrcoef(p_real, p_shift)[0, 1])

    rec = {
        "run": run_dir.name, "dataset": args["dataset"], "model": args["model"],
        "split": args["split"], "n_seqs": int(len(y_te)),
        "real": {"auprc": m_real["auprc"], "auroc": m_real["auroc"]},
        "shifted": {"auprc": m_shift["auprc"], "auroc": m_shift["auroc"]},
        "retained_auprc_frac": float(m_shift["auprc"] / m_real["auprc"]) if m_real["auprc"] else float("nan"),
        "corr": corr,
    }
    print(f"[shift] {run_dir.name}")
    print(f"        real    AUPRC {m_real['auprc']:.4f}  AUROC {m_real['auroc']:.4f}")
    print(f"        shifted AUPRC {m_shift['auprc']:.4f}  AUROC {m_shift['auroc']:.4f}"
          f"  | retained {rec['retained_auprc_frac']:.1%}  | corr {corr:.3f}")
    return rec


def positional_divergence(dataset: str, max_kmers: int = 12) -> dict:
    ids, y, meta = gdata.load_split(dataset, "train")
    codes = dna.kmer_codes(ids, 6)
    k = int(codes.shape[1])
    valid = codes < 4096

    letters = np.array(list("ACGT"))
    names = ["".join(letters[(i // 4 ** (5 - d)) % 4] for d in range(6)) for i in range(4096)]
    chk = int(dna.kmer_codes(dna.seqs_to_ids(["ACGTAC"], max_len=6), 6)[0, 0])
    assert names[chk] == "ACGTAC", "k-mer decode order is wrong"

    def profile(rows: np.ndarray) -> np.ndarray:
        c = codes[rows]
        v = c < 4096
        pos = np.broadcast_to(np.arange(k), c.shape)[v].astype(np.int64)
        idx = pos * 4096 + c[v].astype(np.int64)
        return np.bincount(idx, minlength=k * 4096).reshape(k, 4096).astype(np.float64)

    half = np.random.default_rng(0).random(len(y)) < 0.5
    prof = {"pos": profile(y == 1), "neg": profile(y == 0),
            "A": profile(half), "B": profile(~half)}

    def norm(m):
        s = m.sum(axis=0, keepdims=True)
        return np.divide(m, np.maximum(s, 1), out=np.zeros_like(m), where=s > 0)

    def tv(a, b):
        return 0.5 * np.abs(norm(a) - norm(b)).sum(axis=0)

    d_cls, d_null = tv(prof["pos"], prof["neg"]), tv(prof["A"], prof["B"])
    keep = (prof["pos"].sum(0) >= 100) & (prof["neg"].sum(0) >= 100)
    thr = float(np.percentile(d_null[keep], 99))
    above = int((d_cls[keep] > thr).sum())

    print(f"[6-mer] {dataset}: {int(keep.sum())} / 4096 kept (>=100 per class)")
    for tag, d in (("class vs class", d_cls), ("null (half split)", d_null)):
        q = np.percentile(d[keep], [50, 90, 99, 100])
        print(f"        {tag:18s} median={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.4f} max={q[3]:.4f}")
    print(f"        null p99 = {thr:.4f}; 6-mers above it: {above} "
          f"({100 * above / int(keep.sum()):.1f}%)")

    top = []
    for i in np.argsort(-np.where(keep, d_cls, -1))[:max_kmers]:
        top.append({"kmer": names[i], "D": float(d_cls[i]),
                    "peak_pos": int(norm(prof["pos"][:, i]).argmax()) + 3,
                    "n_pos": int(prof["pos"][:, i].sum()),
                    "n_neg": int(prof["neg"][:, i].sum())})
    for t in top[:6]:
        print(f"        {t['kmer']}  D={t['D']:.4f}  peak_pos={t['peak_pos']}  "
              f"n_pos={t['n_pos']}  n_neg={t['n_neg']}")

    rec = {"dataset": dataset, "n_kmers_kept": int(keep.sum()),
           "class_median": float(np.median(d_cls[keep])),
           "null_median": float(np.median(d_null[keep])),
           "null_p99": thr, "n_above_null_p99": above,
           "frac_above_null_p99": above / int(keep.sum()), "top": top}
    for m in ("TATAAA", "TATATA", "CCAATC", "GGGCGG", "CACGTC"):
        i = names.index(m)
        pp = norm(prof["pos"][:, i])
        rec.setdefault("canonical", {})[m] = {
            "D": float(d_cls[i]), "peak_pos": int(pp.argmax()) + 3,
            "n_pos": int(prof["pos"][:, i].sum()), "n_neg": int(prof["neg"][:, i].sum())}
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--all", action="store_true", help="every CNN run under runs/")
    ap.add_argument("--dataset", default="human_nontata_promoters",
                    help="dataset for the data-level 6-mer measurement ('' to skip)")
    ap.add_argument("--n-seqs", type=int, default=0, help="subsample test rows (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "reports" / "position_probe.json"))
    a = ap.parse_args(argv)

    if a.all:
        dirs = sorted(d for d in (ROOT / "runs").iterdir()
                      if d.is_dir() and (d / "best.pt").exists())
    elif a.run_dir:
        dirs = [a.run_dir if a.run_dir.is_absolute() else (ROOT / "runs" / a.run_dir).resolve()]
    else:
        ap.error("pass --run-dir or --all")

    recs = [shift_probe(d, a.device, a.n_seqs, a.seed) for d in dirs]
    data_rec = positional_divergence(a.dataset) if a.dataset else None

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gev.dump_json({"seed": a.seed, "shift": recs, "kmers": data_rec,
                   "note": "shifted = circular roll of the real bases only: composition "
                           "preserved, absolute position destroyed"}, out)

    md = ["| run | dataset | model | test AUPRC (real) | test AUPRC (shifted) | retained | corr |",
          "|---|---|---|---|---|---|---|"]
    for r in recs:
        md.append(f"| {r['run']} | {r['dataset']} | {r['model']} | {r['real']['auprc']:.4f} | "
                  f"{r['shifted']['auprc']:.4f} | {r['retained_auprc_frac']:.1%} | {r['corr']:.3f} |")
    if data_rec:
        md += ["", f"6-mer positional divergence on `{data_rec['dataset']}`: class median "
                   f"{data_rec['class_median']:.4f} vs null median {data_rec['null_median']:.4f}; "
                   f"{data_rec['n_above_null_p99']} of {data_rec['n_kmers_kept']} k-mers exceed the "
                   f"null p99 ({data_rec['frac_above_null_p99']:.1%})."]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
