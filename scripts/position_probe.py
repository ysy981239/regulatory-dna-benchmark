"""Temporary probe: does any of the promoters models use ABSOLUTE POSITION?

Two independent measurements:
  1. circular-shift probe  - rolls each test sequence by a random offset.  Composition
     is preserved exactly and every local motif is preserved except at the wrap point;
     the only thing destroyed is *where* things sit in the window.  A model that reads
     "TATA at position 100" must lose accuracy; a bag-of-kmers model must not.
  2. per-6-mer positional divergence - purely data-level.  For each 6-mer compare its
     position histogram in positives vs negatives (total variation distance), against a
     null built from two random halves of the same class.
"""
import numpy as np
import torch

from src import attributions as gattr
from src import data as gdata
from src import dna
from src import evaluate as gev
from scripts.shuffle_probe import score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = [
    "human_nontata_promoters__basset__group__s0",
    "human_nontata_promoters__dilated__group__s0",
    "human_nontata_promoters__transformer__group__s0",
]
ROOT = __import__("pathlib").Path(__file__).resolve().parent


def circular_shift(ids, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty_like(ids)
    n = ids.shape[1]
    for i in range(ids.shape[0]):
        out[i] = np.roll(ids[i], int(rng.integers(1, n)))
    return out


print("=" * 70)
print("1) CIRCULAR-SHIFT PROBE (absolute position only)")
print("=" * 70)
rows = []
for name in RUNS:
    run_dir = ROOT / "runs" / name
    model, ckpt = gattr.load_model(run_dir, DEV)
    args = ckpt["args"]
    ids_te, y_te, _ = gdata.load_split(args["dataset"], "test", max_len=args["max_len"])
    p_real = score(model, ids_te, y_te, DEV)
    p_sh = score(model, circular_shift(ids_te, seed=0), y_te, DEV)
    m_r = gev.binary_metrics(y_te, p_real, n_boot=200, seed=0)
    m_s = gev.binary_metrics(y_te, p_sh, n_boot=200, seed=0)
    corr = float(np.corrcoef(p_real, p_sh)[0, 1])
    rows.append((args["model"], m_r["auprc"], m_s["auprc"], m_s["auprc"] / m_r["auprc"],
                 m_r["auroc"], m_s["auroc"], corr))
    print(f"{args['model']:12s} real={m_r['auprc']:.4f} shifted={m_s['auprc']:.4f} "
          f"retained={100*m_s['auprc']/m_r['auprc']:5.1f}%  "
          f"auroc {m_r['auroc']:.4f}->{m_s['auroc']:.4f}  corr={corr:.3f}")

print()
print("=" * 70)
print("2) POSITIONAL DIVERGENCE OF 6-MERS (data only)")
print("=" * 70)

ids_tr, y_tr, meta = gdata.load_split("human_nontata_promoters", "train")
ids_tr = ids_tr[:, :251]
codes = dna.kmer_codes(ids_tr, 6)
P = codes.shape[1]
valid = codes < 4096

letters = np.array(list("ACGT"))
names = ["".join(letters[(i // 4 ** (5 - d)) % 4] for d in range(6)) for i in range(4096)]
probe_seq = dna.seqs_to_ids(["ACGTAC"], max_len=6)
chk = dna.kmer_codes(probe_seq, 6)[0, 0]
assert names[int(chk)] == "ACGTAC", (names[int(chk)], "decode order is wrong")
print("decode convention verified: ACGTAC ->", names[int(chk)])


def profile(rows_mask):
    c = codes[rows_mask]
    v = c < 4096
    pos = np.broadcast_to(np.arange(P), c.shape)[v].astype(np.int64)
    idx = pos * 4096 + c[v].astype(np.int64)
    return np.bincount(idx, minlength=P * 4096).reshape(P, 4096).astype(np.float64)


rng = np.random.default_rng(0)
half = rng.random(len(y_tr)) < 0.5
prof = {
    "pos": profile(y_tr == 1),
    "neg": profile(y_tr == 0),
    "A": profile(half),
    "B": profile(~half),
}


def norm(m):
    s = m.sum(axis=0, keepdims=True)
    return np.divide(m, np.maximum(s, 1), out=np.zeros_like(m), where=s > 0)


def tv(a, b):
    return 0.5 * np.abs(norm(a) - norm(b)).sum(axis=0)


d_cls = tv(prof["pos"], prof["neg"])
d_null = tv(prof["A"], prof["B"])
keep = (prof["pos"].sum(0) >= 100) & (prof["neg"].sum(0) >= 100)
print(f"6-mers kept (>=100 in each class): {int(keep.sum())} / 4096")
for tag, d in (("class vs class", d_cls), ("null (half split)", d_null)):
    q = np.percentile(d[keep], [50, 90, 99, 100])
    print(f"  {tag:18s} median={q[0]:.4f}  p90={q[1]:.4f}  p99={q[2]:.4f}  max={q[3]:.4f}")

thr = np.percentile(d_null[keep], 99)
print(f"  null p99 = {thr:.4f};  6-mers above it: {int((d_cls[keep] > thr).sum())} "
      f"({100*(d_cls[keep] > thr).mean():.1f}%)")

o = np.argsort(-np.where(keep, d_cls, -1))[:12]
print()
print("  top 12 by positional divergence (peak position, 0..245):")
for i in o:
    pp = norm(prof["pos"][:, i])
    pn = norm(prof["neg"][:, i])
    print(f"    {names[i]}  D={d_cls[i]:.4f}  peak_pos={int(pp.argmax())+3}  "
          f"peak_neg={int(pn.argmax())+3}  n_pos={int(prof['pos'][:, i].sum())}  "
          f"n_neg={int(prof['neg'][:, i].sum())}")

print()
print("  canonical core-promoter motifs:")
for m in ["TATAAA", "TATAAT", "TATAAG", "TATATA", "TCAGTT", "CCAATC", "GGGCGG", "CACGTC"]:
    i = names.index(m)
    pp = norm(prof["pos"][:, i])
    pn = norm(prof["neg"][:, i])
    print(f"    {m}  D={d_cls[i]:.4f}  null_p99={thr:.4f}  peak_pos={int(pp.argmax())+3}  "
          f"peak_neg={int(pn.argmax())+3}  n_pos={int(prof['pos'][:, i].sum())}  "
          f"n_neg={int(prof['neg'][:, i].sum())}")
