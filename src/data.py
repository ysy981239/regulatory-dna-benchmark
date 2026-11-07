"""Dataset access + split strategies for the curated Genomic Benchmarks corpus.

Data source
-----------
``genomic-benchmarks/genomic-benchmarks`` on the Hugging Face Hub:
``v1/<dataset>/<dataset>_<split>.csv.gz`` with columns ``id,row_index,seq,label``.
The curated release fixes row order and gives every row a permanent id, which is
what makes a number you report traceable.  See the dataset card for the QC
findings - single-feature separability and train/test leakage are documented
properties of these benchmarks, not hypothetical risks.

Network
-------
The default endpoint is the ``hf-mirror.com`` mirror because huggingface.co is
often unreachable from mainland China.  Override with ``GENOMIC_BENCHMARKS_ENDPOINT``.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from . import dna

REPO = "genomic-benchmarks/genomic-benchmarks"
DEFAULT_ENDPOINT = os.environ.get("GENOMIC_BENCHMARKS_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASETS = {
    "drosophila_enhancers_stark": dict(
        task="enhancer vs non-enhancer", species="Drosophila melanogaster",
        n_train=5184, n_test=1730, difficulty="easy",
        note="smallest benchmark; use it to debug the pipeline in minutes",
    ),
    "human_nontata_promoters": dict(
        task="promoter vs non-promoter", species="Homo sapiens",
        n_train=27097, n_test=9034, difficulty="medium",
        note="recommended first real target: plenty of signal, still fast on 6 GB",
    ),
    "human_enhancers_cohn": dict(
        task="enhancer vs non-enhancer", species="Homo sapiens",
        n_train=20843, n_test=6948, difficulty="medium",
        note="Cohn lab enhancers; different negative-sampling regime than ensembl",
    ),
    "human_enhancers_ensembl": dict(
        task="enhancer vs non-enhancer", species="Homo sapiens",
        n_train=123872, n_test=30970, difficulty="hard",
        note="large; good for testing whether your findings survive scale",
    ),
    "human_ocr_ensembl": dict(
        task="open chromatin region vs not", species="Homo sapiens",
        n_train=139804, n_test=34952, difficulty="hard",
        note="closest to the DNase/ATAC accessibility task behind DeepSEA",
    ),
    "demo_coding_vs_intergenomic_seqs": dict(
        task="coding vs intergenic", species="Homo sapiens",
        n_train=75000, n_test=25000, difficulty="easy",
        note="composition-dominated; a GC-only baseline already scores high",
    ),
    "demo_human_or_worm": dict(
        task="human vs worm sequence", species="Homo sapiens + C. elegans",
        n_train=75000, n_test=25000, difficulty="easy",
        note="species classification: trivially separable, useful only as a sanity check",
    ),
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]


def split_url(dataset: str, split: str, endpoint: str | None = None) -> str:
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    return f"{endpoint}/datasets/{REPO}/resolve/main/v1/{dataset}/{dataset}_{split}.csv.gz"


def download_split(dataset: str, split: str, root: Path | None = None,
                   endpoint: str | None = None, force: bool = False) -> Path:
    root = Path(root or DATA_DIR)
    dest = root / "raw" / dataset / f"{dataset}_{split}.csv.gz"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = split_url(dataset, split, endpoint)
    tmp = dest.with_suffix(".gz.part")
    print(f"[data] downloading {dataset}/{split} <- {url}")
    with urllib.request.urlopen(url, timeout=180) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    print(f"[data] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _read_csv_gz(path: Path):
    import pandas as pd

    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, usecols=["id", "seq", "label"])
    return df


def load_split(dataset: str, split: str, root: Path | None = None,
               max_len: int | None = None, endpoint: str | None = None,
               use_cache: bool = True):
    """Return ``(ids (N,L) uint8, y (N,) float32, meta dict)``.

    Caches the encoded arrays as ``.npz`` so repeated runs do not re-parse 100 MB
    of CSV.  The cache key includes the file size and ``max_len``, so a truncated
    download can never be mistaken for a cached result.
    """
    root = Path(root or DATA_DIR)
    path = download_split(dataset, split, root, endpoint)
    size = path.stat().st_size
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{dataset}_{split}_{size}_{max_len}"
    cache = cache_dir / f"{key}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return z["ids"], z["y"], dict(n=int(z["y"].size), length=int(z["ids"].shape[1]))

    df = _read_csv_gz(path)
    seqs = df["seq"].astype(str).tolist()
    y = df["label"].to_numpy().astype(np.float32)
    lengths = np.array([len(s) for s in seqs])
    L = int(lengths.max()) if max_len is None else int(max_len)
    ids = dna.seqs_to_ids(seqs, L)
    if lengths.min() != lengths.max():
        print(f"[data] WARNING variable lengths {lengths.min()}..{lengths.max()}, padded to {L}")
    if use_cache:
        np.savez_compressed(cache, ids=ids, y=y)
    return ids, y, dict(n=int(y.size), length=L)


def class_balance(y: np.ndarray) -> dict:
    pos = float(np.mean(y == 1))
    return dict(n=int(y.size), n_pos=int(np.sum(y == 1)), n_neg=int(np.sum(y == 0)),
                pos_frac=round(pos, 4))


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #

def stratified_split(y: np.ndarray, val_frac: float = 0.15, seed: int = 0):
    """Plain random stratified split - the *wrong* way, kept as a control."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=y)
    return np.sort(tr), np.sort(va)


def homology_clusters(ids: np.ndarray, k: int = 20) -> np.ndarray:
    """Connected components of the graph "shares >= 1 exact k-mer".

    k=20 is exact (base-4, no hashing), so a link means a genuinely identical
    20-mer.  For 200 bp sequences two unrelated rows share a 20-mer with
    probability ~6e-7, i.e. essentially never, while homologs and duplicated
    regions are linked.  The result is a conservative near-duplicate grouping.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = ids.shape[0]
    codes = dna.kmer_codes(ids, k)
    if codes.size == 0:
        return np.arange(n)

    flat = codes.ravel()
    rows = np.repeat(np.arange(n, dtype=np.int64), codes.shape[1])
    order = np.argsort(flat, kind="stable")
    flat = flat[order]
    rows = rows[order]

    same = np.empty(flat.size, dtype=bool)
    same[0] = False
    same[1:] = flat[1:] == flat[:-1]
    a = rows[1:][same[1:]]
    b = rows[:-1][same[1:]]

    graph = coo_matrix((np.ones(a.size, dtype=np.int8), (a, b)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    return labels


def grouped_split(y: np.ndarray, clusters: np.ndarray, val_frac: float = 0.15,
                  seed: int = 0):
    """Split whole clusters, not rows, so no homolog spans train and validation."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    rng.shuffle(uniq)
    target = int(round(val_frac * len(y)))
    val_clusters, n_val = [], 0
    for c in uniq:
        if n_val >= target:
            break
        val_clusters.append(c)
        n_val += int(np.sum(clusters == c))
    if not val_clusters:  # pathological: one giant cluster
        raise RuntimeError(
            "homology clustering produced a single cluster - the dataset is one "
            "family; lower --kmer-k or use a random split deliberately"
        )
    mask = np.isin(clusters, np.array(val_clusters))
    return np.where(~mask)[0], np.where(mask)[0]


def leakage_report(train_idx, val_idx, clusters, ids, k: int = 20) -> dict:
    """Quantify how much of the val set is near-duplicated in train."""
    shared_clusters = len(set(clusters[train_idx]) & set(clusters[val_idx]))
    # exact k-mer overlap, sampled for speed on large sets
    n_check = min(len(val_idx), 2000)
    rng = np.random.default_rng(0)
    probe = rng.choice(val_idx, size=n_check, replace=False)
    # cap the train side too: 20k rows x 200bp is already ~3.6M codes
    train_sample = (train_idx if len(train_idx) <= 20000
                    else rng.choice(train_idx, size=20000, replace=False))
    train_codes = set(dna.kmer_codes(ids[train_sample], k).ravel().tolist())
    hits = 0
    for i in probe:
        if any(c in train_codes for c in dna.kmer_codes(ids[i : i + 1], k).ravel().tolist()):
            hits += 1
    return dict(shared_homology_clusters=int(shared_clusters),
                val_rows_sharing_a_kmer_with_train=round(hits / n_check, 4),
                kmer_k=k, probed=int(n_check))


def fingerprint(ids: np.ndarray) -> str:
    return hashlib.sha256(ids.tobytes()).hexdigest()[:16]
