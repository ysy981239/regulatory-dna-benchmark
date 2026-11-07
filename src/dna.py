"""DNA sequence encoding utilities.

Everything works on integer ids (A=0, C=1, G=2, T=3, anything else=255)
so that encoding is a single table lookup and stays vectorised.

Conventions
-----------
* ids are uint8, shape (N, L)
* one-hot tensors are (N, 4, L) float32, channel order A, C, G, T
* 255 ("N" / ambiguous) is *always* encoded as all-zero one-hot, never as a base,
  so an N never contributes evidence to a convolution.
"""
from __future__ import annotations

import numpy as np

BASES = "ACGT"
UNK = 255
N_BASES = 4

_LUT = np.full(256, UNK, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i

# complement of A,C,G,T,N -> T,G,C,A,N
_COMPLEMENT = np.array([3, 2, 1, 0, UNK], dtype=np.uint8)


def seq_to_ids(seq: str, max_len: int | None = None, pad_right: bool = True) -> np.ndarray:
    """Encode one sequence to uint8 ids, optionally truncating / right-padding."""
    arr = np.frombuffer(seq.encode("ascii", errors="replace")[:4096], dtype=np.uint8)
    ids = _LUT[arr]
    if max_len is not None:
        if len(ids) > max_len:
            ids = ids[:max_len]
        elif len(ids) < max_len:
            pad = np.full(max_len - len(ids), UNK, dtype=np.uint8)
            ids = np.concatenate([ids, pad]) if pad_right else np.concatenate([pad, ids])
    return ids


def seqs_to_ids(seqs, max_len: int | None = None) -> np.ndarray:
    """Encode many sequences; length is the max observed when max_len is None."""
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    return np.stack([seq_to_ids(s, max_len) for s in seqs])


def ids_to_seq(ids: np.ndarray) -> str:
    out = []
    for v in ids:
        out.append(BASES[v] if v < N_BASES else "N")
    return "".join(out)


def one_hot(ids: np.ndarray) -> np.ndarray:
    """(N, L) uint8 ids -> (N, 4, L) float32 one-hot (N/ambiguous all-zero)."""
    ids = np.asarray(ids)
    n, length = ids.shape
    oh = np.zeros((n, N_BASES, length), dtype=np.float32)
    for b in range(N_BASES):
        oh[:, b, :] = (ids == b)
    return oh


def reverse_complement(ids: np.ndarray) -> np.ndarray:
    """Reverse-complement along the last axis; 255 maps to 255."""
    ids = np.asarray(ids)
    comp = np.where(ids < N_BASES, _COMPLEMENT[np.minimum(ids, N_BASES)], UNK).astype(np.uint8)
    return comp[..., ::-1].copy()


#: Codes for real k-mers are < 4**31, so anything above this marks "ambiguous".
SENTINEL_BASE = 4 ** 31


def _row_keys(ids: np.ndarray) -> np.ndarray:
    """A 64-bit key that depends only on a row's *content*, never on its position.

    This is the reason the function exists: an earlier version used the row index
    inside the array being encoded, which made the code for a given window depend
    on batching.  Probing one row at a time (``kmer_codes(ids[i:i+1], k)``) then
    produced a sentinel that matched row 0 of the full matrix, so *every* padded
    row looked like a near-duplicate of train: the leakage audit reported 99.9%
    k-mer sharing even for a homology-disjoint split, which is impossible.  A
    content hash keeps the sentinel stable across batch shapes.
    """
    from hashlib import blake2b

    u8 = np.ascontiguousarray(ids, dtype=np.uint8)
    keys = [int.from_bytes(blake2b(row.tobytes(), digest_size=8).digest(), "little")
            for row in u8]
    return np.asarray(keys, dtype=np.uint64)


def kmer_codes(ids: np.ndarray, k: int) -> np.ndarray:
    """Exact base-4 encoding of every k-mer window.

    Returns (N, L-k+1) uint64.  Codes are exact (no hashing), so equality means
    identical k-mer.  Windows containing an ambiguous base are replaced by a
    *content-derived, per-row* sentinel in ``[SENTINEL_BASE, 2**64)``: two
    different rows can never be linked through an "NNNN..." window, while two
    byte-identical rows still are (they are genuine duplicates), and the value
    does not change when the row is encoded alone or inside a batch.
    """
    ids = np.asarray(ids, dtype=np.uint64)
    n, length = ids.shape
    if length < k:
        return np.zeros((n, 0), dtype=np.uint64)
    w = length - k + 1
    codes = np.zeros((n, w), dtype=np.uint64)
    invalid = np.zeros((n, w), dtype=bool)
    for j in range(k):
        col = ids[:, j : j + w]
        bad = col >= np.uint64(N_BASES)
        invalid |= bad
        codes = codes * np.uint64(N_BASES) + np.where(bad, np.uint64(0), col)
    if invalid.any():
        # keep the sum inside uint64: mod 2**64 - SENTINEL_BASE can never overflow
        mod = (1 << 64) - SENTINEL_BASE
        sentinel = np.uint64(SENTINEL_BASE) + (_row_keys(ids) % np.uint64(mod))[:, None]
        codes = np.where(invalid, sentinel, codes)
    return codes


def gc_fraction(ids: np.ndarray) -> np.ndarray:
    """GC fraction per row; ambiguous bases are ignored."""
    ids = np.asarray(ids)
    valid = (ids < N_BASES).sum(axis=1).clip(min=1)
    gc = ((ids == 1) | (ids == 2)).sum(axis=1)
    return gc / valid


def dinucleotide_frequencies(ids: np.ndarray) -> np.ndarray:
    """(N, 16) dinucleotide frequency matrix."""
    ids = np.asarray(ids)
    n, length = ids.shape
    counts = np.zeros((n, 16), dtype=np.float32)
    if length < 2:
        return counts
    a = ids[:, :-1].astype(np.int32)
    b = ids[:, 1:].astype(np.int32)
    valid = (a < N_BASES) & (b < N_BASES)
    code = (a * N_BASES + b)[valid]
    rows = np.broadcast_to(np.arange(n)[:, None], valid.shape)[valid]
    np.add.at(counts, (rows, code), 1.0)
    return counts / counts.sum(axis=1, keepdims=True).clip(min=1)


def kmer_frequencies(ids: np.ndarray, k: int = 5) -> np.ndarray:
    """(N, 4**k) k-mer frequency matrix (dense float32; keep k <= 6)."""
    ids = np.asarray(ids)
    n, length = ids.shape
    dim = N_BASES**k
    counts = np.zeros((n, dim), dtype=np.float32)
    if length < k:
        return counts
    # build the code incrementally with the same base-4 trick as kmer_codes,
    # but track validity so ambiguous windows are dropped instead of sentineled
    w = length - k + 1
    code = np.zeros((n, w), dtype=np.int64)
    valid = np.ones((n, w), dtype=bool)
    for j in range(k):
        col = ids[:, j : j + w].astype(np.int64)
        valid &= col < N_BASES
        code = code * N_BASES + np.where(col < N_BASES, col, 0)
    rows = np.broadcast_to(np.arange(n)[:, None], valid.shape)[valid]
    np.add.at(counts, (rows, code[valid]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True).clip(min=1)
