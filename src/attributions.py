"""Attribution for a trained DNA classifier - the part that must be biological.

Two complements, because they answer different questions:

* **in-silico mutagenesis (ISM)** - "what does this model predict for a single
  base change?"  Answerable per position and per substitution, no gradients, and
  it is exactly the counterfactual a wet-lab person asks about.  Its output is a
  (L, 4) delta-effect matrix that can be read as a variant-effect predictor.
* **integrated gradients (IG)** - "which bases does the decision aggregate
  over?", zero baseline, completes the gradient so saturation does not silently
  erase contributions.

A saliency map is *not* evidence of mechanism.  The honest check is either
(a) overlap with a motif database, or (b) consistency with ISM on the same
positions.  ``motif_sanity_check`` below does (b) automatically, and 
``scan_pfm`` does (a) when you supply a PFM.
"""
from __future__ import annotations

import numpy as np
import torch

from . import dna


@torch.no_grad()
def _forward(model, x: torch.Tensor, space: str = "prob") -> torch.Tensor:
    z = model(x)
    return z if space == "logit" else torch.sigmoid(z)


def ism(model, ids: np.ndarray, device="cpu", batch_size: int = 64,
        window: tuple[int, int] | None = None, space: str = "prob") -> np.ndarray:
    """Single-nucleotide effect matrix.

    Returns ``(N, L, 4)``: ``out[n, i, b] = f(ref with base b at i) - f(ref)``
    where ``f`` is the sigmoid (``space="prob"``, the default) or the raw logit
    (``space="logit"``).  The reference base entry is 0 by construction.

    Use ``space="logit"`` whenever the model is saturated: once p is 0.999 any
    single-base change moves the probability by <1e-3, so a probability-space
    matrix looks empty and the top variants become ties at +0.0000.  Logit space
    keeps the ranking intact and is also the space IG works in, which is what
    makes the IG-vs-ISM agreement check meaningful.

    ``window=(start, end)`` restricts the substitutions to that slice and returns
    ``(N, end-start, 4)``.  Everything outside the window stays at its reference
    value, so this is still a conditional effect given the rest of the sequence -
    it only saves the cost of mutating 3 kb of mostly-padding for a 251 bp model.
    """
    model.eval()
    ids = np.asarray(ids)
    n, length = ids.shape
    start, end = (0, length) if window is None else (int(window[0]), int(window[1]))
    start = max(0, min(start, length))
    end = max(start, min(end, length))
    out = np.zeros((n, end - start, 4), dtype=np.float32)
    x_ref = torch.from_numpy(dna.one_hot(ids)).to(device)
    p_ref = _forward(model, x_ref, space).cpu().numpy() if n <= batch_size else None
    if p_ref is None:
        p_ref = np.concatenate([
            _forward(model, x_ref[i:i + batch_size], space).cpu().numpy()
            for i in range(0, n, batch_size)
        ])
    for i in range(start, end):
        # batch the 3 substitutions per position over all sequences at once
        for b in range(4):
            mut = ids.copy()
            mut[:, i] = b
            x = torch.from_numpy(dna.one_hot(mut)).to(device)
            p = np.concatenate([
                _forward(model, x[j:j + batch_size], space).cpu().numpy()
                for j in range(0, n, batch_size)
            ])
            out[:, i - start, b] = p - p_ref
    return out


def integrated_gradients(model, ids: np.ndarray, device="cpu", steps: int = 32) -> np.ndarray:
    """Per-base importance ``(N, L)``, sum over channels, zero baseline."""
    model.eval()
    x = torch.from_numpy(dna.one_hot(np.asarray(ids))).to(device)
    x = x.requires_grad_(True)
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in range(1, steps + 1):
        xi = (base + (a / steps) * (x - base)).detach().requires_grad_(True)
        logits = model(xi)
        grads = torch.autograd.grad(logits.sum(), xi)[0]
        total += grads
    avg = (total / steps) * (x - base)
    return avg.sum(dim=1).detach().cpu().numpy()


def motif_sanity_check(attr: np.ndarray, ism_matrix: np.ndarray, top_frac: float = 0.05) -> dict:
    """Do the two attribution methods agree on *where* the decision is?

    Disagreement means at least one of them is describing the model's noise
    rather than a feature, which is exactly the check people skip.
    """
    a = np.abs(attr)
    k = max(int(top_frac * a.shape[-1]), 1)
    rank = lambda m: np.argsort(np.argsort(-np.abs(m), axis=-1), axis=-1) < k
    top_a = rank(a)
    top_i = rank(np.abs(ism_matrix).max(axis=-1))
    inter = (top_a & top_i).sum(axis=-1)
    union = (top_a | top_i).sum(axis=-1)
    jacc = float(np.mean(inter / np.clip(union, 1, None)))
    # how concentrated is the attribution? a flat map means "no motif"
    conc = float(np.mean(np.abs(attr).max(axis=-1) / np.clip(np.abs(attr).sum(axis=-1), 1e-9, None)))
    return {"top_k": k, "jaccard_ig_vs_ism": round(jacc, 4),
            "mean_peak_concentration": round(conc, 4)}


def top_kmer(attr: np.ndarray, ids: np.ndarray, k: int = 6, top_n: int = 200) -> list[dict]:
    """Most attributed k-mers - a crude, honest 'what did it learn' summary."""
    from collections import Counter

    scores = np.abs(attr).mean(axis=0)  # (L,)
    order = np.argsort(-scores)
    counts = Counter()
    for start in order[:top_n]:
        end = start + k
        if end > len(scores):
            continue
        for row in ids[:64]:
            seg = dna.ids_to_seq(row[start:end])
            if "N" not in seg:
                counts[seg] += 1
    return [{"kmer": s, "count": c} for s, c in counts.most_common(15)]


def load_model(run_dir, device="cpu"):
    from . import models as gmodels

    ckpt = torch.load(str(run_dir) + "/best.pt", map_location=device, weights_only=False)
    a = ckpt["args"]
    kw = {}
    if a.get("width"):
        kw["width"] = a["width"]
    model = gmodels.build_model(a["model"], ckpt["length"], **kw).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
