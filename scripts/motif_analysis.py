"""Turn a trained DNA classifier into something a biologist can argue with.

The claim under test is *not* "the model is accurate".  It is "the model's
decision is attributable to a short, repeatable sequence pattern" - and that
claim can fail while AUPRC looks great.  So this script refuses to print a
pretty saliency picture on its own:

1. integrated gradients (IG) give a per-base track;
2. in-silico mutagenesis (ISM) gives a per-base, per-allele counterfactual;
3. the two must agree on *where* the decision sits (Jaccard over the top 5%);
4. the attributed region must collapse to a position weight matrix (a motif),
   not a smear;
5. individual variants get an effect size, which is the form a wet-lab reader
   actually wants.

If step 3 or 4 fails, the honest writeup says "the AUC is real but the
explanation is not" - which is a result, and a much better one than a fake logo.

Usage
-----
    python scripts/motif_analysis.py --run-dir runs/<run> --n-seqs 200
    python scripts/motif_analysis.py --run-dir runs/<run> --n-seqs 64 --no-ism
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import attributions as attr_lib  # noqa: E402
from src import data as gdata  # noqa: E402
from src import dna  # noqa: E402


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, ids, device, batch_size=128) -> np.ndarray:
    import torch

    model.eval()
    xs = torch.from_numpy(dna.one_hot(ids))
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            out.append(torch.sigmoid(model(xs[i:i + batch_size].to(device))).cpu().numpy())
    return np.concatenate(out)


def pwm_from_top(attr: np.ndarray, ids: np.ndarray, width: int = 8,
                 n_sites: int = 100) -> dict:
    """Build a PWM from the highest-attribution windows.

    Only windows fully inside the sequence and free of padding count, so the
    matrix is not silently dominated by the all-zero one-hot of 'N'.
    """
    per_pos = np.abs(attr).sum(axis=0)
    order = np.argsort(-per_pos)
    counts = np.zeros((width, 4), dtype=np.float64)
    used, skipped = 0, 0
    half = width // 2
    for centre in order:
        if used >= n_sites:
            break
        s = int(centre) - half
        if s < 0 or s + width > attr.shape[1]:
            skipped += 1
            continue
        block = ids[:, s:s + width]
        ok = (block < dna.N_BASES).all(axis=1)
        if not ok.any():
            skipped += 1
            continue
        for row in block[ok]:
            for j, b in enumerate(row):
                counts[j, int(b)] += 1.0
        used += 1
    total = counts.sum(axis=1, keepdims=True)
    probs = counts / np.clip(total, 1, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        info = 2.0 + np.nansum(np.where(probs > 0, probs * np.log2(probs), 0.0), axis=1)
    consensus = "".join(dna.BASES[int(np.argmax(probs[j]))] for j in range(width))
    gc_share = float(np.mean(probs[:, 1] + probs[:, 2]))
    return {"width": width, "n_sites": used, "n_skipped_windows": skipped,
            "consensus": consensus,
            "consensus_from": dna.BASES,
            "probabilities": np.round(probs, 3).tolist(),
            "bits_per_column": np.round(info, 2).tolist(),
            "gc_share": round(gc_share, 3),
            "mean_bits": round(float(np.mean(info)), 2)}


def variant_table(ism_win: np.ndarray, ids: np.ndarray, start: int,
                  probs: np.ndarray, top_n: int = 10, context: int = 12) -> list[dict]:
    """The strongest in-silico variants, with the reference base and a context.

    ``ism_win`` is in **logit** space, so the effect size survives saturation:
    a trained model sits at p=0.999 and every probability-space delta rounds to
    +0.0000.  Both spaces are reported, the logit one is the ranking key.
    """
    n, w, _ = ism_win.shape
    flat = ism_win.reshape(-1, 4)
    ref = ids[:, start:start + w].reshape(-1)
    gain = np.where(np.arange(4)[None, :] == ref[:, None], -np.inf, flat)
    rows = []
    order = np.argsort(-gain.ravel())[:top_n * 20]
    seen = 0
    for flat_idx in order:
        if seen >= top_n:
            break
        n_i, off, alt = np.unravel_index(flat_idx, (n, w, 4))
        ref_base = int(ids[n_i, start + off])
        if ref_base >= dna.N_BASES:
            continue  # mutating padding teaches nothing
        lo = max(0, start + off - context)
        hi = min(ids.shape[1], start + off + context + 1)
        ctx = dna.ids_to_seq(ids[n_i, lo:hi])
        if "N" in ctx:
            continue
        rel = start + off - lo
        dz = float(ism_win[n_i, off, alt])
        z_ref = float(np.log(probs[n_i] / max(1 - probs[n_i], 1e-12)))
        dp = float(1.0 / (1.0 + np.exp(-(z_ref + dz))) - probs[n_i])
        rows.append({
            "seq_index": int(n_i),
            "position": int(start + off),
            "ref": dna.BASES[ref_base],
            "alt": dna.BASES[int(alt)],
            "delta_logit": round(dz, 4),
            "delta_prob": round(dp, 6),
            "context": f"{ctx[:rel]}[{dna.BASES[ref_base]}>{dna.BASES[int(alt)]}]{ctx[rel + 1:]}",
        })
        seen += 1
    return rows


def make_figure(path: Path, ig: np.ndarray, ism_max: np.ndarray, start: int,
                probs: np.ndarray, pwm: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(3, len(ig))
    fig, axes = plt.subplots(k + 1, 1, figsize=(11, 2.4 * (k + 1)))
    for ax, i in zip(axes[:k], range(k)):
        x = np.arange(ig.shape[1])
        ax.fill_between(x, ig[i], 0, color="#3b6ea5", alpha=0.85, linewidth=0)
        ax.set_ylabel(f"IG\np={probs[i]:.2f}", fontsize=8)
        ax.set_xlim(0, ig.shape[1])
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="black", linewidth=0.6)
    # the ISM window shown against the attribution inside it
    axes[k].bar(np.arange(start, start + len(ism_max)), ism_max, color="#b5651d",
                width=1.0)
    axes[k].set_xlim(0, ig.shape[1])
    axes[k].set_ylabel("ISM max\n|Δlogit|", fontsize=8)
    axes[k].tick_params(labelsize=7)
    for ax in axes[:-1]:
        ax.axvspan(start, start + len(ism_max), color="grey", alpha=0.12, linewidth=0)
    axes[0].set_title(
        f"integrated gradients (top) vs in-silico mutagenesis (bottom, shaded window)\n"
        f"learned consensus {pwm['consensus']} ({pwm['mean_bits']} bits/col)",
        fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n-seqs", type=int, default=200,
                   help="how many high-confidence positives to explain")
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--ism-window", type=int, default=256,
                   help="bases around the IG peak to mutate (ISM is O(L*4) forwards)")
    p.add_argument("--no-ism", action="store_true", help="skip ISM (no gradient-free check)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = pick_device(args.device)
    run_dir = Path(args.run_dir)
    model, ckpt = attr_lib.load_model(run_dir, device)
    train_args = ckpt["args"]
    dataset = train_args["dataset"]
    length = int(ckpt["length"])

    ids, y, _ = gdata.load_split(dataset, "test", max_len=length)
    probs = predict(model, ids, device)

    # explain the most confident positives: if a motif exists, it must be here
    pos = np.where(y > 0.5)[0]
    pos = pos[np.argsort(-probs[pos])]
    probe_idx = pos[: args.n_seqs]
    x = ids[probe_idx]

    print(f"[motif] {dataset}: explaining top {len(probe_idx)} positives "
          f"(p in {probs[probe_idx].min():.3f}..{probs[probe_idx].max():.3f}), device={device}")
    ig = attr_lib.integrated_gradients(model, x, device=device, steps=args.ig_steps)

    # centre the ISM window on where the attribution actually is
    peak = int(np.argmax(np.abs(ig).mean(axis=0)))
    half = args.ism_window // 2
    start = max(0, min(peak - half, length - args.ism_window))
    end = min(length, start + args.ism_window)

    ism_win = None
    sanity = {}
    variants: list[dict] = []
    if not args.no_ism:
        print(f"[motif] ISM over bases {start}..{end} ({4 * (end - start)} forwards, logit space)")
        # logit space: the model is saturated, and IG also works on logits, so
        # this is the only space where the agreement check is apples-to-apples
        ism_win = attr_lib.ism(model, x, device=device, window=(start, end), space="logit")
        full_ism = np.zeros((len(x), length, 4), dtype=np.float32)
        full_ism[:, start:end] = ism_win
        sanity = attr_lib.motif_sanity_check(ig, full_ism, top_frac=0.05)
        variants = variant_table(ism_win, x, start, probs[probe_idx])

    ism_max = (np.abs(ism_win).max(axis=-1).mean(axis=0) if ism_win is not None
               else np.zeros(0))
    pwm = pwm_from_top(ig, x, width=8, n_sites=100)
    kmers = attr_lib.top_kmer(ig, x, k=6, top_n=200)

    out_prefix = Path(args.out_prefix or f"motif_{run_dir.name}")
    if out_prefix.parent == Path("."):
        out_prefix = ROOT / "reports" / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    make_figure(png, ig, ism_max, start, probs[probe_idx], pwm)

    verdict = "unverified (ISM skipped)"
    if sanity:
        agree = sanity["jaccard_ig_vs_ism"] >= 0.3
        focused = pwm["mean_bits"] >= 0.6
        if agree and focused:
            verdict = "attribution and counterfactual agree, and the region collapses to a motif"
        elif agree:
            verdict = "methods agree but the attributed region is diffuse - no clean motif"
        else:
            verdict = "IG and ISM disagree - at least one is describing noise; do not claim a motif"

    lines = [f"# Motif / attribution report - `{run_dir.name}`", "",
             f"- dataset: `{dataset}`, model: `{train_args['model']}`, "
             f"input length {length} bp",
             f"- explained: top {len(probe_idx)} test positives "
             f"(p {probs[probe_idx].min():.3f}-{probs[probe_idx].max():.3f})",
             f"- IG steps: {args.ig_steps}; ISM window: {start}..{end} "
             f"({end - start} bp around the IG peak at {peak})", "",
             "## Does the explanation survive checking?", ""]
    if sanity:
        lines += [f"- IG vs ISM top-5% Jaccard: **{sanity['jaccard_ig_vs_ism']}**",
                  f"- mean peak concentration of the IG track: "
                  f"{sanity['mean_peak_concentration']} "
                  f"(1.0 = one hot base, 1/L = flat)", ""]
    lines += [f"- learned consensus: `{pwm['consensus']}` "
              f"({pwm['mean_bits']} bits per column, {pwm['n_sites']} windows)",
              f"- C+G share of those columns: **{pwm['gc_share']:.2f}** "
              f"(0.50 = no preference, 1.00 = pure GC)", ""]
    if pwm["mean_bits"] < 0.6 and pwm["gc_share"] > 0.8:
        lines += ["> The attributed windows are C/G-rich but position-independent: "
                  "this is **base composition**, not a transcription-factor motif. "
                  "It is the same signal the GC-only linear model already captures, "
                  "so the network is not evidence of regulatory grammar here.", ""]
    lines += [f"**Verdict:** {verdict}", "",
              "### Position weight matrix", "",
              "```", "        " + "  ".join(f"{b:>5}" for b in dna.BASES)]
    for j in range(pwm["width"]):
        cells = "  ".join(f"{v:5.2f}" for v in pwm["probabilities"][j])
        lines.append(f"  pos{j}  {cells}   {pwm['bits_per_column'][j]:.2f} bits")
    lines += ["```", "", "### Most attributed 6-mers", "",
              "| 6-mer | count |", "|---|---|"]
    for row in kmers:
        lines.append(f"| `{row['kmer']}` | {row['count']} |")

    if variants:
        lines += ["", "### In-silico variants that increase the predicted activity", "",
                  "The same table would be the deliverable for a variant-effect project. "
                  "Effect sizes are in logit space (Δp is rounded to 6 decimals and is "
                  "near zero simply because the model is saturated).", "",
                  "| sequence | position | change | Δlogit | Δp | context |",
                  "|---|---|---|---|---|---|"]
        for v in variants:
            lines.append(f"| {v['seq_index']} | {v['position']} | {v['ref']}>{v['alt']} | "
                         f"{v['delta_logit']:+.4f} | {v['delta_prob']:+.6f} | "
                         f"`{v['context']}` |")

    md = "\n".join(lines) + "\n"
    md_path = out_prefix.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    (out_prefix.with_suffix(".json")).write_text(json.dumps({
        "run": run_dir.name, "dataset": dataset, "n_seqs": int(len(probe_idx)),
        "sanity": sanity, "pwm": pwm, "top_kmers": kmers, "variants": variants,
        "verdict": verdict, "ism_window": [int(start), int(end)],
        "figure": str(png),
    }, indent=2), encoding="utf-8")
    print(md)
    print(f"[done] {md_path}")
    print(f"[done] {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
