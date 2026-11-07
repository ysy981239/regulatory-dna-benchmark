"""Train a DNA sequence classifier on a Genomic Benchmarks dataset.

Design notes that matter for the writeup
----------------------------------------
* validation split is configurable: ``random`` (rows) or ``group`` (homology
  clusters).  Running both and reporting the gap is the point, not an accessory.
* reverse-complement augmentation is a *biological prior*: the biology of a
  promoter does not depend on which strand the reference genome happens to be
  stored in.  Toggling it is an experiment.
* model selection is on validation AUPRC, never on test.
* the test set is touched exactly once, at the end.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import data as gdata
from . import dna
from . import evaluate as ev
from . import models as gmodels

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, ids: np.ndarray, y: np.ndarray):
        self.ids = ids
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.y[i]


def make_collate(rc_augment: bool, seed: int = 0):
    rng = np.random.default_rng(seed)

    def collate(batch):
        ids = np.stack([b[0] for b in batch])
        y = np.array([b[1] for b in batch], dtype=np.float32)
        if rc_augment:
            flip = rng.random(len(ids)) < 0.5
            if flip.any():
                ids[flip] = dna.reverse_complement(ids[flip])
        x = torch.from_numpy(dna.one_hot(ids))
        return x, torch.from_numpy(y)

    return collate


class EMA:
    """Averaging the weights, not the predictions: cheap variance reduction."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()


def build_loaders(ids, y, batch_size, rc_augment, num_workers, seed, shuffle):
    ds = SeqDataset(ids, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=make_collate(rc_augment and shuffle, seed),
        persistent_workers=False,
    )


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def train_one_epoch(model, loader, opt, scaler, device, use_amp, grad_clip):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = crit(logits, y)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        total += loss.item() * y.numel()
        n += y.numel()
    return total / max(n, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="drosophila_enhancers_stark")
    p.add_argument("--model", default="basset", choices=["basset", "dilated", "transformer"])
    p.add_argument("--split", default="group", choices=["random", "group"],
                   help="random = rows; group = whole homology clusters")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--kmer-k", type=int, default=20, help="k for homology clustering")
    p.add_argument("--rc-augment", action="store_true", default=True)
    p.add_argument("--no-rc-augment", dest="rc_augment", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="0 disables EMA; 0.999 is a reasonable start")
    p.add_argument("--patience", type=int, default=0,
                   help="stop after N epochs without val AUPRC gain; 0 runs the full "
                        "cosine schedule (recommended: the last epochs are the ones "
                        "that anneal the LR, so stopping early throws them away)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--width", type=int, default=None, help="model width override")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    info = gdata.dataset_info(args.dataset)

    ids_tr, y_tr, meta = gdata.load_split(args.dataset, "train", max_len=args.max_len)
    ids_te, y_te, _ = gdata.load_split(args.dataset, "test", max_len=args.max_len)
    L = meta["length"]

    if args.split == "group":
        clusters = gdata.homology_clusters(ids_tr, k=args.kmer_k)
        tr_idx, va_idx = gdata.grouped_split(y_tr, clusters, args.val_frac, args.seed)
        split_note = f"grouped by shared {args.kmer_k}-mer ({len(np.unique(clusters))} clusters)"
    else:
        clusters = None
        tr_idx, va_idx = gdata.stratified_split(y_tr, args.val_frac, args.seed)
        split_note = "random rows (leaky by construction - this is the control)"
    leak = gdata.leakage_report(tr_idx, va_idx, clusters if clusters is not None
                                else np.arange(len(y_tr)), ids_tr, k=args.kmer_k)

    kw = {}
    if args.width:
        kw["width"] = args.width
    model = gmodels.build_model(args.model, L, **kw).to(device)
    print(f"[model] {gmodels.describe(model)}  input_len={L}")

    train_loader = build_loaders(ids_tr[tr_idx], y_tr[tr_idx], args.batch_size,
                                 args.rc_augment, args.num_workers, args.seed, shuffle=True)
    val_loader = build_loaders(ids_tr[va_idx], y_tr[va_idx], args.batch_size,
                               False, args.num_workers, args.seed, shuffle=False)
    test_loader = build_loaders(ids_te, y_te, args.batch_size, False,
                                args.num_workers, args.seed, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # the schedule advances once per epoch (see sched.step() in the loop), so it
    # has to be defined in epochs too.  It used to be defined in *steps* while
    # being stepped in epochs, which meant training never left the linear warmup:
    # the LR reached only ~warmup_frac of args.lr and the cosine never annealed.
    total_steps = max(args.epochs, 1)
    warmup = max(int(round(args.warmup_frac * total_steps)), 1)

    def lr_lambda(step):
        # LambdaLR calls lr_lambda(0) at construction; without this floor the
        # first epoch would train at lr=0 and silently waste 1/30 of the budget
        step = max(step, 1)
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    # the run name has to encode every ablation, otherwise two runs silently
    # overwrite each other (the rc-augment ablation once ate the rc run)
    tag = args.model
    if not args.rc_augment:
        tag += "-norc"
    if args.width:
        tag += f"-w{args.width}"
    run_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "runs" / f"{args.dataset}__{tag}__{args.split}__s{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run_dir / "split_info.json").write_text(json.dumps(
        {"split": split_note, "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
         "n_test": int(len(y_te)), "leakage": leak, "dataset": info,
         "class_balance_train": gdata.class_balance(y_tr[tr_idx]),
         "class_balance_test": gdata.class_balance(y_te)}, indent=2), encoding="utf-8")

    best = {"auprc": -1.0, "epoch": -1, "state": None}
    history = []
    patience_left = args.patience
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, device, use_amp,
                                  args.grad_clip)
        sched.step()
        if ema is not None:
            ema.update(model)
            eval_model = ema.shadow
        else:
            eval_model = model
        yv, pv = predict(eval_model, val_loader, device)
        vm = ev.binary_metrics(yv, pv, n_boot=0)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 5),
                        "val_auprc": vm["auprc"], "val_auroc": vm["auroc"],
                        "val_brier": vm["brier"], "lr": opt.param_groups[0]["lr"]})
        print(f"[{epoch:3d}/{args.epochs}] loss={tr_loss:.4f} "
              f"val_auprc={vm['auprc']:.4f} val_auroc={vm['auroc']:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e}")
        if vm["auprc"] > best["auprc"]:
            state = copy.deepcopy(eval_model.state_dict())
            best = {"auprc": vm["auprc"], "epoch": epoch, "state": state}
            torch.save({"state_dict": state, "args": vars(args), "length": L},
                       run_dir / "best.pt")
            patience_left = args.patience
        else:
            patience_left -= 1
        if args.patience > 0 and patience_left <= 0:
            print(f"[early-stop] no val AUPRC improvement for {args.patience} epochs")
            break
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if best["state"] is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best["state"])

    yt, pt = predict(model, test_loader, device)
    test_metrics = ev.binary_metrics(yt, pt, n_boot=200, seed=args.seed)
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "params": gmodels.n_params(model),
        "receptive_field": model.receptive_field(),
        "split": args.split,
        "split_detail": split_note,
        "leakage": leak,
        "best_epoch": best["epoch"],
        "best_val_auprc": best["auprc"],
        "test": test_metrics,
        "wall_clock_s": round(time.time() - t0, 1),
        "rc_augment": args.rc_augment,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
    }
    ev.dump_json(result, run_dir / "metrics.json")
    np.savez(run_dir / "test_predictions.npz", y=yt, p=pt)
    ev.save_reliability(yt, pt, run_dir / "reliability.png",
                        title=f"{args.dataset} {args.model} ({args.split})")
    ev.save_pr_curve(yt, pt, run_dir / "pr_curve.png", label=args.model,
                     baseline=float(yt.mean()))
    print("\n=== test ===")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"[done] {run_dir}")
    return result


if __name__ == "__main__":
    main()
