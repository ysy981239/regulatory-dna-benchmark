"""Sequence-to-function models on (B, 4, L) one-hot DNA.

Three deliberately different inductive biases, so that a comparison between them
is an experiment and not a leaderboard entry:

* ``basset``   - stacked wide convolutions, the DeepSEA/Basset family.
                 Local motif detectors + a large receptive field.
* ``dilated``  - residual dilated convolutions.  Reaches ~kb context with few
                 parameters; the "does long-range context help?" control.
* ``transformer`` - small DNA transformer with learned positional embeddings.
                 No locality prior; the "is convolution actually doing the work?" control.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, pool: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class Basset(nn.Module):
    """Wide-first CNN (Basset/DeepSEA lineage)."""

    def __init__(self, in_len: int, width: int = 128, dropout: float = 0.3):
        super().__init__()
        self.in_len = in_len
        self.features = nn.Sequential(
            ConvBlock(4, width, 15, 5),
            ConvBlock(width, width, 9, 4),
            ConvBlock(width, int(width * 1.5), 9, 4),
        )
        feat_dim = int(width * 1.5) * 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(h).squeeze(-1)

    def receptive_field(self) -> int:
        """Width of the input window that one final feature is computed from.

        Each layer adds ``(kernel - 1) * (product of the strides before it)``.
        Pooling counts as well: it adds no weights but it does widen the window
        (its own kernel spans ``kernel`` already-pooled positions).
        Verified numerically by backprop: 294 bases for in_len=251 (see
        ``tests/test_core.py``).  Dropping the pooling terms gives the commonly
        quoted 215, which undercounts by 79.
        """
        rf, jump = 1, 1
        for blk in self.features:
            rf += (blk.conv.kernel_size[0] - 1) * jump
            if isinstance(blk.pool, nn.MaxPool1d):
                k, s = blk.pool.kernel_size, blk.pool.stride
                k = k if isinstance(k, int) else k[0]
                s = s if isinstance(s, int) else s[0]
                rf += (k - 1) * jump
                jump *= s
        return rf


class DilatedResidual(nn.Module):
    def __init__(self, channels: int, k: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(channels, channels, k, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.act(self.bn(self.conv(x))))


class DilatedCNN(nn.Module):
    """Stem + exponentially dilated residual stack (WaveNet/Basenji lineage)."""

    def __init__(self, in_len: int, width: int = 96, n_blocks: int = 4,
                 dilations=(1, 2, 4, 8, 16), dropout: float = 0.1):
        super().__init__()
        self.in_len = in_len
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 11, padding=5, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [DilatedResidual(width, 5, d, dropout) for _ in range(n_blocks) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width * 2, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(h).squeeze(-1)

    def receptive_field(self) -> int:
        rf = 11
        for blk in self.blocks:
            rf += 4 * blk.conv.dilation[0]
        return rf


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.emb, std=0.02)

    def forward(self, x):
        return x + self.emb[:, : x.size(1)]


class DNATransformer(nn.Module):
    """Small pre-LN transformer over bases.  No convolutions at all."""

    def __init__(self, in_len: int, d_model: int = 128, nhead: int = 4,
                 layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_len = in_len
        self.proj = nn.Linear(4, d_model)
        self.pos = PositionalEncoding(d_model, in_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, 1)   # mean || max pooling

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))          # (B, L, d)
        h = self.encoder(self.pos(h))
        h = self.norm(h)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)

    def receptive_field(self) -> int:
        return self.in_len  # global by construction


_BUILDERS = {
    "basset": Basset,
    "dilated": DilatedCNN,
    "transformer": DNATransformer,
}


def build_model(name: str, in_len: int, **kwargs) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(_BUILDERS)}")
    model = _BUILDERS[name](in_len, **kwargs)
    # a tiny but real initialisation dependency: keep it deterministic
    model.apply(_init_weights)
    return model


def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module) -> str:
    rf = model.receptive_field() if hasattr(model, "receptive_field") else -1
    return f"{type(model).__name__}(params={n_params(model):,}, receptive_field={rf}bp)"
