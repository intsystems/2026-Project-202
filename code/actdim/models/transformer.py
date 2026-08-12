"""The one-layer transformer of appendix O.

The architecture of Nanda et al. (2023): token and learned positional embeddings, one
causal attention head group, one ReLU MLP, unembedding, and **no layer normalisation**.
Every weight is written out as an ``nn.Parameter`` rather than taken from ``nn.Linear``,
because that is how the published runs were produced and the parameter shapes are what
the trajectory sketch flattens.

Two things here are load-bearing and must not be tidied.

*The construction order.* Every weight is drawn from the global torch stream, which the
train/validation split and the mini-batch order also continue, so moving one
``nn.Parameter`` assignment past another changes the initial weights of every run. The
order below is the published one.

*The absence of normalisation.* The residual stream is unnormalised, which is what makes
the parameter norm the informative observable it is in section 7. Adding a norm layer
would change the quantity the article measures, not merely its scale.

The size is fixed by appendix O: ``d_model`` 128, four heads of 32, ``d_mlp`` 512, giving
226,816 parameters on modular addition at p = 113 and 228,608 on ``S_5``. Those two counts
are asserted in ``tests/test_transformer.py``; they are the cheapest check that the
architecture has not drifted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _std(fan: int, high_variance_init: bool) -> float:
    """The initialisation scale: ``1/sqrt(fan)``, or a deliberately large 2.0.

    The large scale is the Omnigrok ablation, kept because it is the one setting that
    changes the initialisation without changing anything else.
    """
    return 2.0 if high_variance_init else 1.0 / np.sqrt(fan)


class Embed(nn.Module):
    """Token embedding, held transposed as ``(d_model, d_vocab)`` as in the source."""

    def __init__(self, d_vocab: int, d_model: int, high_variance_init: bool = False):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) * _std(d_model, high_variance_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("dbp -> bpd", self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int, high_variance_init: bool = False):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) * _std(d_vocab, high_variance_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx: int, d_model: int, high_variance_init: bool = False):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) * _std(d_model, high_variance_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.W_pos[:x.shape[-2]]


class Attention(nn.Module):
    """Causal multi-head attention, one group of heads, no output bias."""

    def __init__(self, d_model: int, num_heads: int, d_head: int, n_ctx: int,
                 high_variance_init: bool = False):
        super().__init__()
        std = _std(d_model, high_variance_init)
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) * std)
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) * std)
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) * std)
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) * std)

        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)

        n = x.shape[-2]
        scores = torch.einsum("biph,biqh->biqp", k, q)
        masked = torch.tril(scores) - 1e10 * (1 - self.mask[:n, :n])
        attention = F.softmax(masked / np.sqrt(self.d_head), dim=-1)

        z = torch.einsum("biph,biqp->biqh", v, attention)
        # The heads are merged by permute/reshape rather than by einops: einops was a
        # dependency for one line, and its removal is why the shapes are spelled out.
        z_flat = z.permute(0, 2, 1, 3).reshape(z.shape[0], z.shape[2], -1)
        return torch.einsum("df,bqf->bqd", self.W_O, z_flat)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, high_variance_init: bool = False):
        super().__init__()
        std = _std(d_model, high_variance_init)
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) * std)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) * std)
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        x = F.relu(x)
        return torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, d_head: int, num_heads: int, n_ctx: int,
                 high_variance_init: bool = False):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, high_variance_init)
        self.mlp = MLP(d_model, d_mlp, high_variance_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class NandaTransformer(nn.Module):
    """One layer, no layer normalisation, logits for the last position only.

    The prompt is ``[a, b, =]`` and only the ``=`` position is scored, so ``forward``
    returns ``(batch, d_vocab)`` rather than a per-position tensor. Everything downstream
    -- the accuracy, the loss and the function sketch -- reads that one row.
    """

    def __init__(self, d_vocab: int, d_model: int = 128, d_mlp: int = 512, d_head: int = 32,
                 num_heads: int = 4, n_ctx: int = 3, high_variance_init: bool = False):
        super().__init__()
        self.embed = Embed(d_vocab, d_model, high_variance_init)
        self.pos_embed = PosEmbed(n_ctx, d_model, high_variance_init)
        self.block = TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx,
                                      high_variance_init)
        self.unembed = Unembed(d_vocab, d_model, high_variance_init)

    @property
    def embedding_weight(self) -> torch.Tensor:
        """The token-embedding tensor, for the ``embed_grad_norm`` observable."""
        return self.embed.W_E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.pos_embed(x)
        x = self.block(x)
        x = self.unembed(x)
        return x[:, -1]


def parameter_count(d_vocab: int, d_model: int = 128, d_mlp: int = 512, d_head: int = 32,
                    num_heads: int = 4, n_ctx: int = 3) -> int:
    """The parameter count, by arithmetic rather than by construction.

    Building the model to count its weights costs a second and a few hundred megabytes of
    float64; the article quotes the count in appendix O, and a test that has to allocate
    the network to check it is a test people stop running.
    """
    return (
        2 * d_model * d_vocab                 # W_E and W_U
        + n_ctx * d_model                     # W_pos
        + 3 * num_heads * d_head * d_model    # W_K, W_Q, W_V
        + d_model * d_head * num_heads        # W_O
        + d_mlp * d_model + d_mlp             # W_in, b_in
        + d_model * d_mlp + d_model           # W_out, b_out
    )


def build(config: Any, d_vocab: int, n_ctx: int = 3) -> NandaTransformer:
    """Instantiate the architecture a run configuration describes."""
    return NandaTransformer(
        d_vocab=d_vocab,
        d_model=config.d_model,
        d_mlp=config.d_mlp,
        d_head=config.d_head,
        num_heads=config.num_heads,
        n_ctx=n_ctx,
        high_variance_init=config.high_variance_init,
    )
