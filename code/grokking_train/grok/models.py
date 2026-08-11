"""The two architectures the paper's logs were produced with.

``omnigrok``  -- the 1-layer transformer of **Nanda et al. (2023)**, written out
                 parameter-by-parameter (no ``nn.Linear``, no LayerNorm).  Used
                 for every mini-batch experiment (Figs. 1-3).

                 The key and the class name are historical and misleading: this
                 configuration (d_model 128, 4 heads of 32, d_mlp 512, p = 113,
                 AdamW at 1e-3, betas 0.9/0.98, weight decay 1) is Nanda et
                 al.'s, which Liu et al. (2022) adopt rather than originate.
                 They are left in place because the registry key and the
                 parameter-construction order below are load-bearing for
                 bit-identical reproduction of the published logs; the
                 attribution in the paper is to Nanda et al.

                 Note also that we train on mini-batches of 256 where the
                 source is full-batch.  That departure is deliberate and is
                 what places these runs in the paper's stochastic regime.
``encoder``   -- a stock ``nn.TransformerEncoder`` stack, used only for the
                 full-batch baseline of App. B.

The parameter-construction order of ``OmnigrokTransformer`` is load-bearing:
every weight is drawn from the global torch RNG, so reordering the ``nn.Parameter``
assignments changes the initial weights and the run stops reproducing the
published logs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _std(d, high_variance_init):
    """Omnigrok's initialisation scale: ``1/sqrt(fan)``, or a deliberately large 2.0."""
    return 2.0 if high_variance_init else 1.0 / np.sqrt(d)


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model, high_variance_init=False):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) * _std(d_model, high_variance_init))

    def forward(self, x):
        return torch.einsum("dbp -> bpd", self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model, high_variance_init=False):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) * _std(d_vocab, high_variance_init))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model, high_variance_init=False):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) * _std(d_model, high_variance_init))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, high_variance_init=False):
        super().__init__()
        std = _std(d_model, high_variance_init)
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) * std)
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) * std)
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) * std)
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) * std)

        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)

        n = x.shape[-2]
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:n, :n])
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)

        z = torch.einsum("biph,biqp->biqh", v, attn_matrix)
        # einops.rearrange(z, 'b i q h -> b q (i h)')
        z_flat = z.permute(0, 2, 1, 3).reshape(z.shape[0], z.shape[2], -1)
        return torch.einsum("df,bqf->bqd", self.W_O, z_flat)


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, high_variance_init=False):
        super().__init__()
        std = _std(d_model, high_variance_init)
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) * std)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) * std)
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        x = F.relu(x)
        return torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, high_variance_init=False):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, high_variance_init)
        self.mlp = MLP(d_model, d_mlp, high_variance_init)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class OmnigrokTransformer(nn.Module):
    """1-layer, LayerNorm-free transformer; logits for the last position only."""

    def __init__(self, d_vocab, d_model=128, d_mlp=512, d_head=32, num_heads=4,
                 n_ctx=3, high_variance_init=False):
        super().__init__()
        self.embed = Embed(d_vocab, d_model, high_variance_init)
        self.pos_embed = PosEmbed(n_ctx, d_model, high_variance_init)
        self.block = TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, high_variance_init)
        self.unembed = Unembed(d_vocab, d_model, high_variance_init)

    @property
    def embedding_weight(self):
        """The token-embedding tensor, for the ``embed_grad_norm`` observable."""
        return self.embed.W_E

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        x = self.block(x)
        x = self.unembed(x)
        return x[:, -1]


class EncoderTransformer(nn.Module):
    """Stock ``nn.TransformerEncoder`` baseline (App. B, full-batch GD)."""

    def __init__(self, d_vocab, d_model=128, num_heads=4, n_layers=1, n_ctx=3):
        super().__init__()
        self.token_emb = nn.Embedding(d_vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_ctx, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.unembed = nn.Linear(d_model, d_vocab)

    @property
    def embedding_weight(self):
        return self.token_emb.weight

    def forward(self, x):
        out = self.transformer(self.token_emb(x) + self.pos_emb)
        return self.unembed(out[:, -1, :])


def _build_omnigrok(config, vocab_size, n_ctx):
    return OmnigrokTransformer(
        d_vocab=vocab_size,
        d_model=config.d_model,
        d_mlp=config.d_mlp,
        d_head=config.d_head,
        num_heads=config.num_heads,
        n_ctx=n_ctx,
        high_variance_init=config.high_variance_init,
    )


def _build_encoder(config, vocab_size, n_ctx):
    return EncoderTransformer(
        d_vocab=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        n_layers=config.n_layers,
        n_ctx=n_ctx,
    )


MODELS = {"omnigrok": _build_omnigrok, "encoder": _build_encoder}


def build(config, vocab_size, n_ctx=3):
    """Instantiate the architecture named by ``config.model``."""
    if config.model not in MODELS:
        raise KeyError(f"unknown model '{config.model}'. Available: {', '.join(MODELS)}")
    return MODELS[config.model](config, vocab_size, n_ctx)
