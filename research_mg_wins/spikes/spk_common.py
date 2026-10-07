"""Shared pieces for S4 (loss-spike early warning): corpus, tiny transformer, logged training.

Every scalar the training loop writes is logged at every step:
  loss          mini-batch cross-entropy (nats/char)
  grad_norm     global L2 norm of the gradient (before the optimiser step, no clipping)
  param_norm    global L2 norm of all parameters (after the step)
  update_norm   L2 norm of the parameter change made by the step
  attn_max      max attention logit q.k/sqrt(dh) over layers, heads, batch, positions (internal)
  attn_ent      min over layers/heads of the mean attention entropy (internal)
  logz          mean log-partition (logsumexp) of the output logits (internal; Wortsman 2023)
Validation loss on a fixed held-out batch every VAL_EVERY steps.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.npy"
SITE = Path(r"C:\Users\karlo\AppData\Local\Programs\Python\Python313\Lib\site-packages")
VOCAB = 98            # 0 = unknown, 1 = '\n', 2 = '\t', 3.. = chr(32..126)
VAL_EVERY = 250


def build_corpus(max_bytes=4_000_000):
    files = []
    for p in ("numpy", "scipy", "sklearn"):
        files += sorted((SITE / p).rglob("*.py"))
    rng = np.random.default_rng(20261007)
    rng.shuffle(files)
    buf, n = [], 0
    for f in files:
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        buf.append(t)
        n += len(t)
        if n >= max_bytes:
            break
    text = "\n".join(buf)[:max_bytes]
    codes = np.frombuffer(text.encode("ascii", errors="replace"), dtype=np.uint8).astype(np.int64)
    out = np.zeros_like(codes)
    out[codes == 10] = 1
    out[codes == 9] = 2
    pr = (codes >= 32) & (codes <= 126)
    out[pr] = codes[pr] - 29
    np.save(CORPUS, out.astype(np.uint8))
    return out


def load_corpus():
    if not CORPUS.exists():
        build_corpus()
    c = np.load(CORPUS).astype(np.int64)
    n = int(0.9 * len(c))
    return torch.from_numpy(c[:n]), torch.from_numpy(c[n:])


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, mask, stats):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, -1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        logits = logits.masked_fill(~mask, float("-inf"))
        att = torch.softmax(logits, -1)
        with torch.no_grad():
            stats["attn_max"] = max(stats.get("attn_max", -1e30), float(logits.max()))
            ent = -(att * torch.log(att.clamp_min(1e-30))).sum(-1).mean((0, 2))   # per head
            stats["attn_ent"] = min(stats.get("attn_ent", 1e30), float(ent.min()))
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyGPT(nn.Module):
    """Pre-LN decoder-only transformer, no qk-layernorm, no z-loss (Wortsman et al. 2023)."""

    def __init__(self, d=64, layers=2, heads=4, ctx=64, vocab=VOCAB):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx, dtype=torch.bool)))

    def forward(self, idx, stats):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos[:T]
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m, stats)
        return self.head(self.lnf(x))


def lr_at(step, lr, warm, total, sched):
    if step < warm:
        return lr * (step + 1) / warm
    if sched == "const":
        return lr
    p = (step - warm) / max(1, total - warm)            # cosine to 10 %
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train(seed, lr, steps=8000, warm=500, beta2=0.95, wd=0.1, eps=1e-8, d=64, layers=2,
          heads=4, ctx=64, batch=32, sched="const", data_seed=None, intervene=None,
          init_state=None, start_step=0, log_prefix=None, diverge_loss=8.0, task="char",
          mod_p=97, train_frac=0.5):
    """Train one run and return a dict of per-step numpy logs.

    intervene: optional callable(step, logs_so_far) -> None | ("lr_mult", f) | ("rewind", f).
    It is consulted every step; used only by the downstream intervention experiment.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed if data_seed is None else data_seed)
    if task == "char":
        tr, va = load_corpus()
        gv = torch.Generator().manual_seed(12345)
        vstart = torch.randint(0, len(va) - ctx - 1, (256,), generator=gv)
        vx = torch.stack([va[s:s + ctx] for s in vstart]); vy = torch.stack([va[s + 1:s + ctx + 1] for s in vstart])
        model = TinyGPT(d, layers, heads, ctx)

        def get_batch():
            s = torch.randint(0, len(tr) - ctx - 1, (batch,), generator=g)
            return torch.stack([tr[i:i + ctx] for i in s]), torch.stack([tr[i + 1:i + ctx + 1] for i in s])
    else:                                   # modular addition a+b mod p, answer at last position
        P = mod_p
        a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
        X = torch.stack([a.reshape(-1), b.reshape(-1), torch.full((P * P,), P)], 1)
        Y = (X[:, 0] + X[:, 1]) % P
        perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(0))
        ntr = int(train_frac * P * P)
        Xtr, Ytr, vx, vy = X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]]
        model = TinyGPT(d, layers, heads, 3, vocab=P + 1)

        def get_batch():
            i = torch.randint(0, ntr, (batch,), generator=g)
            return Xtr[i], Ytr[i]
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, beta2), eps=eps)
    keys = ("loss", "grad_norm", "param_norm", "update_norm", "attn_max", "attn_ent", "logz", "lr")
    logs = {k: np.full(steps, np.nan, np.float32) for k in keys}
    val = {}
    params = list(model.parameters())
    lr_mult = 1.0
    t0 = time.perf_counter()
    bad = 0
    diverged_at = None
    for step in range(steps):
        cur_lr = lr_at(step, lr, warm, steps, sched) * lr_mult
        for gr in opt.param_groups:
            gr["lr"] = cur_lr
        x, y = get_batch()
        stats = {}
        out = model(x, stats)
        if task != "char":
            out = out[:, -1:, :]; y = y[:, None]
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))
            old = [p.detach().clone() for p in params]
        opt.step()
        with torch.no_grad():
            un = torch.sqrt(sum(((p - o) ** 2).sum() for p, o in zip(params, old)))
            pn = torch.sqrt(sum((p ** 2).sum() for p in params))
            logz = torch.logsumexp(out.detach(), -1).mean()
        lv = float(loss.detach())
        logs["loss"][step] = lv
        logs["grad_norm"][step] = float(gn)
        logs["param_norm"][step] = float(pn)
        logs["update_norm"][step] = float(un)
        logs["attn_max"][step] = stats["attn_max"]
        logs["attn_ent"][step] = stats["attn_ent"]
        logs["logz"][step] = float(logz)
        logs["lr"][step] = cur_lr
        if (step + 1) % VAL_EVERY == 0 or step == steps - 1:
            with torch.no_grad():
                vo = model(vx, {})
                if task != "char":
                    vo = vo[:, -1:, :]
                val[step + 1] = float(F.cross_entropy(vo.reshape(-1, vo.shape[-1]), vy.reshape(-1)))
            if log_prefix is not None and (step + 1) % 1000 == 0:
                np.savez(str(log_prefix) + ".partial.npz", **{k: v[:step + 1] for k, v in logs.items()})
        if not math.isfinite(lv) or lv > diverge_loss:
            bad += 1
        else:
            bad = 0
        if bad >= 200 or not all(math.isfinite(logs[k][step]) for k in ("loss", "grad_norm", "param_norm")):
            diverged_at = step
            break
        if intervene is not None:
            act = intervene(step, logs)
            if act is not None and act[0] == "lr_mult":
                lr_mult *= act[1]
    res = {k: v for k, v in logs.items()}
    res["val_steps"] = np.array(sorted(val), np.int64)
    res["val_loss"] = np.array([val[k] for k in sorted(val)], np.float32)
    res["diverged_at"] = -1 if diverged_at is None else diverged_at
    res["seconds"] = time.perf_counter() - t0
    return res
