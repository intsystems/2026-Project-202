"""Small decoder-only language-model benchmark for MG versus Hessian diagnostics.

This is a practical proof of concept, not a claim about large-scale LLM pretraining.
The model is trained on a local technical-text corpus.  We record scalar probe loss and
gradient norm, estimate MG from those logs, and compare it with a Hessian top-eigenvalue
estimate at the same checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = ROOT / "research_llm_eos"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_corpus() -> str:
    files = [
        ROOT / "aistats_article" / "aistats2027.tex",
        ROOT / "icomp_article" / "grokking_en.tex",
        ROOT / "icomp_v2" / "MG_THEORETICAL_PROPERTIES.md",
    ]
    chunks = []
    for path in files:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n\n".join(chunks)
    if len(text) < 20_000:
        raise RuntimeError("the local corpus is unexpectedly short")
    return text


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Explicit attention is a little slower than PyTorch's fused CPU kernel, but its
        # second derivative is available and is required by the Hessian-vector baseline.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        batch, length, width = y.shape
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask[None, None]
        weights = torch.softmax(scores, dim=-1)
        y = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, length, width)
        y = self.attn_out(y)
        x = x + y
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, context: int = 64, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        mask = torch.full((context, context), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t, device=x.device))[None]
        mask = self.mask[:t, :t]
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln(h))


def batches(data: torch.Tensor, batch: int, context: int, generator: torch.Generator):
    max_start = len(data) - context - 1
    while True:
        starts = torch.randint(max_start, (batch,), generator=generator)
        x = torch.stack([data[int(s):int(s) + context] for s in starts])
        y = torch.stack([data[int(s) + 1:int(s) + context + 1] for s in starts])
        yield x, y


def flat(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in values])


def grad_norm(model: nn.Module) -> float:
    return float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                for p in model.parameters() if p.grad is not None)))


def top_hessian_eigenvalue(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                           iters: int = 8) -> Tuple[float, int]:
    """Largest algebraic Hessian eigenvalue estimated by HVP power iteration."""
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x).reshape(-1, model.head.out_features),
                                       y.reshape(-1))
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    v = [torch.randn_like(p) for p in params]
    norm = torch.linalg.vector_norm(flat(v))
    v = [u / norm for u in v]
    previous = None
    lam = float("nan")
    used = 0
    for used in range(1, iters + 1):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        hv_flat = flat(hv)
        v_flat = flat(v)
        lam = float(torch.dot(hv_flat, v_flat))
        nrm = torch.linalg.vector_norm(hv_flat)
        if not torch.isfinite(nrm) or float(nrm) == 0.0:
            break
        v = [h / nrm for h in hv]
        if previous is not None and abs(lam - previous) <= 1e-3 * max(1.0, abs(lam)):
            break
        previous = lam
    model.zero_grad(set_to_none=True)
    return lam, used


def run_one(seed: int, mode: str, steps: int, outdir: Path) -> Dict[str, object]:
    seed_all(seed)
    device = torch.device("cpu")
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(encoded))
    train_data, val_data = encoded[:split], encoded[split:]
    context, batch = 64, 32
    model = TinyGPT(len(chars), context=context).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed + 1000)
    train_stream = batches(train_data, batch, context, gen)
    probe_gen = torch.Generator().manual_seed(seed + 2000)
    probe_x, probe_y = next(batches(val_data, 32, context, probe_gen))
    hess_x, hess_y = probe_x[:8], probe_y[:8]
    switch = steps // 2
    hess_every = max(100, steps // 10)
    rows: List[Dict[str, float]] = []
    hess_rows: List[Dict[str, float]] = []
    started = time.perf_counter()

    for step in range(steps):
        if mode == "lr_switch" and step == switch:
            for group in opt.param_groups:
                group["lr"] = 3e-4
        x, y = next(train_stream)
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        gnorm = grad_norm(model)
        opt.step()

        with torch.no_grad():
            model.eval()
            probe_logits = model(probe_x)
            probe_loss = loss_fn(probe_logits.reshape(-1, len(chars)), probe_y.reshape(-1))
        rows.append({
            "step": step,
            "train_loss": float(loss.detach()),
            "probe_loss": float(probe_loss),
            "grad_norm": gnorm,
            "lr": float(opt.param_groups[0]["lr"]),
        })

        if step % hess_every == 0 or step == steps - 1:
            h_started = time.perf_counter()
            lam, used = top_hessian_eigenvalue(model, hess_x, hess_y)
            hess_rows.append({
                "step": step,
                "lambda_max": lam,
                "eta_lambda_over_2": float(opt.param_groups[0]["lr"] * lam / 2),
                "power_iters": used,
                "seconds": time.perf_counter() - h_started,
            })

    train_seconds = time.perf_counter() - started
    frame = pd.DataFrame(rows)
    hframe = pd.DataFrame(hess_rows)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / f"{mode}_seed{seed}_trace.csv", index=False)
    hframe.to_csv(outdir / f"{mode}_seed{seed}_hessian.csv", index=False)
    return {
        "seed": seed,
        "mode": mode,
        "steps": steps,
        "train_seconds": train_seconds,
        "hessian_seconds": float(hframe["seconds"].sum()),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "vocab": len(chars),
        "context": context,
        "batch": batch,
        "switch_step": switch if mode == "lr_switch" else None,
        "hessian_every": hess_every,
    }


def run_mg(outdir: Path, records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(CODE))
    from actdim.estimator import windows
    from actdim.frozen import eight_direction

    max_steps = max(int(pd.read_csv(outdir / f"{records[0]['mode']}_seed{records[0]['seed']}_trace.csv").shape[0]), 1)
    window = min(500, max(120, max_steps // 3))
    stride = max(20, window // 5)
    cfg = eight_direction(max_E=20, tau="acorr", theiler="autocorr",
                          theiler_cap=150, window=window, stride=stride,
                          spectral_bins=())
    rows = []
    for rec in records:
        prefix = f"{rec['mode']}_seed{rec['seed']}"
        trace = pd.read_csv(outdir / f"{prefix}_trace.csv")
        for signal in ("probe_loss", "grad_norm"):
            values = trace[signal].to_numpy(dtype=float)
            for transform in ("raw", "log"):
                x = np.log(np.maximum(values, 1e-12)) if transform == "log" else values
                started = time.perf_counter()
                right, traces = windows.sliding(x, cfg, seed=int(rec["seed"]))
                elapsed = time.perf_counter() - started
                for i, edge in enumerate(right):
                    rows.append({
                        "mode": rec["mode"], "seed": rec["seed"],
                        "signal": signal, "transform": transform,
                        "step": int(edge), "MG": float(traces["MG"][i]),
                        "LB": float(traces["LB"][i]),
                        "PRdelay": float(traces["PRdelay"][i]),
                        "degenerate": bool(traces["degenerate"][i]),
                        "mg_seconds": elapsed,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "mg_trace.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--modes", nargs="+", default=["constant_lr", "lr_switch"])
    parser.add_argument("--outdir", type=Path, default=OUT / "results")
    args = parser.parse_args()
    records = []
    for mode in args.modes:
        for seed in args.seeds:
            records.append(run_one(seed, mode, args.steps, args.outdir))
    summary = pd.DataFrame(records)
    summary.to_csv(args.outdir / "run_summary.csv", index=False)
    mg = run_mg(args.outdir, records)
    mg_time = (mg.groupby(["mode", "seed", "signal", "transform"], as_index=False)
                 .agg(mg_seconds=("mg_seconds", "first"), n_windows=("MG", "size")))
    mg_time.to_csv(args.outdir / "mg_timing.csv", index=False)
    (args.outdir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2),
                                              encoding="utf-8")
    print(summary.to_string(index=False))
    print(mg_time.to_string(index=False))


if __name__ == "__main__":
    main()
