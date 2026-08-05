"""Canonical modular-addition grokking experiment (Google Colab friendly).

Task: predict (a + b) mod p from a fixed subset of pairs.  This is the
benchmark introduced in the original grokking work and is considerably more
reliable than S_n for testing delayed generalisation.  Every run writes
``training_log.csv``, ``config.json`` and periodic ``checkpoint.pt`` files,
plus TensorBoard event files when tensorboard is available.

The script is intentionally self contained: paste it into one Colab cell or
run it from Drive.  It does not claim a mathematical guarantee (grokking is
seed/implementation sensitive); ``seeds`` and ``retries_until_grok`` make the
protocol robust and record every attempt rather than silently selecting a
successful run.
"""
from __future__ import annotations

import argparse, json, math, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


@dataclass
class Config:
    output_root: str = "/content/drive/MyDrive/grokking_modular_addition"
    protocol_name: str = "modadd_p97_frac30_adamw_wd1"
    p: int = 97
    train_fraction: float = 0.30
    seeds: tuple[int, ...] = (0, 1, 2)
    d_model: int = 128
    d_hidden: int = 512
    n_heads: int = 4
    max_steps: int = 200_000
    batch_size: int = 0  # 0 = full batch; mini-batch is also supported
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    log_every: int = 100
    checkpoint_every: int = 5_000
    target_train_acc: float = 0.99
    target_val_acc: float = 0.95
    patience_logs: int = 5
    required_gap_steps: int = 10_000
    post_grok_steps: int = 5_000
    retries_until_grok: int = 1  # stop after first success; 0 runs every seed
    force_restart: bool = False
    projection_count: int = 3
    loss_projection_count: int = 3


class ModularAdditionTransformer(nn.Module):
    """Canonical one-block transformer used in modular-addition grokking."""
    def __init__(self, p: int, d_model: int, d_hidden: int, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads: raise ValueError("d_model must be divisible by n_heads")
        self.left = nn.Embedding(p, d_model)
        self.right = nn.Embedding(p, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 2, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_hidden, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)
        nn.init.normal_(self.left.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.right.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        h = torch.stack([self.left(x[:, 0]), self.right(x[:, 1])], dim=1) + self.pos
        h = self.norm(self.encoder(h))
        return self.unembed(h.mean(dim=1))


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_data(cfg: Config, seed: int):
    pairs = torch.cartesian_prod(torch.arange(cfg.p), torch.arange(cfg.p))
    # deterministic permutation gives a disjoint, reproducible split
    g = torch.Generator().manual_seed(seed)
    pairs = pairs[torch.randperm(len(pairs), generator=g)]
    n_train = int(round(len(pairs) * cfg.train_fraction))
    n_train = min(max(n_train, 1), len(pairs) - 1)
    y = (pairs[:, 0] + pairs[:, 1]) % cfg.p
    return pairs[:n_train], y[:n_train], pairs[n_train:], y[n_train:]


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=8192):
    model.eval(); total_loss = total_correct = 0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i+batch_size].to(device)); yy = y[i:i+batch_size].to(device)
        total_loss += F.cross_entropy(logits, yy, reduction="sum").item()
        total_correct += (logits.argmax(-1) == yy).sum().item()
    return total_loss / len(x), total_correct / len(x)


def norms(model):
    w2 = sum((p.detach() ** 2).sum() for p in model.parameters())
    g2 = sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
    return float(torch.sqrt(w2).cpu()), float(torch.sqrt(g2).cpu())


def make_random_projections(model, count: int, seed: int):
    """Fixed random directions; cheap scalar observers for subsequent EDM."""
    result = []
    for r in range(count):
        g = torch.Generator(device="cpu").manual_seed(seed * 1009 + r)
        dirs = {}
        norm2 = 0.0
        for name, p in model.named_parameters():
            d = torch.randn(p.shape, generator=g, dtype=torch.float32)
            dirs[name] = d; norm2 += float((d * d).sum())
        scale = math.sqrt(norm2)
        result.append({name: d / scale for name, d in dirs.items()})
    return result


@torch.no_grad()
def parameter_projection(model, direction):
    return sum(float((p.detach().cpu().float() * direction[name]).sum())
               for name, p in model.named_parameters())


def gradient_projection(model, direction):
    return sum(float((p.grad.detach().cpu().float() * direction[name]).sum())
               for name, p in model.named_parameters() if p.grad is not None)


@torch.no_grad()
def rich_diagnostics(model, x, y, device, previous_parameters):
    """Extra generic 1-D traces designed for cheap EDM experiments."""
    model.eval(); logits = model(x[:min(2048, len(x))].to(device))
    probs = logits.softmax(-1); top2 = logits.topk(2, dim=-1).values
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean().item()
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    confidence = probs.max(-1).values.mean().item()
    delta2 = delta4 = 0.0; current = {}
    layer_norms = {}
    for name, p in model.named_parameters():
        q = p.detach().cpu().float(); current[name] = q.clone()
        layer = name.rsplit(".", 1)[0]
        layer_norms[layer] = layer_norms.get(layer, 0.0) + float((q*q).sum())
        if previous_parameters is not None:
            d = q - previous_parameters[name]; delta2 += float((d*d).sum()); delta4 += float((d**4).sum())
    result = {"train_predictive_entropy": entropy, "train_logit_margin": margin,
              "train_mean_confidence": confidence}
    for layer, value in layer_norms.items():
        result["layer_weight_norm__" + layer.replace(".", "__")] = math.sqrt(value)
    result["parameter_displacement_norm"] = math.sqrt(delta2) if previous_parameters is not None else 0.0
    # Participation ratio: effective number of coordinates in the displacement,
    # not an intrinsic dimension estimate, but a useful complementary observer.
    result["parameter_displacement_participation_ratio"] = (delta2*delta2/delta4) if delta4 > 0 else 0.0
    return result, current


def run_one(cfg: Config, seed: int, device: torch.device):
    set_seed(seed)
    run_dir = Path(cfg.output_root) / cfg.protocol_name / f"p_{cfg.p}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.force_restart:
        for name in ("training_log.csv", "summary.json", "checkpoint.pt"):
            (run_dir / name).unlink(missing_ok=True)
    train_x, train_y, val_x, val_y = make_data(cfg, seed)
    model = ModularAdditionTransformer(cfg.p, cfg.d_model, cfg.d_hidden, cfg.n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            betas=cfg.betas, weight_decay=cfg.weight_decay)
    projections = make_random_projections(model, max(cfg.projection_count, cfg.loss_projection_count), seed)
    writer = SummaryWriter(str(run_dir / "tensorboard")) if SummaryWriter else None
    logs, memo_step, gen_step = [], None, None
    previous_log_parameters = None
    train_streak = val_streak = 0
    start_time = time.time()
    for step in tqdm(range(1, cfg.max_steps + 1), desc=f"ModAdd p={cfg.p}, seed={seed}"):
        model.train(); opt.zero_grad(set_to_none=True)
        if cfg.batch_size and cfg.batch_size < len(train_x):
            idx = torch.randint(len(train_x), (cfg.batch_size,))
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)
        else:
            xb, yb = train_x.to(device), train_y.to(device)
        loss = F.cross_entropy(model(xb), yb); loss.backward(); wnorm, gnorm = norms(model); opt.step()
        if step % cfg.log_every != 0 and step != 1: continue
        tr_loss, tr_acc = evaluate(model, train_x, train_y, device)
        va_loss, va_acc = evaluate(model, val_x, val_y, device)
        row = dict(step=step, train_loss=tr_loss, train_acc=tr_acc,
                   val_loss=va_loss, val_acc=va_acc, weight_norm=wnorm,
                   grad_norm=gnorm, elapsed_seconds=time.time()-start_time)
        # Fixed random weight/gradient projections. The latter are directional
        # derivatives of the current mini/full-batch loss.
        for r in range(cfg.projection_count):
            row[f"weight_projection_r{r}"] = parameter_projection(model, projections[r])
        for r in range(cfg.loss_projection_count):
            row[f"loss_directional_derivative_r{r}"] = gradient_projection(model, projections[r])
        extra, previous_log_parameters = rich_diagnostics(model, train_x, train_y, device, previous_log_parameters)
        row.update(extra)
        logs.append(row); pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
        if writer:
            for k, v in row.items():
                if k != "step": writer.add_scalar(k, v, step)
            writer.flush()
        if tr_acc >= cfg.target_train_acc: train_streak += 1
        else: train_streak = 0
        if va_acc >= cfg.target_val_acc: val_streak += 1
        else: val_streak = 0
        if memo_step is None and train_streak >= cfg.patience_logs: memo_step = step
        if gen_step is None and val_streak >= cfg.patience_logs:
            if memo_step is not None and step - memo_step >= cfg.required_gap_steps: gen_step = step
        if step % cfg.checkpoint_every == 0:
            torch.save({"step": step, "model": model.state_dict(), "optimizer": opt.state_dict()}, run_dir / "checkpoint.pt")
        if gen_step is not None and step >= gen_step + cfg.post_grok_steps: break
    summary = dict(completed=True, p=cfg.p, seed=seed, train_pairs=len(train_x), val_pairs=len(val_x),
                   memo_step=memo_step, generalization_step=gen_step,
                   gap_steps=(gen_step-memo_step if memo_step is not None and gen_step is not None else None),
                   final_step=(logs[-1]["step"] if logs else 0),
                   final_train_acc=(logs[-1]["train_acc"] if logs else None),
                   final_val_acc=(logs[-1]["val_acc"] if logs else None),
                   outcome="genuine_grokking" if gen_step is not None else ("memorized_only" if memo_step is not None else "not_memorized"))
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if writer: writer.close()
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-root", default=None)
    ap.add_argument("--p", type=int, default=None); ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(); cfg = Config()
    if args.output_root: cfg.output_root = args.output_root
    if args.p: cfg.p = args.p
    if args.max_steps: cfg.max_steps = args.max_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "Config:", cfg)
    summaries = []
    for seed in cfg.seeds:
        if cfg.retries_until_grok and sum(s["outcome"] == "genuine_grokking" for s in summaries) >= cfg.retries_until_grok: break
        summaries.append(run_one(cfg, seed, device))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__": main()
