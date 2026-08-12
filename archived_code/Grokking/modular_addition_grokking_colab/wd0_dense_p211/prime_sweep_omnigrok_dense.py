"""Omnigrok modular-addition sweep over prime moduli (Colab/Kaggle friendly).

This is the exact 1-layer architecture used by this project's Omnigrok
notebooks: shared token embedding, learned positional embedding, one causal
attention+ReLU-MLP residual block, and unembedding from the '=' position.

For large p, materialising p**2 pairs and doing full-batch updates is not
feasible.  Therefore each run uses a fixed, disjoint subset and mini-batches;
the cap and realised fraction are written to metadata (never hidden).
"""
from __future__ import annotations

import csv, gc, hashlib, json, math, random, shutil, time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


DEFAULT_PRIMES = (101, 113, 211, 307, 431, 607, 857, 1201,
                  1693, 2371, 3323, 4651, 6521, 9151, 9973)


@dataclass
class Config:
    output_root: str = "/content/drive/MyDrive/grokking_prime_sweep"
    protocol_name: str = "omnigrok_prime_sweep_v9_fixed_wd100"
    primes: tuple[int, ...] = DEFAULT_PRIMES
    seeds: tuple[int, ...] = (42,)
    # Upper bound on the population training fraction. The actual v4 split
    # also normalizes equations per output class (see the fields below).
    train_fraction: float = 0.30
    # With a constant fraction, the number of training equations per output
    # class grows linearly with p: about 34 at p=113 but 63 at p=211. That
    # removes the memorization/generalization separation. Cap class exposure
    # at the successful p=113 baseline while retaining train_fraction as the
    # maximum fraction for the smallest tasks.
    normalize_train_examples_per_class: bool = True
    train_examples_per_class: float = 34.0
    max_sampled_pairs: int = 500_000
    d_model: int = 128
    d_mlp: int = 512
    num_heads: int = 4
    d_head: int = 32
    init_scale_multiplier: float = 1.0
    # Optional initialization ablations.  The default keeps the exact project
    # Omnigrok initialization; the delayed regularization schedule below is
    # sufficient to create a clean phase separation without changing it.
    init_scale_by_p: dict[int, float] = field(
        default_factory=dict
    )
    # Optional Omnigrok delay control. Keep 1.0 for the faithful baseline;
    # initialization-scale ablations must use a separate protocol name.
    # This deliberately matches generator_logs_to_flat_grokking_with_stochastic.ipynb.
    batch_size: int = 256
    batch_size_by_p: dict[int, int] = field(
        default_factory=lambda: {211: 512}
    )
    # T4 has very poor FP64 throughput. FP32 is dramatically faster and is
    # numerically sufficient for this task. Keep fp64 only for explicit
    # reproduction controls.
    model_dtype: str = "float32"
    max_steps: int = 100_000
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    # Explicit per-task settings are resolved and written to config/summary.
    weight_decay_by_p: dict[int, float] = field(
        default_factory=lambda: {211: 1.0, 307: 1.0}
    )
    # Robust large-p protocol.  First fit without shrinkage, keep the already
    # memorized model for a short, real optimization interval, and only then
    # enable AdamW shrinkage that drives the usual Fourier generalization.
    # This is an explicit regularization schedule, not hidden relabelling of
    # phases: the effective decay is logged at every point.
    delayed_weight_decay: bool = False
    pre_memorization_weight_decay: float = 0.0
    weight_decay_hold_steps: int = 8_000
    weight_decay_hold_steps_by_p: dict[int, int] = field(
        default_factory=lambda: {211: 8_000, 307: 10_000}
    )
    betas: tuple[float, float] = (0.9, 0.98)
    # The working project notebook logged every 10 updates. Coarser logging can
    # make two separate transitions look simultaneous.
    log_every: int = 50
    diagnostic_every: int = 500
    checkpoint_every: int = 5_000
    tensorboard_enabled: bool = True
    tensorboard_flush_secs: int = 5
    tensorboard_histogram_every: int = 5_000
    text_log_enabled: bool = True
    text_log_filename: str = "training.log"
    text_log_every: int = 1_000
    # Dense mode appends CSV rows instead of rewriting the whole file.
    # This is required for log_every=1 runs lasting hundreds of thousands of steps.
    stream_csv: bool = False
    csv_flush_rows: int = 100
    # 8192+8192 monitor examples every 10 steps dominated wall time. A fixed
    # 2048+2048 monitor is already precise to about 2 percentage points and
    # reduces diagnostic work by roughly 20x together with log_every=50.
    monitor_train_pairs: int = 2_048
    monitor_val_pairs: int = 2_048
    eval_batch_size: int = 2_048
    projection_count: int = 3
    target_train_acc: float = 0.99
    target_val_acc: float = 0.95
    patience_logs: int = 5
    required_gap_steps: int = 5_000
    post_grok_steps: int = 5_000
    stop_after_first_success: bool = False
    force_restart: bool = False
    # If force_restart=False, automatically continue from the checkpoint with
    # the largest saved step (checkpoint.pt or checkpoint_final.pt).
    resume_from_checkpoint: bool = True
    device: str = "auto"
    fused_adamw: bool = True
    # Colab is unlikely to finish all 15 in one session. Set this to, e.g.,
    # (101, 307, 857) for a pilot, then resume with the remaining values.
    skip_completed: bool = True


class TextRunLogger:
    """Line-buffered UTF-8 log that is visible while a Colab run is active."""
    def __init__(self, path: Path, enabled: bool = True, append: bool = False):
        self.handle = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a" if append else "w", encoding="utf-8", buffering=1)

    def log(self, message: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()


def is_prime(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0: return False
        d += 2
    return True


class Embed(nn.Module):
    def __init__(self, vocab: int, d_model: int, scale: float):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, vocab) * scale / math.sqrt(d_model))
    def forward(self, x): return self.W_E[:, x].permute(1, 2, 0)


class PosEmbed(nn.Module):
    def __init__(self, n_ctx: int, d_model: int, scale: float):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) * scale / math.sqrt(d_model))
    def forward(self, x): return x + self.W_pos[:x.shape[1]]


class Attention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_head: int, n_ctx: int, scale: float):
        super().__init__(); std = scale / math.sqrt(d_model)
        self.W_K = nn.Parameter(torch.randn(heads, d_head, d_model) * std)
        self.W_Q = nn.Parameter(torch.randn(heads, d_head, d_model) * std)
        self.W_V = nn.Parameter(torch.randn(heads, d_head, d_model) * std)
        self.W_O = nn.Parameter(torch.randn(d_model, heads * d_head) * std)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)), persistent=False)
        self.d_head = d_head
    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        scores = torch.einsum("biph,biqh->biqp", k, q) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.mask[:x.shape[1], :x.shape[1]].eq(0), -1e10)
        z = torch.einsum("biph,biqp->biqh", v, scores.softmax(-1))
        z = z.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return torch.einsum("df,bqf->bqd", self.W_O, z)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, scale: float):
        super().__init__(); std = scale / math.sqrt(d_model)
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) * std)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) * std)
        self.b_out = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        h = F.relu(torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in)
        return torch.einsum("dm,bpm->bpd", self.W_out, h) + self.b_out


class OmnigrokTransformer(nn.Module):
    """1L Transformer matching grokking_model.py (no LayerNorm/dropout)."""
    def __init__(self, p: int, cfg: Config):
        super().__init__(); vocab = p + 1; s = cfg.init_scale_multiplier
        self.p = p
        self.embed = Embed(vocab, cfg.d_model, s)
        self.pos_embed = PosEmbed(3, cfg.d_model, s)
        self.attn = Attention(cfg.d_model, cfg.num_heads, cfg.d_head, 3, s)
        self.mlp = MLP(cfg.d_model, cfg.d_mlp, s)
        self.W_U = nn.Parameter(torch.randn(cfg.d_model, vocab) * s / math.sqrt(vocab))
    def hidden(self, x):
        h = self.pos_embed(self.embed(x)); h = h + self.attn(h); h = h + self.mlp(h)
        return h
    def forward(self, x, return_hidden=False):
        h = self.hidden(x); logits = h[:, -1] @ self.W_U
        # Preserve the exact Omnigrok head: it includes the '=' token as an
        # extra (never-targeted) class instead of slicing it away.
        return (logits, h) if return_hidden else logits


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def select_device(name: str):
    if name == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def select_dtype(name: str):
    aliases = {"float64": torch.float64, "double": torch.float64,
               "float32": torch.float32, "float": torch.float32}
    if name not in aliases:
        raise ValueError(f"Unsupported model_dtype={name!r}; choose float32 or float64")
    return aliases[name]


def make_fixed_data(p: int, cfg: Config, seed: int):
    total = p * p
    sampled = min(total, cfg.max_sampled_pairs)
    fraction_target = int(round(total * cfg.train_fraction))
    if cfg.normalize_train_examples_per_class:
        exposure_target = int(round(p * cfg.train_examples_per_class))
        requested_train_n = min(fraction_target, exposure_target)
        split_policy = "class_exposure_normalized"
    else:
        requested_train_n = fraction_target
        split_policy = "constant_population_fraction"
    train_n = min(sampled-1, max(1, requested_train_n))
    val_n = sampled-train_n
    if val_n < 1: raise ValueError("validation set is empty")
    # Match get_modular_addition_data exactly whenever the complete table fits:
    # torch.manual_seed(seed) + torch.randperm(total), rather than Python RNG.
    # For a capped large-p universe, randperm(total) would be too expensive, so
    # the fallback samples unique pair ids without materialising p**2 entries.
    if sampled == total:
        # set_seed(seed) was called immediately before this function. Using the
        # global generator also reproduces its effect on subsequent model init.
        ids = torch.randperm(total)
        split_method = "torch_randperm_full_population"
    else:
        ids = torch.tensor(random.Random(seed).sample(range(total), sampled), dtype=torch.long)
        split_method = "python_sample_capped_population"
    a, b = ids.div(p, rounding_mode="floor"), ids.remainder(p)
    x = torch.stack([a, b, torch.full_like(a, p)], 1)
    y = (a + b).remainder(p)
    meta = dict(total_pairs=total, sampled_pairs=sampled, train_pairs=train_n, val_pairs=val_n,
                requested_train_fraction=cfg.train_fraction,
                split_train_fraction=train_n / sampled,
                population_train_fraction=train_n / total,
                train_examples_per_class=train_n / p,
                split_policy=split_policy,
                population_coverage=sampled/total,
                universe_is_sampled=sampled<total,
                split_method=split_method)
    return x[:train_n], y[:train_n], x[train_n:], y[train_n:], meta


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**31)


class LayerProjector:
    """Fixed unit Rademacher directions for every parameter tensor."""
    def __init__(self, model: nn.Module, count: int, seed: int, device):
        self.directions = {}
        for name, p in model.named_parameters():
            dirs = []
            for r in range(count):
                g = torch.Generator(device="cpu").manual_seed(seed + stable_seed(name) + 104729*r)
                d = torch.randint(0, 2, p.shape, generator=g, dtype=torch.float32).mul_(2).sub_(1)
                d.div_(math.sqrt(p.numel())); dirs.append(d.to(device))
            self.directions[name] = dirs
    @torch.no_grad()
    def values(self, model, gradients=False, prefix=None):
        out = {}
        for name, p in model.named_parameters():
            value = p.grad if gradients else p
            if value is None: continue
            key = name.replace(".", "__")
            for r, d in enumerate(self.directions[name]):
                out[f"{prefix or ('gradproj' if gradients else 'weightproj')}__{key}__r{r}"] = float((value.float()*d).sum())
        return out
    @torch.no_grad()
    def tensor_values(self, model, tensors, prefix):
        """Project an update/displacement aligned with model.parameters()."""
        out = {}
        for (name, _), value in zip(model.named_parameters(), tensors):
            key = name.replace(".", "__")
            for r, d in enumerate(self.directions[name]):
                out[f"{prefix}__{key}__r{r}"] = float((value.float()*d).sum())
        return out


def loss_signs(n: int, count: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, count), generator=g, dtype=torch.float32).mul_(2).sub_(1)


@torch.inference_mode()
def evaluate(model, x, y, signs, cfg, device, activation_dirs=None, logit_dirs=None):
    model.eval(); loss_sum = correct = 0; projected = torch.zeros(signs.shape[1], dtype=torch.float64)
    act_proj = torch.zeros(signs.shape[1], dtype=torch.float64)
    logit_proj = torch.zeros(signs.shape[1], dtype=torch.float64)
    entropy_sum = margin_sum = confidence_sum = hidden_norm_sum = 0.0
    n = len(y)
    for start in range(0, n, cfg.eval_batch_size):
        stop = min(n, start + cfg.eval_batch_size)
        logits, hidden = model(x[start:stop].to(device), return_hidden=True)
        yy = y[start:stop].to(device); losses = F.cross_entropy(logits, yy, reduction="none")
        probs = logits.softmax(-1); top2 = logits.topk(2, -1).values
        loss_sum += losses.sum().item(); correct += (logits.argmax(-1) == yy).sum().item()
        entropy_sum += float((-(probs * probs.clamp_min(1e-12).log()).sum(-1)).sum())
        margin_sum += float((top2[:, 0]-top2[:, 1]).sum()); confidence_sum += float(probs.max(-1).values.sum())
        hidden_norm_sum += float(hidden[:, -1].norm(dim=-1).sum())
        ss = signs[start:stop].to(device)
        projected += (losses[:, None]*ss).sum(0).double().cpu()
        if activation_dirs is not None:
            for r, d in enumerate(activation_dirs):
                # Fixed random direction across both examples and features.
                per_example = (hidden[:, -1] * d).sum(-1)
                act_proj[r] += float((per_example * ss[:, r]).sum())
        if logit_dirs is not None:
            for r, d in enumerate(logit_dirs):
                # Factorised fixed example/class random projection.
                logit_proj[r] += float((logits*d*ss[:, r:r+1]).sum())
    scale = math.sqrt(n)
    result = dict(loss=loss_sum/n, acc=correct/n, entropy=entropy_sum/n,
                  margin=margin_sum/n, confidence=confidence_sum/n,
                  hidden_norm=hidden_norm_sum/n,
                  lossproj=(projected/scale).numpy())
    result["activationproj"] = (act_proj/scale).numpy()
    result["logitproj"] = (logit_proj/math.sqrt(n*(model.p+1))).numpy()
    return result


@torch.no_grad()
def vector_stats(tensors: Iterable[torch.Tensor]):
    l2sq = l4 = 0.0
    for t in tensors:
        q = t.detach().float(); l2sq += float((q*q).sum()); l4 += float((q**4).sum())
    return math.sqrt(l2sq), (l2sq*l2sq/l4 if l4 > 0 else 0.0)


@torch.no_grad()
def cosine(a, b):
    if a is None or b is None: return float("nan")
    dot = aa = bb = 0.0
    for x, y in zip(a, b): dot += float((x*y).sum()); aa += float((x*x).sum()); bb += float((y*y).sum())
    return dot/math.sqrt(aa*bb) if aa > 0 and bb > 0 else float("nan")


@torch.no_grad()
def fourier_stats(matrix: torch.Tensor, p: int):
    # matrix is [features, residue/classes (+ optional '=')]
    z = torch.fft.rfft(matrix[:, :p].float(), dim=1); energy = (z.real**2+z.imag**2).sum(0)
    energy = energy[1:]  # remove DC
    total = float(energy.sum())
    if total <= 0: return 0.0, 0.0
    pr = total*total/float((energy*energy).sum())
    top = float(torch.topk(energy, min(5, len(energy))).values.sum())/total
    return pr, top


def write_logs(run_dir, logs): pd.DataFrame(logs).to_csv(run_dir/"training_log.csv", index=False)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _best_checkpoint(run_dir: Path):
    candidates = []
    for name in ("checkpoint.pt", "checkpoint_final.pt"):
        path = run_dir / name
        if path.exists():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # compatibility with older torch
                payload = torch.load(path, map_location="cpu")
            candidates.append((int(payload.get("step", -1)), path, payload))
    return max(candidates, default=None, key=lambda item: item[0])


def _last_csv_row(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        last = None
        for last in csv.DictReader(handle):
            pass
    return last


def _trim_csv_after_step(csv_path: Path, step: int):
    """Keep one header and rows through checkpoint step; remove partial tail."""
    if not csv_path.exists():
        return
    frame = pd.read_csv(csv_path)
    if "step" not in frame.columns:
        raise ValueError(f"Cannot resume: {csv_path} has no step column")
    frame = frame[pd.to_numeric(frame["step"], errors="coerce") <= step]
    frame = frame.drop_duplicates(subset=["step"], keep="last").sort_values("step")
    frame.to_csv(csv_path, index=False)


class StatefulBatchStream:
    """Shuffle-once-per-epoch stream whose exact position can be checkpointed."""
    def __init__(self, x, y, batch_size: int, seed: int):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = None
        self.cursor = 0

    def next(self):
        if self.order is None or self.cursor >= len(self.y):
            self.order = torch.randperm(len(self.y), generator=self.generator)
            self.cursor = 0
        end = min(self.cursor + self.batch_size, len(self.y))
        indices = self.order[self.cursor:end]
        self.cursor = end
        return self.x[indices], self.y[indices]

    def state_dict(self):
        return {
            "generator_state": self.generator.get_state(),
            "order": self.order,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.generator.set_state(state["generator_state"])
        self.order = state.get("order")
        self.cursor = int(state.get("cursor", 0))


def make_tensorboard_writer(run_dir: Path, cfg: Config, p: int, seed: int,
                            resume: bool = False):
    if not cfg.tensorboard_enabled:
        return None
    if SummaryWriter is None:
        print("WARNING: TensorBoard is unavailable; run: pip install tensorboard")
        return None
    tb_dir = run_dir / "tensorboard"
    # A fresh run gets a clean directory; a resumed run appends a new event
    # file with monotonically increasing global steps.
    if tb_dir.exists() and not resume:
        shutil.rmtree(tb_dir)
    writer = SummaryWriter(
        log_dir=str(tb_dir),
        max_queue=10,
        flush_secs=cfg.tensorboard_flush_secs,
        filename_suffix=f".p{p}.seed{seed}",
    )
    writer.add_text("Run/config", "```json\n" + json.dumps(asdict(cfg), indent=2) + "\n```", 0)
    writer.add_text("Run/identity", f"p={p}, seed={seed}", 0)
    writer.add_custom_scalars({
        "Core": {
            "Loss": ["Multiline", ["Loss/train", "Loss/validation", "Loss/train_batch"]],
            "Accuracy": ["Multiline", ["Accuracy/train", "Accuracy/validation"]],
        },
        "Optimization": {
            "Norms": ["Multiline", ["Norm/weights", "Norm/gradients", "Norm/update", "Norm/displacement"]],
            "Participation ratios": ["Multiline", [
                "Participation_ratio/parameters", "Participation_ratio/gradients",
                "Participation_ratio/update", "Participation_ratio/displacement"]],
        },
        "Predictions": {
            "Entropy": ["Multiline", ["Prediction/train_entropy", "Prediction/val_entropy"]],
            "Margin": ["Multiline", ["Prediction/train_margin", "Prediction/val_margin"]],
            "Confidence": ["Multiline", ["Prediction/train_confidence", "Prediction/val_confidence"]],
        },
    })
    writer.flush()
    print(f"TensorBoard live log: {tb_dir}")
    return writer


def log_tensorboard(writer, row, step: int, model=None, histogram=False):
    """Write stable, grouped TensorBoard tags; CSV keeps the flat schema."""
    if writer is None:
        return
    tags = {
        "Loss/train": "train_loss", "Loss/validation": "val_loss",
        "Loss/train_batch": "train_batch_loss",
        "Accuracy/train": "train_acc", "Accuracy/validation": "val_acc",
        "Norm/weights": "weight_norm", "Norm/gradients": "gradient_norm",
        "Norm/update": "update_norm", "Norm/displacement": "displacement_since_log_norm",
        "Participation_ratio/parameters": "parameter_participation_ratio",
        "Participation_ratio/gradients": "gradient_participation_ratio",
        "Participation_ratio/update": "update_participation_ratio",
        "Participation_ratio/displacement": "displacement_since_log_participation_ratio",
        "Optimization/gradient_cosine": "gradient_cosine",
        "Optimization/learning_rate": "learning_rate",
        "Optimization/weight_decay": "weight_decay",
        "Prediction/train_entropy": "train_entropy", "Prediction/val_entropy": "val_entropy",
        "Prediction/train_margin": "train_margin", "Prediction/val_margin": "val_margin",
        "Prediction/train_confidence": "train_confidence", "Prediction/val_confidence": "val_confidence",
        "Representation/train_hidden_norm": "train_hidden_norm",
        "Representation/val_hidden_norm": "val_hidden_norm",
        "Progress/epoch_equivalent": "epoch_equivalent",
        "Progress/examples_seen": "examples_seen", "Progress/elapsed_seconds": "elapsed_seconds",
        "Fourier/embedding_effective_frequencies": "embedding_fourier_effective_frequencies",
        "Fourier/embedding_top5_energy": "embedding_fourier_top5_energy",
        "Fourier/unembedding_effective_frequencies": "unembedding_fourier_effective_frequencies",
        "Fourier/unembedding_top5_energy": "unembedding_fourier_top5_energy",
    }
    for tag, key in tags.items():
        value = row.get(key)
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            writer.add_scalar(tag, float(value), step)
    for key, value in row.items():
        if not (isinstance(value, (int, float, np.number)) and np.isfinite(value)):
            continue
        if key.startswith("train_lossproj_r"):
            tag = "Projection/loss/train/" + key.rsplit("_", 1)[-1]
        elif key.startswith("val_lossproj_r"):
            tag = "Projection/loss/validation/" + key.rsplit("_", 1)[-1]
        elif key.startswith("train_activationproj_r"):
            tag = "Projection/activation/train/" + key.rsplit("_", 1)[-1]
        elif key.startswith("val_activationproj_r"):
            tag = "Projection/activation/validation/" + key.rsplit("_", 1)[-1]
        elif key.startswith("train_logitproj_r"):
            tag = "Projection/logits/train/" + key.rsplit("_", 1)[-1]
        elif key.startswith("val_logitproj_r"):
            tag = "Projection/logits/validation/" + key.rsplit("_", 1)[-1]
        elif key.startswith(("weightproj__", "gradproj__", "updateproj__", "displacementproj__")):
            tag = "Projection/parameters/" + key.replace("__", "/")
        else:
            continue
        writer.add_scalar(tag, float(value), step)
    if histogram and model is not None:
        for name, parameter in model.named_parameters():
            safe = name.replace(".", "/")
            writer.add_histogram(f"Histogram/weights/{safe}", parameter.detach().float().cpu(), step)
            if parameter.grad is not None:
                writer.add_histogram(f"Histogram/gradients/{safe}", parameter.grad.detach().float().cpu(), step)
    writer.flush()


def train_one(cfg: Config, p: int, seed: int):
    if not is_prime(p): raise ValueError(f"{p} is not prime")
    device = select_device(cfg.device); set_seed(seed)
    run_dir = Path(cfg.output_root)/cfg.protocol_name/f"p_{p}"/f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir/"COMPLETED.json"
    resume_candidate = (None if cfg.force_restart or not cfg.resume_from_checkpoint
                        else _best_checkpoint(run_dir))
    if done.exists() and cfg.skip_completed and not cfg.force_restart:
        completed = json.loads(done.read_text())
        completed_step = int(completed.get("final_step", -1))
        if completed_step >= cfg.max_steps:
            print("Skipping completed", run_dir); return completed
        if resume_candidate is None:
            print("Skipping completed run without a checkpoint", run_dir); return completed
        print(f"Extending completed run from step {resume_candidate[0]:,} "
              f"to {cfg.max_steps:,}")
    if cfg.force_restart and done.exists():
        done.unlink()
    # Resolve all task-specific controls once.  The resolved values are used
    # everywhere below and saved separately, so the run is fully auditable.
    run_init_scale = cfg.init_scale_by_p.get(p, cfg.init_scale_multiplier)
    run_batch_size = cfg.batch_size_by_p.get(p, cfg.batch_size)
    run_weight_decay = cfg.weight_decay_by_p.get(p, cfg.weight_decay)
    run_hold_steps = cfg.weight_decay_hold_steps_by_p.get(p, cfg.weight_decay_hold_steps)
    run_cfg = replace(cfg, init_scale_multiplier=run_init_scale,
                      batch_size=run_batch_size, weight_decay=run_weight_decay,
                      weight_decay_hold_steps=run_hold_steps)
    xtr, ytr, xva, yva, data_meta = make_fixed_data(p, run_cfg, seed)
    if run_batch_size <= 0:
        raise ValueError("This protocol is mini-batch only: batch_size must be positive")
    effective_batch_size = min(run_batch_size, len(ytr))
    full_batch_training = False
    data_meta.update(effective_batch_size=effective_batch_size,
                     full_batch_training=full_batch_training)
    model_dtype = select_dtype(run_cfg.model_dtype)
    # Construct directly in the requested dtype rather than constructing in
    # float32 and casting afterwards. This reproduces the initialization (and
    # RNG consumption) of the successful notebook, which globally used fp64.
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(model_dtype)
    try:
        model = OmnigrokTransformer(p, run_cfg)
    finally:
        torch.set_default_dtype(previous_default_dtype)
    model = model.to(device=device)
    initial_decay = (run_cfg.pre_memorization_weight_decay
                     if run_cfg.delayed_weight_decay else run_weight_decay)
    optimizer_kwargs = dict(lr=run_cfg.learning_rate,
                            weight_decay=initial_decay, betas=run_cfg.betas)
    if run_cfg.fused_adamw and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        opt = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        # Older Colab torch builds have no fused AdamW implementation.
        optimizer_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    start_step = 1
    resume_payload = None
    if resume_candidate is not None:
        checkpoint_step, checkpoint_path, resume_payload = resume_candidate
        if checkpoint_step >= run_cfg.max_steps:
            print(f"Checkpoint already reached step {checkpoint_step:,}; "
                  f"max_steps={run_cfg.max_steps:,}")
            if done.exists():
                return json.loads(done.read_text())
            raise ValueError("Increase max_steps to continue this checkpoint")
        model.load_state_dict(resume_payload["model"])
        opt.load_state_dict(resume_payload["optimizer"])
        # Optimizer tensors must live on the same device as the model.
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_step = checkpoint_step + 1
    projector = LayerProjector(model, run_cfg.projection_count, seed+90001, device)
    mt = min(len(ytr), run_cfg.monitor_train_pairs); mv = min(len(yva), run_cfg.monitor_val_pairs)
    # Monitoring examples and projections remain identical throughout training.
    train_x, train_y = xtr[:mt], ytr[:mt]; val_x, val_y = xva[:mv], yva[:mv]
    tr_signs = loss_signs(mt, run_cfg.projection_count, seed+1001)
    va_signs = loss_signs(mv, run_cfg.projection_count, seed+2001)
    activation_dirs=[]; logit_dirs=[]
    for r in range(run_cfg.projection_count):
        g=torch.Generator().manual_seed(seed+3001+r)
        activation_dirs.append(torch.randint(0,2,(run_cfg.d_model,),generator=g,dtype=torch.float32).mul_(2).sub_(1).div_(math.sqrt(run_cfg.d_model)).to(device))
        logit_dirs.append(torch.randint(0,2,(p+1,),generator=g,dtype=torch.float32).mul_(2).sub_(1).to(device))
    is_resuming = resume_payload is not None
    writer = make_tensorboard_writer(run_dir, run_cfg, p, seed, resume=is_resuming)
    text_log = TextRunLogger(run_dir/run_cfg.text_log_filename,
                             run_cfg.text_log_enabled, append=is_resuming)
    text_log.log(
        f"START p={p} seed={seed} device={device} dtype={run_cfg.model_dtype} "
        f"train_pairs={len(ytr)} val_pairs={len(yva)} batch_size={effective_batch_size} "
        f"split_policy={data_meta['split_policy']} "
        f"train_examples_per_class={data_meta['train_examples_per_class']:.3f} "
        f"population_train_fraction={data_meta['population_train_fraction']:.6f} "
        f"lr={run_cfg.learning_rate:g} init_scale={run_init_scale:g} "
        f"target_weight_decay={run_weight_decay:g} initial_weight_decay={initial_decay:g} "
        f"delayed_weight_decay={run_cfg.delayed_weight_decay} hold_steps={run_hold_steps} "
        f"required_gap_steps={run_cfg.required_gap_steps} post_grok_steps={run_cfg.post_grok_steps} "
        f"resume={is_resuming} start_step={start_step}"
    )
    logs=[] if not run_cfg.stream_csv else None; last_row=None; pending_rows=[]; csv_columns=None
    csv_path = run_dir / "training_log.csv"
    if run_cfg.stream_csv and run_cfg.force_restart and csv_path.exists():
        csv_path.unlink()
    memo_step=gen_step=first_generalization_step=None
    grok_detected_at_step=planned_stop_step=None
    memo_streak=gen_streak=0; previous_grad=None; previous_logged_params=None
    current_phase = "fitting"
    decay_activation_step = None

    if is_resuming:
        checkpoint_step = start_step - 1
        _trim_csv_after_step(csv_path, checkpoint_step)
        train_state = resume_payload.get("train_state", {})
        old_summary = {}
        if done.exists():
            try:
                old_summary = json.loads(done.read_text(encoding="utf-8"))
            except Exception:
                pass
        memo_step = train_state.get("memo_step", old_summary.get("memo_step"))
        gen_step = train_state.get("gen_step", old_summary.get("generalization_step"))
        first_generalization_step = train_state.get(
            "first_generalization_step", old_summary.get("first_generalization_step"))
        grok_detected_at_step = train_state.get("grok_detected_at_step")
        planned_stop_step = train_state.get("planned_stop_step")
        memo_streak = int(train_state.get("memo_streak", 0))
        gen_streak = int(train_state.get("gen_streak", 0))
        current_phase = train_state.get("current_phase", "fitting")
        decay_activation_step = train_state.get("decay_activation_step")
        examples_seen = int(train_state.get("examples_seen", 0))
        previous_grad = train_state.get("previous_grad")
        previous_logged_params = train_state.get("previous_logged_params")
        if previous_grad is not None:
            previous_grad = [x.to(device) for x in previous_grad]
        if previous_logged_params is not None:
            previous_logged_params = [x.to(device) for x in previous_logged_params]
        last_csv = _last_csv_row(csv_path)
        if last_csv:
            if not examples_seen:
                examples_seen = int(float(last_csv.get("examples_seen", 0)))
            current_phase = train_state.get("current_phase",
                                            last_csv.get("phase", current_phase))
            if not train_state.get("elapsed_seconds"):
                train_state["elapsed_seconds"] = float(last_csv.get("elapsed_seconds", 0))
        # Keep the normalized/fallback state in the payload as well.  This is
        # important for old checkpoints which did not contain ``train_state``:
        # elapsed time recovered from CSV must still be carried into new rows.
        resume_payload["train_state"] = train_state
        if csv_path.exists():
            csv_columns = list(pd.read_csv(csv_path, nrows=0).columns)
        text_log.log(
            f"RESUME checkpoint_step={checkpoint_step} next_step={start_step} "
            f"csv_last_step={last_csv.get('step') if last_csv else None} "
            f"memo_step={memo_step} gen_step={gen_step} phase={current_phase}"
        )
    else:
        examples_seen = 0

    def set_phase(new_phase: str, step: int, reason: str):
        nonlocal current_phase
        if new_phase != current_phase:
            text_log.log(
                f"PHASE step={step} from={current_phase} to={new_phase} reason={reason}"
            )
            current_phase = new_phase
    # Faithful shuffle-once-per-epoch mini-batches, now with checkpointable
    # permutation/cursor state for exact continuation.
    batch_stream = StatefulBatchStream(xtr, ytr, effective_batch_size, seed + 70001)
    if is_resuming:
        batch_stream.load_state_dict(resume_payload.get("batch_stream"))
        _restore_rng_state(resume_payload.get("rng_state"))
    start_time = time.time()
    elapsed_before_resume = float(
        resume_payload.get("train_state", {}).get("elapsed_seconds", 0.0)
        if is_resuming else 0.0
    )
    progress = tqdm(range(start_step, run_cfg.max_steps + 1),
                    total=run_cfg.max_steps, initial=start_step - 1,
                    desc=f"Omnigrok mod {p}, seed {seed}")
    for step in progress:
        # Activate the simplification pressure only after the network has had
        # an explicit memorization-only interval.  Changing param_group here
        # preserves Adam moments and is much cheaper than rebuilding training.
        if (run_cfg.delayed_weight_decay and decay_activation_step is not None and
                step == decay_activation_step):
            for group in opt.param_groups:
                group["weight_decay"] = run_weight_decay
            # Reset hits collected during the decay hold; detection starts post-activation.
            gen_streak = 0
            text_log.log(
                f"EVENT weight_decay_activated step={step} weight_decay={run_weight_decay:g}"
            )
        batch_x, batch_y = batch_stream.next()
        examples_seen += len(batch_y)
        model.train(); opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(batch_x.to(device)), batch_y.to(device)); loss.backward()
        # The planned stop becomes an additional logging point, so a
        # post-grok tail need not be divisible by log_every and is still saved
        # at exactly the requested optimizer step.
        is_log = (step==1 or step%run_cfg.log_every==0 or
                  (planned_stop_step is not None and step>=planned_stop_step))
        if is_log:
            current_grad=[q.grad.detach().float().clone() for q in model.parameters()]
            grad_norm, grad_pr=vector_stats(current_grad); grad_cos=cosine(previous_grad,current_grad); previous_grad=current_grad
            gradproj=projector.values(model,gradients=True)
            pre=[q.detach().float().clone() for q in model.parameters()]
        opt.step()
        if not is_log: continue
        post=[q.detach().float() for q in model.parameters()]; updates=[b-a for a,b in zip(pre,post)]
        weight_norm, weight_pr=vector_stats(post); update_norm, update_pr=vector_stats(updates)
        displacement = [torch.zeros_like(q) for q in post] if previous_logged_params is None else [q-z for q,z in zip(post,previous_logged_params)]
        displacement_norm, displacement_pr=vector_stats(displacement); previous_logged_params=[q.clone() for q in post]
        tr=evaluate(model,train_x,train_y,tr_signs,run_cfg,device,activation_dirs,logit_dirs)
        va=evaluate(model,val_x,val_y,va_signs,run_cfg,device,activation_dirs,logit_dirs)
        effective_batch=len(batch_y)
        row=dict(step=step,examples_seen=examples_seen,epoch_equivalent=examples_seen/len(ytr),
                 train_batch_loss=float(loss.detach()),train_loss=tr["loss"],train_acc=tr["acc"],
                 val_loss=va["loss"],val_acc=va["acc"],weight_norm=weight_norm,
                 learning_rate=opt.param_groups[0]["lr"],
                 weight_decay=opt.param_groups[0]["weight_decay"],
                 target_weight_decay=run_weight_decay,
                 gradient_norm=grad_norm,gradient_cosine=grad_cos,gradient_participation_ratio=grad_pr,
                 update_norm=update_norm,update_participation_ratio=update_pr,
                 displacement_since_log_norm=displacement_norm,
                 displacement_since_log_participation_ratio=displacement_pr,
                 train_entropy=tr["entropy"],val_entropy=va["entropy"],
                 train_margin=tr["margin"],val_margin=va["margin"],
                 train_confidence=tr["confidence"],val_confidence=va["confidence"],
                 train_hidden_norm=tr["hidden_norm"],val_hidden_norm=va["hidden_norm"],
                 parameter_participation_ratio=weight_pr,
                 elapsed_seconds=elapsed_before_resume+time.time()-start_time)
        for r in range(run_cfg.projection_count):
            row[f"train_lossproj_r{r}"]=tr["lossproj"][r]; row[f"val_lossproj_r{r}"]=va["lossproj"][r]
            row[f"train_activationproj_r{r}"]=tr["activationproj"][r]; row[f"val_activationproj_r{r}"]=va["activationproj"][r]
            row[f"train_logitproj_r{r}"]=tr["logitproj"][r]; row[f"val_logitproj_r{r}"]=va["logitproj"][r]
        row.update(gradproj); row.update(projector.values(model,gradients=False))
        row.update(projector.tensor_values(model,updates,"updateproj"))
        row.update(projector.tensor_values(model,displacement,"displacementproj"))
        if step==1 or step%run_cfg.diagnostic_every==0:
            epr,etop=fourier_stats(model.embed.W_E,p); upr,utop=fourier_stats(model.W_U,p)
            row.update(embedding_fourier_effective_frequencies=epr,embedding_fourier_top5_energy=etop,
                       unembedding_fourier_effective_frequencies=upr,unembedding_fourier_top5_energy=utop)
        memo_streak = memo_streak+1 if tr["acc"]>=run_cfg.target_train_acc else 0
        gen_streak = gen_streak+1 if va["acc"]>=run_cfg.target_val_acc else 0
        if memo_step is None and memo_streak>=run_cfg.patience_logs:
            memo_step=step-(run_cfg.patience_logs-1)*run_cfg.log_every
            text_log.log(
                f"EVENT stable_memorization onset_step={memo_step} detected_at_step={step}"
            )
            if first_generalization_step is None:
                set_phase("memorization", step, "stable_train_accuracy")
            if run_cfg.delayed_weight_decay and decay_activation_step is None:
                decay_activation_step = step + run_hold_steps
                text_log.log(
                    f"EVENT memorization_hold_started step={step} "
                    f"decay_activation_step={decay_activation_step} hold_steps={run_hold_steps}"
                )
        # With delayed decay enabled, do not call a high validation score
        # "generalization" during the intentional memorization-only hold.
        # Otherwise p=211 can be reported as having no gap even though the
        # schedule has not yet applied the grokking-inducing decay.
        generalization_allowed = (
            not run_cfg.delayed_weight_decay
            or (decay_activation_step is not None and step >= decay_activation_step)
        )
        if (first_generalization_step is None and gen_streak >= run_cfg.patience_logs
                and generalization_allowed):
            # Estimate onset, rather than the final point of the patience streak.
            first_generalization_step = step-(run_cfg.patience_logs-1)*run_cfg.log_every
            observed_gap = (first_generalization_step-memo_step
                            if memo_step is not None else None)
            text_log.log(
                f"EVENT stable_generalization onset_step={first_generalization_step} "
                f"detected_at_step={step} observed_gap={observed_gap}"
            )
            if memo_step is not None and observed_gap >= run_cfg.required_gap_steps:
                gen_step=first_generalization_step
                grok_detected_at_step=step
                # Count the requested tail from detection, not from the onset
                # reconstructed from the patience window. Thus the run always
                # records exactly post_grok_steps of new optimization data.
                planned_stop_step=min(run_cfg.max_steps, step+run_cfg.post_grok_steps)
                text_log.log(
                    f"EVENT genuine_grokking grok_step={gen_step} gap_steps={observed_gap} "
                    f"detected_at_step={grok_detected_at_step} "
                    f"planned_stop_step={planned_stop_step}"
                )
                set_phase("post_grok", step, "genuine_grokking_detected")
            else:
                text_log.log(
                    "EVENT early_generalization run_not_classified_as_grokking; "
                    f"required_gap_steps={run_cfg.required_gap_steps}"
                )
                set_phase("early_generalization", step, "gap_too_short_or_no_memorization")
        remaining = (max(0, planned_stop_step-step)
                     if planned_stop_step is not None else None)
        row.update(
            stable_memorization=int(memo_step is not None),
            stable_generalization=int(first_generalization_step is not None),
            genuine_grokking=int(gen_step is not None),
            phase=current_phase,
            phase_id={"fitting": 0, "memorization": 1,
                      "early_generalization": 2, "post_grok": 3}[current_phase],
            post_grok_remaining_steps=remaining,
        )
        last_row = row
        if run_cfg.stream_csv:
            if csv_columns is None:
                csv_columns = list(row.keys())
            pending_rows.append(row)
            if len(pending_rows) >= max(1, run_cfg.csv_flush_rows):
                pd.DataFrame(pending_rows, columns=csv_columns).to_csv(
                    csv_path, mode="a", header=not csv_path.exists(), index=False
                )
                pending_rows.clear()
        else:
            logs.append(row); write_logs(run_dir, logs)
        histogram = (step == 1 or step % run_cfg.tensorboard_histogram_every == 0)
        log_tensorboard(writer, row, step, model=model, histogram=histogram)
        # CSV/TensorBoard retain the dense cadence (log_every) for EDM.
        # Human-readable metric lines are deliberately much sparser.
        should_text_metric = (
            step == 1 or step % run_cfg.text_log_every == 0 or
            (planned_stop_step is not None and step >= planned_stop_step)
        )
        if should_text_metric:
            text_log.log(
                f"METRIC p={p} seed={seed} step={step} phase={current_phase} "
                f"epoch={row['epoch_equivalent']:.3f} "
                f"batch_loss={row['train_batch_loss']:.6g} "
                f"train_loss={tr['loss']:.6g} train_acc={tr['acc']:.6f} "
                f"val_loss={va['loss']:.6g} val_acc={va['acc']:.6f} "
                f"weight_decay={opt.param_groups[0]['weight_decay']:.6g} "
                f"weight_norm={weight_norm:.6g} grad_norm={grad_norm:.6g} "
                f"update_norm={update_norm:.6g} elapsed_s={row['elapsed_seconds']:.1f}" +
                (f" post_grok_remaining={remaining}" if remaining is not None else "")
            )
        if writer:
            writer.add_scalar("Events/memorized", float(memo_step is not None), step)
            writer.add_scalar("Events/generalized", float(first_generalization_step is not None), step)
            writer.add_scalar("Events/genuine_grokking", float(gen_step is not None), step)
            writer.add_scalar("Events/phase_id", row["phase_id"], step)
            if remaining is not None:
                writer.add_scalar("Events/post_grok_remaining_steps", remaining, step)
            writer.flush()
        if step%run_cfg.checkpoint_every==0:
            train_state = dict(
                memo_step=memo_step, gen_step=gen_step,
                first_generalization_step=first_generalization_step,
                grok_detected_at_step=grok_detected_at_step,
                planned_stop_step=planned_stop_step,
                memo_streak=memo_streak, gen_streak=gen_streak,
                current_phase=current_phase,
                decay_activation_step=decay_activation_step,
                examples_seen=examples_seen,
                elapsed_seconds=row["elapsed_seconds"],
                previous_grad=previous_grad,
                previous_logged_params=previous_logged_params,
            )
            torch.save(dict(
                checkpoint_version=2, step=step,
                model=model.state_dict(), optimizer=opt.state_dict(),
                train_state=train_state,
                batch_stream=batch_stream.state_dict(),
                rng_state=_capture_rng_state(),
            ), run_dir/"checkpoint.pt")
            text_log.log(f"CHECKPOINT step={step} path={run_dir/'checkpoint.pt'}")
        if planned_stop_step is not None and step>=planned_stop_step:
            text_log.log(
                f"STOP post-grok tail completed: detected_at_step={grok_detected_at_step}, "
                f"post_grok_steps={run_cfg.post_grok_steps}, final_step={step}"
            )
            break
    if run_cfg.stream_csv and pending_rows:
        pd.DataFrame(pending_rows, columns=csv_columns).to_csv(
            csv_path, mode="a", header=not csv_path.exists(), index=False
        )
        pending_rows.clear()
    if last_row is None:
        raise RuntimeError("No log row was produced")
    early_generalization = (first_generalization_step is not None and gen_step is None)
    summary=dict(completed=True,p=p,seed=seed,**data_meta,
                 resolved_batch_size=effective_batch_size,
                 resolved_init_scale=run_init_scale,
                 resolved_weight_decay=run_weight_decay,
                 decay_activation_step=decay_activation_step,
                 memo_step=memo_step,
                 first_generalization_step=first_generalization_step,generalization_step=gen_step,
                 grok_detected_at_step=grok_detected_at_step,
                 planned_stop_step=planned_stop_step,
                 gap_steps=(gen_step-memo_step if gen_step is not None else None),final_step=last_row["step"],
                 final_train_acc=last_row["train_acc"],final_val_acc=last_row["val_acc"],
                 outcome=("genuine_grokking" if gen_step is not None else
                          "early_generalization" if early_generalization else
                          "memorized_only" if memo_step else "not_memorized"),
                 run_dir=str(run_dir))
    (run_dir/"config.json").write_text(json.dumps(asdict(run_cfg),indent=2),encoding="utf-8")
    done.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    final_train_state = dict(
        memo_step=memo_step, gen_step=gen_step,
        first_generalization_step=first_generalization_step,
        grok_detected_at_step=grok_detected_at_step,
        planned_stop_step=planned_stop_step,
        memo_streak=memo_streak, gen_streak=gen_streak,
        current_phase=current_phase,
        decay_activation_step=decay_activation_step,
        examples_seen=examples_seen,
        elapsed_seconds=last_row["elapsed_seconds"],
        previous_grad=previous_grad,
        previous_logged_params=previous_logged_params,
    )
    torch.save(dict(
        checkpoint_version=2, step=last_row["step"],
        model=model.state_dict(), optimizer=opt.state_dict(),
        train_state=final_train_state,
        batch_stream=batch_stream.state_dict(),
        rng_state=_capture_rng_state(),
    ), run_dir/"checkpoint_final.pt")
    text_log.log(
        f"END outcome={summary['outcome']} final_step={summary['final_step']} "
        f"final_train_acc={summary['final_train_acc']:.6f} "
        f"final_val_acc={summary['final_val_acc']:.6f}"
    )
    if writer: writer.close()
    text_log.close()
    del model,opt,projector; gc.collect(); torch.cuda.empty_cache()
    return summary


def run_sweep(cfg: Config):
    print("Device:",select_device(cfg.device)); summaries=[]
    root=Path(cfg.output_root)/cfg.protocol_name; root.mkdir(parents=True,exist_ok=True)
    for p in cfg.primes:
        for seed in cfg.seeds:
            summary=train_one(cfg,p,seed); summaries.append(summary)
            pd.DataFrame(summaries).to_csv(root/"sweep_summary.csv",index=False)
            print(json.dumps(summary,indent=2))
            if cfg.stop_after_first_success and summary["outcome"]=="genuine_grokking": return summaries
    return summaries


if __name__ == "__main__": run_sweep(Config())

