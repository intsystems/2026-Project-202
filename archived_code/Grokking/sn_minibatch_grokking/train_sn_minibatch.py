"""Mini-batch grokking experiments for multiplication in S_n.

The model follows Stander et al. (ICML 2024): separate left/right opaque-ID
embeddings, concatenation, one ReLU hidden layer and a linear unembedding.
Their published S5/S6 runs were full-batch.  This file deliberately studies a
mini-batch variant and extends the same family to S7; that extension is an
experiment, not a claimed reproduction.

The implementation is designed for Kaggle:
* deterministic, stateless sampling (the full S7 table is never materialised);
* checkpoints/logs under /kaggle/working by default;
* optional resume from an attached previous Kaggle output under /kaggle/input;
* monitor-set evaluation and exact chunked evaluation for S5/S6;
* EDM-friendly scalar logs, including three fixed random projections per layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


PAPER_URL = "https://proceedings.mlr.press/v235/stander24a.html"
MASK64 = (1 << 64) - 1


@dataclass
class Config:
    output_root: str = "/kaggle/working/sn_minibatch_grokking"
    # v1 used torch.optim.Adam(..., weight_decay=1).  In a mini-batch regime
    # that coupled penalty is divided by Adam's second-moment estimate and
    # collapses every parameter to (almost) zero.  v2 uses decoupled AdamW.
    protocol_name: str = "stander_mlp_minibatch_v2_adamw"
    n_values: tuple[int, ...] = (5, 6, 7)
    seeds: tuple[int, ...] = (42,)
    train_fraction: float = 0.40

    # Published sizes: S5=(embedding 256, hidden 128),
    # S6=(embedding 512, hidden 256). S7 is an explicit extrapolation.
    embedding_dim_by_n: dict[int, int] = field(
        default_factory=lambda: {5: 256, 6: 512, 7: 512}
    )
    hidden_dim_by_n: dict[int, int] = field(
        default_factory=lambda: {5: 128, 6: 256, 7: 512}
    )

    # True mini-batch defaults. Per-n overrides let S7 use a larger batch.
    batch_size_by_n: dict[int, int] = field(
        # S7's 5040-way logits dominate memory, hence its smaller batch.
        default_factory=lambda: {5: 4096, 6: 4096, 7: 2048}
    )
    # One "epoch" means approximately train_fraction * |S_n|^2 examples.
    # The run is capped by optimizer steps because S7 epochs are huge.
    max_steps_by_n: dict[int, int] = field(
        default_factory=lambda: {5: 250_000, 6: 300_000, 7: 400_000}
    )

    learning_rate: float = 1e-3
    # The paper used full-batch Adam with weight decay 1.  A literal coupled
    # L2 penalty is not transferable to mini-batch Adam: with this task it
    # overwhelms the stochastic data gradient.  Decoupled AdamW preserves the
    # intended strong shrinkage without preconditioning the penalty itself.
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)

    log_every: int = 20
    diagnostic_every: int = 1_000
    checkpoint_every: int = 2_000
    monitor_train_pairs: int = 8_192
    monitor_val_pairs: int = 8_192
    eval_batch_size: int = 8_192
    exact_eval_chunk_pairs: int = 262_144
    exact_eval_n_max: int = 6

    train_threshold: float = 0.99
    val_threshold: float = 0.95
    patience_logs: int = 10
    required_gap_steps: int = 10_000
    post_grok_steps: int = 5_000

    projection_count: int = 3
    # Fixed Rademacher projections of per-example losses on the monitor sets.
    # These remain cheap scalar observables but retain information cancelled by
    # the ordinary mean validation loss.
    loss_projection_count: int = 3
    device: str = "auto"
    use_amp: bool = True
    force_restart: bool = False
    resume_search_roots: tuple[str, ...] = ("/kaggle/input",)


class CosetMLP(nn.Module):
    """Separate embeddings -> concatenate -> Linear/ReLU -> unembedding."""

    def __init__(self, group_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.left_embedding = nn.Embedding(group_size, embedding_dim)
        self.right_embedding = nn.Embedding(group_size, embedding_dim)
        self.hidden = nn.Linear(2 * embedding_dim, hidden_dim)
        self.unembedding = nn.Linear(hidden_dim, group_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Conservative Xavier initialisation; no architecture-level shortcut.
        nn.init.normal_(self.left_embedding.weight, std=1.0 / math.sqrt(self.left_embedding.embedding_dim))
        nn.init.normal_(self.right_embedding.weight, std=1.0 / math.sqrt(self.right_embedding.embedding_dim))
        nn.init.xavier_normal_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.unembedding.weight)
        nn.init.zeros_(self.unembedding.bias)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat((self.left_embedding(left), self.right_embedding(right)), dim=-1)
        return self.unembedding(F.relu(self.hidden(embedded)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def pair_is_train(pair_id: int, seed: int, fraction: float) -> bool:
    """Deterministic Bernoulli split without allocating |G|^2 entries."""
    threshold = int(fraction * (1 << 64))
    return splitmix64(pair_id ^ splitmix64(seed)) < threshold


def factorials(n: int) -> list[int]:
    result = [1]
    for value in range(1, n + 1):
        result.append(result[-1] * value)
    return result


def build_permutations(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return permutations, Lehmer weights and rank lookup auxiliaries."""
    import itertools

    permutations = np.asarray(list(itertools.permutations(range(n))), dtype=np.int16)
    facts = np.asarray(factorials(n), dtype=np.int64)
    weights = np.asarray([facts[n - i - 1] for i in range(n)], dtype=np.int64)
    return permutations, facts, weights


def rank_permutations(batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorised Lehmer ranking for an array shaped [batch, n]."""
    # For n <= 7 this O(n^2) formula is faster and smaller than a Cayley table.
    ranks = np.zeros(len(batch), dtype=np.int64)
    n = batch.shape[1]
    for i in range(n - 1):
        ranks += (batch[:, i + 1:] < batch[:, i:i + 1]).sum(axis=1) * weights[i]
    return ranks


def product_ids(left: np.ndarray, right: np.ndarray, permutations: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    products = np.take_along_axis(permutations[left], permutations[right], axis=1)
    return rank_permutations(products, weights)


class StatelessPairSampler:
    """Uniform mini-batches from one deterministic side of the pair split."""

    def __init__(self, group_size: int, split_seed: int, fraction: float,
                 want_train: bool, sampling_seed: int):
        self.group_size = group_size
        self.total_pairs = group_size * group_size
        self.split_seed = split_seed
        self.fraction = fraction
        self.want_train = want_train
        self.rng = np.random.default_rng(sampling_seed)
        self.threshold = np.uint64(int(fraction * (1 << 64)))
        self.seed_hash = np.uint64(splitmix64(split_seed))

    @staticmethod
    def _mix_array(x: np.ndarray) -> np.ndarray:
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))

    def sample_pair_ids(self, count: int) -> np.ndarray:
        accepted: list[np.ndarray] = []
        remaining = count
        accept_rate = self.fraction if self.want_train else 1.0 - self.fraction
        while remaining:
            proposal_count = max(remaining + 64, math.ceil(1.12 * remaining / accept_rate))
            proposal = self.rng.integers(0, self.total_pairs, proposal_count, dtype=np.int64)
            hashed = self._mix_array(proposal.astype(np.uint64) ^ self.seed_hash)
            mask = hashed < self.threshold
            if not self.want_train:
                mask = ~mask
            chosen = proposal[mask][:remaining]
            accepted.append(chosen)
            remaining -= len(chosen)
        return np.concatenate(accepted)

    def sample(self, count: int, permutations: np.ndarray,
               weights: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_ids = self.sample_pair_ids(count)
        left, right = np.divmod(pair_ids, self.group_size)
        targets = product_ids(left, right, permutations, weights)
        return (torch.from_numpy(left), torch.from_numpy(right), torch.from_numpy(targets))


def make_monitor_set(sampler: StatelessPairSampler, count: int, permutations: np.ndarray,
                     weights: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Sampling with replacement also works when S5's requested monitor is
    # larger than its realised hash-split subset.
    array = sampler.sample_pair_ids(count)
    left, right = np.divmod(array, sampler.group_size)
    targets = product_ids(left, right, permutations, weights)
    return torch.from_numpy(left), torch.from_numpy(right), torch.from_numpy(targets)


@torch.inference_mode()
def evaluate_tensors(model: nn.Module, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     batch_size: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    left, right, targets = data
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(targets), batch_size):
        stop = start + batch_size
        l = left[start:stop].to(device, non_blocking=True)
        r = right[start:stop].to(device, non_blocking=True)
        y = targets[start:stop].to(device, non_blocking=True)
        logits = model(l, r)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    return loss_sum / len(targets), correct / len(targets)


def fixed_loss_projection_signs(left: torch.Tensor, right: torch.Tensor,
                                group_size: int, seed: int,
                                count: int) -> torch.Tensor:
    """Create fixed +/-1 readouts indexed by the underlying input pair."""
    pair_ids = (left.numpy().astype(np.uint64) * np.uint64(group_size)
                + right.numpy().astype(np.uint64))
    columns = []
    for index in range(count):
        salt = np.uint64(splitmix64(seed + 104_729 * index))
        hashed = StatelessPairSampler._mix_array(pair_ids ^ salt)
        signs = np.where((hashed & np.uint64(1)) == 0, -1.0, 1.0).astype(np.float32)
        columns.append(torch.from_numpy(signs))
    return torch.stack(columns, dim=1)


@torch.inference_mode()
def evaluate_with_loss_projections(
    model: nn.Module,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    signs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Mean CE/accuracy plus fixed random 1-D projections of sample losses.

    Projection j is sum_i r_ij * loss_i / sqrt(N).  The sqrt(N)
    normalisation keeps its numerical scale comparable across monitor sizes;
    the signs and examples are fixed throughout a run.
    """
    model.eval()
    left, right, targets = data
    loss_sum = 0.0
    correct = 0
    projected = torch.zeros(signs.shape[1], dtype=torch.float64)
    for start in range(0, len(targets), batch_size):
        stop = start + batch_size
        l = left[start:stop].to(device, non_blocking=True)
        r = right[start:stop].to(device, non_blocking=True)
        y = targets[start:stop].to(device, non_blocking=True)
        logits = model(l, r)
        losses = F.cross_entropy(logits, y, reduction="none")
        loss_sum += losses.sum().item()
        correct += (logits.argmax(-1) == y).sum().item()
        projected += (losses.detach().double().cpu()[:, None]
                      * signs[start:stop].double()).sum(dim=0)
    projected /= math.sqrt(len(targets))
    return loss_sum / len(targets), correct / len(targets), projected.numpy()


@torch.inference_mode()
def evaluate_exact_split(model: nn.Module, group_size: int, split_seed: int,
                         fraction: float, want_train: bool, permutations: np.ndarray,
                         weights: np.ndarray, pair_chunk: int, eval_batch: int,
                         device: torch.device) -> tuple[float, float, int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_pairs = group_size * group_size
    for start in range(0, total_pairs, pair_chunk):
        ids = np.arange(start, min(total_pairs, start + pair_chunk), dtype=np.int64)
        hashes = StatelessPairSampler._mix_array(ids.astype(np.uint64) ^ np.uint64(splitmix64(split_seed)))
        mask = hashes < np.uint64(int(fraction * (1 << 64)))
        if not want_train:
            mask = ~mask
        ids = ids[mask]
        left, right = np.divmod(ids, group_size)
        targets = product_ids(left, right, permutations, weights)
        data = torch.from_numpy(left), torch.from_numpy(right), torch.from_numpy(targets)
        loss, accuracy = evaluate_tensors(model, data, eval_batch, device)
        total_loss += loss * len(targets)
        total_correct += round(accuracy * len(targets))
        total_count += len(targets)
    return total_loss / total_count, total_correct / total_count, total_count


class LayerProjector:
    """Reproducible 1-D generic observers without storing random tensors."""

    def __init__(self, model: nn.Module, count: int, seed: int):
        self.count = count
        self.seed = seed
        self.named_parameters = [(name, parameter) for name, parameter in model.named_parameters()]

    def values(self, gradients: bool = False,
               names: set[str] | None = None) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, parameter in self.named_parameters:
            if names is not None and name not in names:
                continue
            tensor = parameter.grad if gradients else parameter
            if tensor is None:
                continue
            flat = tensor.detach().float().reshape(-1)
            for index in range(self.count):
                digest = hashlib.blake2b(
                    f"{self.seed}:{index}:{name}".encode(), digest_size=8
                ).digest()
                local_seed = int.from_bytes(digest, "little") & ((1 << 63) - 1)
                generator = torch.Generator(device=flat.device)
                generator.manual_seed(local_seed)
                # Chunking avoids an additional full-model vector allocation.
                dot = torch.zeros((), device=flat.device)
                for start in range(0, flat.numel(), 1_000_000):
                    chunk = flat[start:start + 1_000_000]
                    signs = torch.randint(0, 2, chunk.shape, generator=generator,
                                          device=flat.device, dtype=torch.int8)
                    signs = signs.float().mul_(2).sub_(1)
                    dot += torch.dot(chunk, signs)
                key = name.replace(".", "__")
                prefix = "gradproj" if gradients else "weightproj"
                result[f"{prefix}__{key}__r{index}"] = float(dot.cpu() / math.sqrt(flat.numel()))
        return result

    def selected_values(self, parameter_name: str, gradients: bool = False) -> dict[str, float]:
        return self.values(gradients=gradients, names={parameter_name})


def model_weight_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(parameter.detach().float().square().sum().cpu())
                         for parameter in model.parameters()))


def gradient_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().float().square().sum().cpu())
                         for parameter in model.parameters() if parameter.grad is not None))


def import_checkpoint(run_dir: Path, config: Config, n: int, seed: int) -> None:
    if config.force_restart or (run_dir / "checkpoint.pt").exists():
        return
    suffix = Path(config.protocol_name) / f"S_{n}" / f"seed_{seed}" / "checkpoint.pt"
    for root_text in config.resume_search_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        matches = list(root.rglob(str(suffix).replace("\\", "/")))
        if not matches:
            matches = [candidate for candidate in root.rglob("checkpoint.pt")
                       if tuple(candidate.parts[-len(suffix.parts):]) == suffix.parts]
        if matches:
            shutil.copy2(matches[0], run_dir / "checkpoint.pt")
            print("Imported checkpoint:", matches[0])
            return


def save_state(run_dir: Path, next_step: int, model: nn.Module,
               optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
               logs: list[dict], state: dict) -> None:
    temporary = run_dir / "checkpoint.pt.tmp"
    torch.save({
        "next_step": next_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "logs": logs,
        "state": state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, temporary)
    temporary.replace(run_dir / "checkpoint.pt")
    pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)


def select_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def run_one(config: Config, n: int, seed: int) -> Path:
    device = select_device(config.device)
    group_size = math.factorial(n)
    embedding_dim = config.embedding_dim_by_n[n]
    hidden_dim = config.hidden_dim_by_n[n]
    batch_size = config.batch_size_by_n[n]
    max_steps = config.max_steps_by_n[n]
    run_dir = Path(config.output_root) / config.protocol_name / f"S_{n}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_path = run_dir / "COMPLETED.json"
    if completed_path.exists() and not config.force_restart:
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("completed"):
            print("Skipping completed run:", run_dir)
            return run_dir

    import_checkpoint(run_dir, config, n, seed)
    set_seed(seed)
    permutations, _, weights = build_permutations(n)
    train_sampler = StatelessPairSampler(group_size, seed, config.train_fraction, True, seed + 10_001)
    val_sampler = StatelessPairSampler(group_size, seed, config.train_fraction, False, seed + 20_001)
    monitor_train = make_monitor_set(train_sampler, min(config.monitor_train_pairs,
                                    max(1, round(config.train_fraction * group_size * group_size))),
                                    permutations, weights)
    monitor_val = make_monitor_set(val_sampler, min(config.monitor_val_pairs,
                                  max(1, round((1 - config.train_fraction) * group_size * group_size))),
                                  permutations, weights)
    train_loss_signs = fixed_loss_projection_signs(
        monitor_train[0], monitor_train[1], group_size, seed + 70_001,
        config.loss_projection_count,
    )
    val_loss_signs = fixed_loss_projection_signs(
        monitor_val[0], monitor_val[1], group_size, seed + 80_001,
        config.loss_projection_count,
    )

    model = CosetMLP(group_size, embedding_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=config.betas, weight_decay=config.weight_decay)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    projector = LayerProjector(model, config.projection_count, seed + 90_001)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_train_pairs = config.train_fraction * group_size * group_size
    steps_per_epoch = expected_train_pairs / batch_size
    metadata = {
        "paper": PAPER_URL,
        "scope": "mini-batch adaptation; S7 is an extrapolation",
        "n": n,
        "group_size": group_size,
        "total_pairs": group_size * group_size,
        "split": {"kind": "stateless_hash_bernoulli", "seed": seed,
                  "train_fraction": config.train_fraction},
        "model": {"class": "CosetMLP", "embedding_dim": embedding_dim,
                  "hidden_dim": hidden_dim, "separate_left_right_embeddings": True,
                  "parameter_count": parameter_count},
        "optimizer": {"class": "AdamW", "lr": config.learning_rate,
                      "weight_decay": config.weight_decay, "betas": config.betas,
                      "batch_size": batch_size},
        "training": {"max_steps": max_steps, "expected_steps_per_epoch": steps_per_epoch,
                     "required_gap_steps": config.required_gap_steps},
        "config": asdict(config),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logs: list[dict] = []
    state = {"memo_logs": 0, "gen_logs": 0, "memo_step": None,
             "generalization_step": None, "grok_step": None, "stop_step": None,
             "examples_seen": 0}
    start_step = 0
    checkpoint = run_dir / "checkpoint.pt"
    if checkpoint.exists() and not config.force_restart:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved.get("scaler", {}))
        logs = saved.get("logs", [])
        state.update(saved.get("state", {}))
        start_step = int(saved["next_step"])
        if saved.get("torch_rng") is not None:
            torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if saved.get("numpy_rng") is not None:
            np.random.set_state(saved["numpy_rng"])
        if saved.get("python_rng") is not None:
            random.setstate(saved["python_rng"])
        if state.get("train_sampler_rng") is not None:
            train_sampler.rng.bit_generator.state = state["train_sampler_rng"]
        print(f"Resumed S_{n} seed={seed} at step {start_step}")

    started = time.time()
    progress = tqdm(range(start_step, max_steps), desc=f"Coset MLP S_{n}, seed {seed}")
    for step in progress:
        left, right, targets = train_sampler.sample(batch_size, permutations, weights)
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(left, right)
            loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        should_log = step % config.log_every == 0
        if should_log:
            train_batch_acc = float((logits.argmax(-1) == targets).float().mean().detach().cpu())
            grad_norm = gradient_norm(model)
            # Generic layerwise observers are useful but expensive. Log all
            # weight projections at the diagnostic cadence; at ordinary log
            # points keep one cheap observer of the embedding dynamics.
            if step % config.diagnostic_every == 0:
                gradient_projections = projector.values(gradients=True)
            else:
                gradient_projections = projector.selected_values(
                    "left_embedding.weight", gradients=True
                )
        scaler.step(optimizer)
        scaler.update()
        state["examples_seen"] += batch_size

        if should_log:
            monitor_train_loss, monitor_train_acc, train_loss_projections = evaluate_with_loss_projections(
                model, monitor_train, train_loss_signs, config.eval_batch_size, device
            )
            val_loss, val_acc, val_loss_projections = evaluate_with_loss_projections(
                model, monitor_val, val_loss_signs, config.eval_batch_size, device
            )
            if step % config.diagnostic_every == 0:
                weight_projections = projector.values(gradients=False)
            else:
                weight_projections = projector.selected_values(
                    "left_embedding.weight", gradients=False
                )
            row = {
                "step": step,
                "epoch_equivalent": state["examples_seen"] / expected_train_pairs,
                "examples_seen": state["examples_seen"],
                "train_batch_loss": float(loss.detach().cpu()),
                "train_batch_acc": train_batch_acc,
                "monitor_train_loss": monitor_train_loss,
                "monitor_train_acc": monitor_train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "weight_norm": model_weight_norm(model),
                "grad_norm": grad_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"monitor_train_lossproj_r{index}": float(value)
                   for index, value in enumerate(train_loss_projections)},
                **{f"val_lossproj_r{index}": float(value)
                   for index, value in enumerate(val_loss_projections)},
                **gradient_projections,
                **weight_projections,
            }
            logs.append(row)
            progress.set_postfix(train_acc=f"{monitor_train_acc:.3f}",
                                 val_acc=f"{val_acc:.3f}")

            state["memo_logs"] = state["memo_logs"] + 1 if monitor_train_acc >= config.train_threshold else 0
            state["gen_logs"] = state["gen_logs"] + 1 if val_acc >= config.val_threshold else 0
            if state["memo_step"] is None and state["memo_logs"] >= config.patience_logs:
                state["memo_step"] = step
                print(f"S_{n}: stable monitor memorization at step {step}")
            if state["generalization_step"] is None and state["gen_logs"] >= config.patience_logs:
                state["generalization_step"] = step
                gap = None if state["memo_step"] is None else step - state["memo_step"]
                if gap is not None and gap >= config.required_gap_steps:
                    state["grok_step"] = step
                    state["stop_step"] = min(max_steps, step + config.post_grok_steps)
                    print(f"S_{n}: genuine mini-batch grokking at step {step}, gap={gap}")
                else:
                    # Do not reject/stop: a monitor can cross early and later fluctuate.
                    print(f"S_{n}: early monitor generalization at step {step}, gap={gap}; continuing")

            if step % config.diagnostic_every == 0:
                pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)

        if (step + 1) % config.checkpoint_every == 0:
            state["train_sampler_rng"] = train_sampler.rng.bit_generator.state
            save_state(run_dir, step + 1, model, optimizer, scaler, logs, state)
        if state["stop_step"] is not None and step + 1 >= state["stop_step"]:
            break

    final_step = logs[-1]["step"] if logs else start_step
    final_train = evaluate_tensors(model, monitor_train, config.eval_batch_size, device)
    final_val = evaluate_tensors(model, monitor_val, config.eval_batch_size, device)
    exact = {}
    if n <= config.exact_eval_n_max:
        exact_train_loss, exact_train_acc, exact_train_count = evaluate_exact_split(
            model, group_size, seed, config.train_fraction, True, permutations, weights,
            config.exact_eval_chunk_pairs, config.eval_batch_size, device
        )
        exact_val_loss, exact_val_acc, exact_val_count = evaluate_exact_split(
            model, group_size, seed, config.train_fraction, False, permutations, weights,
            config.exact_eval_chunk_pairs, config.eval_batch_size, device
        )
        exact = {"exact_train_loss": exact_train_loss, "exact_train_acc": exact_train_acc,
                 "exact_train_pairs": exact_train_count, "exact_val_loss": exact_val_loss,
                 "exact_val_acc": exact_val_acc, "exact_val_pairs": exact_val_count}

    result = {
        "completed": True,
        "n": n,
        "seed": seed,
        "success": state["grok_step"] is not None,
        "outcome": ("genuine_grokking" if state["grok_step"] is not None else
                    "early_generalization" if state["generalization_step"] is not None else
                    "memorized_without_generalization" if state["memo_step"] is not None else
                    "not_memorized"),
        "memo_step": state["memo_step"],
        "generalization_step": state["generalization_step"],
        "grok_step": state["grok_step"],
        "gap_steps": (None if state["memo_step"] is None or state["generalization_step"] is None
                      else state["generalization_step"] - state["memo_step"]),
        "final_step": final_step,
        "final_monitor_train_loss": final_train[0],
        "final_monitor_train_acc": final_train[1],
        "final_monitor_val_loss": final_val[0],
        "final_monitor_val_acc": final_val[1],
        "wall_time_seconds": time.time() - started,
        **exact,
    }
    completed_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    state["train_sampler_rng"] = train_sampler.rng.bit_generator.state
    save_state(run_dir, int(final_step) + 1, model, optimizer, scaler, logs, state)
    print(json.dumps(result, indent=2))
    return run_dir


def run(config: Config) -> list[Path]:
    print("Device:", select_device(config.device))
    paths = []
    for n in config.n_values:
        if n not in config.embedding_dim_by_n or n not in config.hidden_dim_by_n:
            raise ValueError(f"Missing architecture for S_{n}")
        for seed in config.seeds:
            paths.append(run_one(config, n, seed))
    return paths


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="5,6,7")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-root", default="/kaggle/working/sn_minibatch_grokking")
    parser.add_argument("--protocol-name", default="stander_mlp_minibatch_v2_adamw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None,
                        help="Override every max_steps value for a local smoke test")
    args = parser.parse_args()
    config = Config(
        output_root=args.output_root,
        protocol_name=args.protocol_name,
        n_values=parse_int_tuple(args.n_values),
        seeds=parse_int_tuple(args.seeds),
        device=args.device,
        force_restart=args.force_restart,
    )
    if args.smoke_steps is not None:
        config.max_steps_by_n = {n: args.smoke_steps for n in config.n_values}
        config.log_every = 1
        config.diagnostic_every = 1
        config.checkpoint_every = max(1, args.smoke_steps)
        config.monitor_train_pairs = 64
        config.monitor_val_pairs = 64
        config.batch_size_by_n = {n: 32 for n in config.n_values}
        config.projection_count = 1
        config.loss_projection_count = 1
        config.exact_eval_n_max = 0
    run(config)


if __name__ == "__main__":
    main()
