"""Faithful full-batch S_n group-composition grokking runs for Kaggle.

This is intentionally a separate protocol from the experimental mini-batch
runner.  For S5/S6 it follows the released Chughtai/Stander setup: bias-free
one-hidden-layer MLP, a torch.randperm 40/60 split, full-batch AdamW, lr 1e-3,
weight decay 1, betas (0.9, 0.98), and float64 cross entropy.

S7 full batch is computationally infeasible (about 10.2M training pairs and
5040 logits per pair), so this runner explicitly supports S5 and S6 only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


BUILD_ID = "sn-fullbatch-paper-v1.1-2026-08-02"
PAPER_URL = "https://proceedings.mlr.press/v235/stander24a.html"


@dataclass
class Config:
    output_root: str = "/kaggle/working/sn_fullbatch_grokking"
    protocol_name: str = "stander_exact_fullbatch_v1"
    n_values: tuple[int, ...] = (5,)
    seeds: tuple[int, ...] = (42,)
    train_fraction: float = 0.40
    embedding_dim_by_n: dict[int, int] = field(
        default_factory=lambda: {5: 256, 6: 512}
    )
    hidden_dim_by_n: dict[int, int] = field(
        default_factory=lambda: {5: 128, 6: 256}
    )
    # Published Stander protocol: S5 = 250k, S6 = 50k full-batch epochs.
    # Do not give S6 the S5 budget: one S6 epoch is about 150x heavier.
    max_steps_by_n: dict[int, int] = field(
        default_factory=lambda: {5: 250_000, 6: 50_000}
    )
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    log_every: int = 20
    diagnostic_every: int = 1_000
    checkpoint_every: int = 1_000
    eval_batch_size: int = 16_384
    # One chunk is fastest on a 16 GB Kaggle GPU. If another accelerator runs
    # out of memory, reduce S6 to 32768; the accumulated gradient stays exact.
    train_chunk_size_by_n: dict[int, int] = field(
        default_factory=lambda: {5: 5_760, 6: 207_360}
    )
    train_threshold: float = 0.99
    val_threshold: float = 0.95
    patience_logs: int = 10
    required_gap_steps: int = 10_000
    post_grok_steps: int = 5_000
    projection_count: int = 3
    loss_projection_count: int = 3
    device: str = "auto"
    # False is deliberate: the released protocol trains in FP32 and evaluates
    # CE in FP64. AMP changes the long late-time dynamics that produce grokking.
    use_amp: bool = False
    force_restart: bool = False
    resume_search_roots: tuple[str, ...] = ("/kaggle/input",)


class PaperMLP(nn.Module):
    """Exact released architecture: four parameter matrices, no biases."""

    def __init__(self, group_size: int, embedding_dim: int, hidden_dim: int,
                 seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.left_embedding = nn.Parameter(
            torch.randn(group_size, embedding_dim) / math.sqrt(embedding_dim)
        )
        self.right_embedding = nn.Parameter(
            torch.randn(group_size, embedding_dim) / math.sqrt(embedding_dim)
        )
        self.hidden = nn.Parameter(
            torch.randn(2 * embedding_dim, hidden_dim) / math.sqrt(2 * embedding_dim)
        )
        self.unembedding = nn.Parameter(
            torch.randn(hidden_dim, group_size) / math.sqrt(hidden_dim)
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat((self.left_embedding[left], self.right_embedding[right]), dim=-1)
        return F.relu(embedded @ self.hidden) @ self.unembedding


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def factorials(n: int) -> list[int]:
    values = [1]
    for k in range(1, n + 1):
        values.append(values[-1] * k)
    return values


def build_permutations(n: int) -> tuple[np.ndarray, np.ndarray]:
    import itertools
    permutations = np.asarray(list(itertools.permutations(range(n))), dtype=np.int16)
    facts = factorials(n)
    weights = np.asarray([facts[n - i - 1] for i in range(n)], dtype=np.int64)
    return permutations, weights


def rank_permutations(batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ranks = np.zeros(len(batch), dtype=np.int64)
    for i in range(batch.shape[1] - 1):
        ranks += (batch[:, i + 1:] < batch[:, i:i + 1]).sum(axis=1) * weights[i]
    return ranks


def make_all_data(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    permutations, weights = build_permutations(n)
    group_size = len(permutations)
    pair_ids = np.arange(group_size * group_size, dtype=np.int64)
    left, right = np.divmod(pair_ids, group_size)
    products = np.take_along_axis(permutations[left], permutations[right], axis=1)
    targets = rank_permutations(products, weights)
    return torch.from_numpy(left), torch.from_numpy(right), torch.from_numpy(targets)


def paper_split(n: int, fraction: float, seed: int):
    """The released code shuffles all pairs with torch.randperm(seed)."""
    left, right, targets = make_all_data(n)
    torch.manual_seed(seed)
    order = torch.randperm(len(targets))
    split = int(fraction * len(targets))
    train_ids, val_ids = order[:split], order[split:]
    train = left[train_ids], right[train_ids], targets[train_ids]
    val = left[val_ids], right[val_ids], targets[val_ids]
    return train, val


def paper_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """FP64 CE, matching the public training implementation."""
    return F.cross_entropy(logits.double(), labels)


def paper_loss_chunked_forward(model: nn.Module, data, chunk_size: int) -> torch.Tensor:
    """Exact full-batch gradient without materialising all S6 logits at once.

    Every chunk loss is weighted by its share of the complete train set, hence
    accumulated gradients equal the gradient of one global mean CE.  This cuts
    peak memory substantially but does not turn training into mini-batch SGD:
    optimizer.step() is still called exactly once per full-data epoch.
    """
    left, right, targets = data
    total = len(targets)
    loss_value = torch.zeros((), device=left.device, dtype=torch.float64)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        chunk_loss = paper_loss(model(left[start:stop], right[start:stop]),
                                targets[start:stop])
        weight = (stop - start) / total
        (chunk_loss * weight).backward()
        loss_value += chunk_loss.detach() * weight
    return loss_value


@torch.inference_mode()
def evaluate(model: nn.Module, data, batch_size: int, device: torch.device):
    model.eval()
    left, right, targets = data
    loss_sum, correct = 0.0, 0
    for start in range(0, len(targets), batch_size):
        stop = min(len(targets), start + batch_size)
        l = left[start:stop].to(device, non_blocking=True)
        r = right[start:stop].to(device, non_blocking=True)
        y = targets[start:stop].to(device, non_blocking=True)
        logits = model(l, r)
        loss_sum += F.cross_entropy(logits.double(), y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    return loss_sum / len(targets), correct / len(targets)


def fixed_loss_signs(left: torch.Tensor, right: torch.Tensor, group_size: int,
                     seed: int, count: int) -> torch.Tensor:
    pair_ids = left.numpy().astype(np.uint64) * np.uint64(group_size) + right.numpy().astype(np.uint64)
    cols = []
    for j in range(count):
        rng = np.random.default_rng(seed + 104_729 * j)
        # A deterministic table indexed by pair ID means duplicate-free exact
        # splits get stable observers across all checkpoints.
        table = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32),
                           size=group_size * group_size)
        cols.append(torch.from_numpy(table[pair_ids.astype(np.int64)]))
    return torch.stack(cols, dim=1)


@torch.inference_mode()
def evaluate_with_projections(model: nn.Module, data, signs: torch.Tensor,
                              batch_size: int, device: torch.device):
    model.eval()
    left, right, targets = data
    loss_sum, correct = 0.0, 0
    projected = torch.zeros(signs.shape[1], dtype=torch.float64)
    for start in range(0, len(targets), batch_size):
        stop = min(len(targets), start + batch_size)
        l = left[start:stop].to(device, non_blocking=True)
        r = right[start:stop].to(device, non_blocking=True)
        y = targets[start:stop].to(device, non_blocking=True)
        logits = model(l, r)
        losses = F.cross_entropy(logits.double(), y, reduction="none")
        loss_sum += losses.sum().item()
        correct += (logits.argmax(-1) == y).sum().item()
        projected += (losses.cpu()[:, None] * signs[start:stop].double()).sum(0)
    projected /= math.sqrt(len(targets))
    return loss_sum / len(targets), correct / len(targets), projected.numpy()


class LayerProjector:
    def __init__(self, model: nn.Module, count: int, seed: int):
        self.parameters = list(model.named_parameters())
        self.count, self.seed = count, seed

    def values(self, gradients: bool) -> dict[str, float]:
        out = {}
        prefix = "gradproj" if gradients else "weightproj"
        for name, parameter in self.parameters:
            value = parameter.grad if gradients else parameter
            if value is None:
                continue
            flat = value.detach().float().flatten()
            for j in range(self.count):
                digest = hashlib.blake2b(f"{self.seed}:{j}:{name}".encode(), digest_size=8).digest()
                generator = torch.Generator(device=flat.device)
                generator.manual_seed(int.from_bytes(digest, "little") & ((1 << 63) - 1))
                dot = torch.zeros((), device=flat.device)
                for start in range(0, flat.numel(), 1_000_000):
                    chunk = flat[start:start + 1_000_000]
                    signs = torch.randint(0, 2, chunk.shape, generator=generator,
                                          device=flat.device, dtype=torch.int8)
                    dot += torch.dot(chunk, signs.float().mul_(2).sub_(1))
                out[f"{prefix}__{name}__r{j}"] = float(dot.cpu() / math.sqrt(flat.numel()))
        return out


def tensor_norm(values) -> float:
    return math.sqrt(sum(float(x.detach().float().square().sum().cpu()) for x in values))


def select_device(text: str) -> torch.device:
    return torch.device("cuda" if text == "auto" and torch.cuda.is_available() else
                        "cpu" if text == "auto" else text)


def import_checkpoint(run_dir: Path, config: Config, n: int, seed: int) -> None:
    if config.force_restart or (run_dir / "checkpoint.pt").exists():
        return
    tail = (config.protocol_name, f"S_{n}", f"seed_{seed}", "checkpoint.pt")
    for root_text in config.resume_search_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for candidate in root.rglob("checkpoint.pt"):
            if tuple(candidate.parts[-4:]) == tail:
                shutil.copy2(candidate, run_dir / "checkpoint.pt")
                print("Imported checkpoint:", candidate)
                return


def save_checkpoint(path: Path, next_step: int, model, optimizer, logs, state):
    temp = path / "checkpoint.pt.tmp"
    torch.save({"build_id": BUILD_ID, "next_step": next_step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "logs": logs, "state": state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temp)
    temp.replace(path / "checkpoint.pt")
    pd.DataFrame(logs).to_csv(path / "training_log.csv", index=False)


def run_one(config: Config, n: int, seed: int) -> Path:
    if n not in (5, 6):
        raise ValueError("Exact full-batch protocol supports only S5/S6; S7 full batch is infeasible")
    device = select_device(config.device)
    run_dir = Path(config.output_root) / config.protocol_name / f"S_{n}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "COMPLETED.json"
    if completed.exists() and not config.force_restart:
        print("Skipping completed run:", run_dir)
        return run_dir
    import_checkpoint(run_dir, config, n, seed)
    set_seed(seed)
    train, val = paper_split(n, config.train_fraction, seed)
    group_size = math.factorial(n)
    # Public code resets torch's seed inside model construction after splitting.
    model = PaperMLP(group_size, config.embedding_dim_by_n[n],
                     config.hidden_dim_by_n[n], seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=config.betas, weight_decay=config.weight_decay)
    train_gpu = tuple(x.to(device) for x in train)
    projector = LayerProjector(model, config.projection_count, seed + 90_001)
    train_signs = fixed_loss_signs(train[0], train[1], group_size, seed + 70_001,
                                   config.loss_projection_count)
    val_signs = fixed_loss_signs(val[0], val[1], group_size, seed + 80_001,
                                 config.loss_projection_count)
    metadata = {"build_id": BUILD_ID, "paper": PAPER_URL, "n": n, "seed": seed,
                "group_size": group_size, "split": "torch.randperm exact 40/60",
                "model": "bias-free PaperMLP", "optimizer": "AdamW",
                "config": asdict(config)}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logs, start = [], 0
    state = {"memo_logs": 0, "gen_logs": 0, "memo_step": None,
             "generalization_step": None, "grok_step": None, "stop_step": None}
    checkpoint = run_dir / "checkpoint.pt"
    if checkpoint.exists() and not config.force_restart:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("build_id") != BUILD_ID:
            raise RuntimeError("Refusing incompatible/stale checkpoint: " + str(checkpoint))
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        logs, state = saved.get("logs", []), saved.get("state", state)
        start = int(saved["next_step"])
        print(f"Resumed exact full-batch S_{n} seed={seed} at step {start}")

    started = time.time()
    max_steps = config.max_steps_by_n[n]
    progress = tqdm(range(start, max_steps), desc=f"Full-batch MLP S_{n}, seed {seed}")
    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = paper_loss_chunked_forward(
            model, train_gpu, config.train_chunk_size_by_n[n]
        )
        grad_norm = tensor_norm(p.grad for p in model.parameters() if p.grad is not None)
        diagnostic = step % config.diagnostic_every == 0
        grad_projections = projector.values(True) if diagnostic else {}
        optimizer.step()

        if step % config.log_every == 0:
            train_loss, train_acc, train_lossproj = evaluate_with_projections(
                model, train, train_signs, config.eval_batch_size, device)
            val_loss, val_acc, val_lossproj = evaluate_with_projections(
                model, val, val_signs, config.eval_batch_size, device)
            weight_projections = projector.values(False) if diagnostic else {}
            row = {
                # Canonical names restored for the existing analysis notebooks.
                "step": step, "train_loss": train_loss,
                "full_train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "full_train_acc": train_acc,
                "val_acc": val_acc,
                "weight_norm": tensor_norm(model.parameters()),
                "grad_norm": grad_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_lossproj_r{j}": float(x) for j, x in enumerate(train_lossproj)},
                **{f"val_lossproj_r{j}": float(x) for j, x in enumerate(val_lossproj)},
                **grad_projections, **weight_projections,
            }
            logs.append(row)
            # Restore the useful old progress display: loss, not accuracy.
            progress.set_postfix(train=f"{train_loss:.3g}", val_acc=f"{val_acc:.3f}")
            state["memo_logs"] = state["memo_logs"] + 1 if train_acc >= config.train_threshold else 0
            state["gen_logs"] = state["gen_logs"] + 1 if val_acc >= config.val_threshold else 0
            if state["memo_step"] is None and state["memo_logs"] >= config.patience_logs:
                state["memo_step"] = step
                print(f"S_{n}: stable memorization at step {step}")
            if state["generalization_step"] is None and state["gen_logs"] >= config.patience_logs:
                state["generalization_step"] = step
                gap = None if state["memo_step"] is None else step - state["memo_step"]
                if gap is not None and gap >= config.required_gap_steps:
                    state["grok_step"] = step
                    state["stop_step"] = min(max_steps, step + config.post_grok_steps)
                    print(f"S_{n}: genuine grokking at step {step}, gap={gap}")
                else:
                    print(f"S_{n}: early generalization at step {step}, gap={gap}; continuing")
            if diagnostic:
                pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
        if (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(run_dir, step + 1, model, optimizer, logs, state)
        if state["stop_step"] is not None and step + 1 >= state["stop_step"]:
            break

    final_step = int(logs[-1]["step"] if logs else start)
    final_train = evaluate(model, train, config.eval_batch_size, device)
    final_val = evaluate(model, val, config.eval_batch_size, device)
    result = {"completed": True, "build_id": BUILD_ID, "n": n, "seed": seed,
              "success": state["grok_step"] is not None,
              "outcome": "genuine_grokking" if state["grok_step"] is not None else
                         "memorized_without_generalization" if state["memo_step"] is not None else
                         "not_memorized",
              "memo_step": state["memo_step"],
              "generalization_step": state["generalization_step"],
              "grok_step": state["grok_step"],
              "gap_steps": None if state["memo_step"] is None or state["generalization_step"] is None
                           else state["generalization_step"] - state["memo_step"],
              "final_step": final_step, "final_train_loss": final_train[0],
              "final_train_acc": final_train[1], "final_val_loss": final_val[0],
              "final_val_acc": final_val[1], "wall_time_seconds": time.time() - started}
    completed.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_checkpoint(run_dir, final_step + 1, model, optimizer, logs, state)
    print(json.dumps(result, indent=2))
    return run_dir


def run(config: Config) -> list[Path]:
    print("Build:", BUILD_ID)
    print("Device:", select_device(config.device))
    return [run_one(config, n, seed) for n in config.n_values for seed in config.seeds]
