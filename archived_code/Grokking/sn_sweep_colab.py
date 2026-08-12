"""Colab-friendly sweep over composition tasks in symmetric groups S_n.

The legacy S_5/S_6 generator materialises all |S_n|^2 pairs and classifies a
product by its permutation id.  That representation stops scaling long before
S_10.  This module keeps the mathematical task (composition of permutations),
but represents permutations as token sequences and predicts the n output
positions.  Its vocabulary is O(n), not O(n!), so the identical protocol can
be run for n=3,...,10 on a standard Colab GPU.

Each run writes a resumable checkpoint, a CSV log, and metadata to Google
Drive or any other writable directory.  The log schema is compatible with the
project's downstream EDM analysis: step, train/val loss and accuracy, weight
norm, gradient norm and gradient cosine are present.
"""

from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

MAX_N = 10
EQUAL_TOKEN = MAX_N
MASK_TOKEN = MAX_N + 1
PAD_TOKEN = MAX_N + 2
VOCAB_SIZE = MAX_N + 3
SEQUENCE_LENGTH = 3 * MAX_N + 1


@dataclass
class SweepConfig:
    """Configuration shared by all n and seeds in one sweep."""

    output_root: str = "/content/drive/MyDrive/grokking_sn_sweep"
    protocol_name: str = "token_v4_genuine_gap10k"
    n_values: tuple[int, ...] = tuple(range(3, 11))
    seeds: tuple[int, ...] = (42,)
    # A 50/50 split is the standard grokking regime used by the original
    # S_5 notebook.  The former 80/20 split made the algorithm identifiable
    # almost immediately, especially with coordinate embeddings.
    train_fraction: float = 0.5
    train_fraction_by_n: dict[int, float] | None = None
    # S_5 has exactly 14,400 pairs.  Capping every larger group at this value
    # prevents dataset size from becoming a confound in the S_n comparison.
    max_unique_pairs: int = 14_400
    max_epochs: int = 3_000
    min_steps: int = 20_000
    max_steps: int = 200_000
    batch_size: int = 256
    eval_batch_size: int = 1024
    log_every: int = 50
    checkpoint_every: int = 1_000
    learning_rate: float = 1e-3
    # Constant throughout training.  No event-triggered schedule is used:
    # any observed delay is therefore a property of the learning dynamics.
    weight_decay: float = 2e-1
    min_grok_gap_steps: int = 10_000
    betas: tuple[float, float] = (0.9, 0.98)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.0
    target_train_acc: float = 0.99
    target_val_acc: float = 0.95
    memorization_patience_logs: int = 10
    generalization_patience_logs: int = 10
    post_grok_steps: int = 5_000
    num_workers: int = 2
    force_restart: bool = False


def factorial(n: int) -> int:
    return math.factorial(n)


def unrank_permutation(rank: int, n: int) -> list[int]:
    """Lehmer-code unranking of a permutation in lexicographic order."""
    remaining = list(range(n))
    result: list[int] = []
    for width in range(n, 0, -1):
        block = factorial(width - 1)
        index, rank = divmod(rank, block)
        result.append(remaining.pop(index))
    return result


def compose(left: list[int], right: list[int]) -> list[int]:
    """Composition convention matching the existing S_5 generator: left[right[k]]."""
    return [left[right[k]] for k in range(len(left))]


def make_fixed_split(n: int, fraction: float, max_unique_pairs: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a deterministic, disjoint fixed train/validation split.

    At small n every pair is used.  At large n, a reproducible subset is drawn
    without replacement from the conceptual Cartesian product; no S_n x S_n
    table is materialised.
    """
    group_size = factorial(n)
    total_pairs = group_size * group_size
    sampled_pairs = min(total_pairs, max_unique_pairs)
    if sampled_pairs < 2:
        raise ValueError(f"S_{n} provides fewer than two pairs")
    train_count = max(1, min(sampled_pairs - 1, int(round(sampled_pairs * fraction))))
    rng = random.Random(seed)
    pair_ids = rng.sample(range(total_pairs), sampled_pairs)

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for pair_id in pair_ids:
        left_id, right_id = divmod(pair_id, group_size)
        left = unrank_permutation(left_id, n)
        right = unrank_permutation(right_id, n)
        product = compose(left, right)
        padding = [PAD_TOKEN] * (MAX_N - n)
        # Fixed length and vocabulary make the parameterised architecture
        # exactly identical for every n.  Only the task mask changes.
        inputs.append(
            left + padding + right + padding + [EQUAL_TOKEN]
            + [MASK_TOKEN] * n + padding
        )
        targets.append(product)

    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    train = torch.cat([x[:train_count], y[:train_count]], dim=1)
    validation = torch.cat([x[train_count:], y[train_count:]], dim=1)
    metadata = {
        "group": f"S_{n}", "n": n, "group_size": group_size,
        "total_group_pairs": total_pairs, "sampled_unique_pairs": sampled_pairs,
        "train_pairs": train_count, "validation_pairs": sampled_pairs - train_count,
        "split_seed": seed, "representation": "tokenised_permutation_sequence_v1",
        "input_length": SEQUENCE_LENGTH, "target_length": n,
        "max_supported_n": MAX_N, "vocabulary_size": VOCAB_SIZE,
        "special_tokens": {"equals": EQUAL_TOKEN, "mask": MASK_TOKEN, "pad": PAD_TOKEN},
    }
    return train, validation, metadata


class SNCompositionTransformer(nn.Module):
    """Small bidirectional transformer for tokenised permutation composition."""

    def __init__(self, n: int, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.n = n
        self.sequence_length = SEQUENCE_LENGTH
        self.output_start = 2 * MAX_N + 1
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.position_embedding = nn.Parameter(torch.empty(1, self.sequence_length, d_model))
        # Explicit roles remove an avoidable symmetry: the same symbol has a
        # different meaning in the left factor, right factor and output slots.
        self.role_embedding = nn.Embedding(4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, MAX_N)
        nn.init.normal_(self.position_embedding, std=0.02)

        role_ids = [0] * MAX_N + [1] * MAX_N + [2] + [3] * MAX_N
        self.register_buffer("role_ids", torch.tensor(role_ids, dtype=torch.long), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        states = (
            self.token_embedding(tokens)
            + self.position_embedding
            + self.role_embedding(self.role_ids)[None, :, :]
        )
        states = self.encoder(states, src_key_padding_mask=tokens.eq(PAD_TOKEN))
        states = self.final_norm(states)
        # The head has MAX_N outputs in every experiment; slicing merely masks
        # impossible symbols for the current S_n task.
        output_states = states[:, self.output_start:self.output_start + self.n, :]
        return self.output(output_states)[:, :, :self.n]  # [batch, n, n]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weight_norm(model: nn.Module) -> float:
    return float(torch.sqrt(sum(parameter.detach().pow(2).sum() for parameter in model.parameters())).cpu())


@torch.no_grad()
def evaluate(model: nn.Module, packed: torch.Tensor, batch_size: int, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_sequences_correct = 0
    total_sequences = 0
    total_tokens = 0
    sequence_length = packed.shape[1] - model.n
    for start in range(0, len(packed), batch_size):
        batch = packed[start:start + batch_size].to(device, non_blocking=True)
        logits = model(batch[:, :sequence_length])
        targets = batch[:, sequence_length:]
        total_loss += F.cross_entropy(logits.reshape(-1, model.n), targets.reshape(-1), reduction="sum").item()
        correct = logits.argmax(dim=-1) == targets
        total_correct += correct.sum().item()
        total_sequences_correct += correct.all(dim=1).sum().item()
        total_sequences += len(targets)
        total_tokens += targets.numel()
    return (total_loss / total_tokens, total_correct / total_tokens,
            total_sequences_correct / total_sequences)


def save_run_state(run_dir: Path, payload: dict, logs: list[dict]) -> None:
    """Persist both a checkpoint and an analysis-ready CSV after each interval."""
    torch.save(payload, run_dir / "checkpoint.pt")
    pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)


def train_one(n: int, seed: int, config: SweepConfig, device: torch.device) -> Path:
    if n < 3 or n > 10:
        raise ValueError("This sweep is deliberately bounded to n=3,...,10")
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if config.d_model % config.n_heads:
        raise ValueError("d_model must be divisible by n_heads")

    run_dir = Path(config.output_root) / config.protocol_name / f"S_{n}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_marker = run_dir / "COMPLETED.json"
    if completed_marker.exists() and not config.force_restart:
        completed = json.loads(completed_marker.read_text(encoding="utf-8"))
        if completed.get("success", False):
            print(f"Skipping successful run: {run_dir}")
            return run_dir
        print(f"Previous run reached its old limit without grokking: {run_dir}; "
              "attempting to resume with the current limits")

    set_seed(seed)
    if config.train_fraction_by_n is not None and n not in config.train_fraction_by_n:
        raise ValueError(
            f"S_{n} has no calibrated train fraction. Calibrate it before the final sweep."
        )
    train_fraction = (config.train_fraction if config.train_fraction_by_n is None
                      else config.train_fraction_by_n[n])
    train_packed, validation_packed, split_metadata = make_fixed_split(
        n, train_fraction, config.max_unique_pairs, seed
    )
    steps_per_epoch = math.ceil(len(train_packed) / min(config.batch_size, len(train_packed)))
    max_steps = min(config.max_steps, max(config.min_steps, config.max_epochs * steps_per_epoch))
    run_metadata = {
        **split_metadata, "seed": seed, "protocol_name": config.protocol_name,
        "train_fraction": train_fraction,
        "max_epochs": config.max_epochs, "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "optimizer": "AdamW", "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": list(config.betas),
        "model": {"d_model": config.d_model, "n_heads": config.n_heads,
                  "n_layers": config.n_layers, "dropout": config.dropout,
                  "role_embeddings": True, "coordinate_embeddings": False,
                  "fixed_max_n": MAX_N, "vocabulary_size": VOCAB_SIZE,
                  "output_classes": MAX_N, "sequence_length": SEQUENCE_LENGTH},
        "logging": {"log_every": config.log_every,
                    "checkpoint_every": config.checkpoint_every,
                    "grad_cosine": "cosine between gradients at consecutive logged updates"},
        "stopping": {"target_train_acc": config.target_train_acc,
                     "target_val_acc": config.target_val_acc,
                     "memorization_patience_logs": config.memorization_patience_logs,
                     "generalization_patience_logs": config.generalization_patience_logs,
                     "min_grok_gap_steps": config.min_grok_gap_steps,
                     "post_grok_steps": config.post_grok_steps},
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    model = SNCompositionTransformer(n, config.d_model, config.n_heads, config.n_layers, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=config.betas)
    logs: list[dict] = []
    start_step = 0
    checkpoint_path = run_dir / "checkpoint.pt"
    if checkpoint_path.exists() and not config.force_restart:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logs = checkpoint["logs"]
        start_step = int(checkpoint["next_step"])
        memorization_logs = int(checkpoint.get("memorization_logs", 0))
        generalization_logs = int(checkpoint.get("generalization_logs", 0))
        first_memorization_step = checkpoint.get("first_memorization_step")
        first_generalization_step = checkpoint.get("first_generalization_step")
        first_grok_step = checkpoint.get("first_grok_step")
        stop_step = checkpoint.get("stop_step")
        print(f"Resuming S_{n}, seed={seed} from step {start_step}")

    loader = DataLoader(
        TensorDataset(train_packed), batch_size=min(config.batch_size, len(train_packed)),
        shuffle=True, drop_last=False, pin_memory=device.type == "cuda",
        num_workers=config.num_workers, persistent_workers=config.num_workers > 0,
    )
    iterator = iter(loader)
    previous_logged_gradient: torch.Tensor | None = None
    if not checkpoint_path.exists() or config.force_restart:
        memorization_logs = 0
        generalization_logs = 0
        first_memorization_step: int | None = None
        first_generalization_step: int | None = None
        first_grok_step: int | None = None
        stop_step: int | None = None
    sequence_length = train_packed.shape[1] - n
    progress = tqdm(range(start_step, max_steps), desc=f"S_{n}, seed {seed}")

    for step in progress:
        try:
            (packed_cpu,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (packed_cpu,) = next(iterator)
        packed = packed_cpu.to(device, non_blocking=True)
        tokens, targets = packed[:, :sequence_length], packed[:, sequence_length:]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, n), targets.reshape(-1))
        loss.backward()

        # The legacy notebook called optimizer.step() twice.  Here there is
        # exactly one update, after all gradient diagnostics have been read.
        if step % config.log_every == 0:
            grad_sq = sum(parameter.grad.detach().pow(2).sum()
                          for parameter in model.parameters() if parameter.grad is not None)
            grad_norm = float(torch.sqrt(grad_sq).cpu())
            gradient = torch.cat([parameter.grad.detach().reshape(-1)
                                  for parameter in model.parameters() if parameter.grad is not None])
            grad_cosine = (0.0 if previous_logged_gradient is None else
                           float(F.cosine_similarity(gradient, previous_logged_gradient, dim=0).cpu()))
            previous_logged_gradient = gradient.clone()
            train_correct = logits.argmax(dim=-1) == targets
            train_acc = float(train_correct.float().mean().detach().cpu())
            train_sequence_acc = float(train_correct.all(dim=1).float().mean().detach().cpu())
        optimizer.step()

        if step % config.log_every == 0:
            val_loss, val_acc, val_sequence_acc = evaluate(
                model, validation_packed, config.eval_batch_size, device
            )
            full_train_acc = np.nan
            full_train_sequence_acc = np.nan
            # Evaluate the entire training split near either transition.  This
            # makes t_memo independent of the current stochastic mini-batch.
            if train_acc >= config.target_train_acc or val_acc >= config.target_val_acc:
                _, full_train_acc, full_train_sequence_acc = evaluate(
                    model, train_packed, config.eval_batch_size, device
                )
            record = {
                "step": step, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
                "train_sequence_acc": train_sequence_acc,
                "val_sequence_acc": val_sequence_acc,
                "full_train_acc": full_train_acc,
                "full_train_sequence_acc": full_train_sequence_acc,
                "optimizer_weight_decay": config.weight_decay,
                "weight_norm": weight_norm(model),
                "grad_norm": grad_norm, "embed_grad_norm": float(model.token_embedding.weight.grad.detach().norm().cpu()),
                "grad_cosine": grad_cosine,
            }
            logs.append(record)
            progress.set_postfix(train=f"{record['train_loss']:.3f}", val_acc=f"{val_acc:.3f}")

            if full_train_acc >= config.target_train_acc:
                memorization_logs += 1
            else:
                memorization_logs = 0
            if (first_memorization_step is None
                    and memorization_logs >= config.memorization_patience_logs):
                first_memorization_step = step
                print(f"S_{n}, seed={seed}: stable memorization at step {step}")

            if full_train_acc >= config.target_train_acc and val_acc >= config.target_val_acc:
                generalization_logs += 1
            else:
                generalization_logs = 0
            if (first_generalization_step is None
                    and generalization_logs >= config.generalization_patience_logs):
                first_generalization_step = step
                gap = (None if first_memorization_step is None
                       else step - first_memorization_step)
                if gap is not None and gap >= config.min_grok_gap_steps:
                    first_grok_step = step
                    stop_step = min(max_steps, step + config.post_grok_steps)
                    print(f"S_{n}, seed={seed}: genuine grokking at step {step} "
                          f"(measured gap={gap}); continuing to {stop_step}")
                else:
                    print(f"S_{n}, seed={seed}: early generalization at step {step} "
                          f"(measured gap={gap}); this run is NOT classified as grokking")
                    # The first stable generalisation time is already fixed;
                    # this run can never acquire a >=10k first-transition gap.
                    stop_step = step + 1

        if (step + 1) % config.checkpoint_every == 0 or step + 1 == max_steps:
            save_run_state(run_dir, {"next_step": step + 1, "model": model.state_dict(),
                                     "optimizer": optimizer.state_dict(), "logs": logs,
                                     "memorization_logs": memorization_logs,
                                     "generalization_logs": generalization_logs,
                                     "first_memorization_step": first_memorization_step,
                                     "first_generalization_step": first_generalization_step,
                                     "first_grok_step": first_grok_step,
                                     "stop_step": stop_step}, logs)

        if stop_step is not None and step + 1 >= stop_step:
            save_run_state(run_dir, {"next_step": step + 1, "model": model.state_dict(),
                                     "optimizer": optimizer.state_dict(), "logs": logs,
                                     "memorization_logs": memorization_logs,
                                     "generalization_logs": generalization_logs,
                                     "first_memorization_step": first_memorization_step,
                                     "first_generalization_step": first_generalization_step,
                                     "first_grok_step": first_grok_step,
                                     "stop_step": stop_step}, logs)
            break

    final_step = int(logs[-1]["step"]) if logs else start_step
    success = first_grok_step is not None
    completed_marker.write_text(json.dumps({
        "completed": True, "success": success, "final_step": final_step,
        "first_memorization_step": first_memorization_step,
        "first_generalization_step": first_generalization_step,
        "first_grok_step": first_grok_step,
        "grok_gap_steps": (None if first_generalization_step is None or first_memorization_step is None
                           else first_generalization_step - first_memorization_step),
        "log_rows": len(logs),
        "best_val_acc": max((row["val_acc"] for row in logs), default=None),
    }, indent=2), encoding="utf-8")
    del model, optimizer, train_packed, validation_packed, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return run_dir


def run_sweep(config: SweepConfig) -> list[Path]:
    """Run every requested (n, seed) pair, resuming safely after Colab resets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("Warning: a Colab GPU runtime is strongly recommended.")
    output_root = Path(config.output_root)
    protocol_root = output_root / config.protocol_name
    protocol_root.mkdir(parents=True, exist_ok=True)
    (protocol_root / "sweep_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    results = []
    for n in config.n_values:
        for seed in config.seeds:
            results.append(train_one(int(n), int(seed), config, device))
    return results


def calibrate_train_fraction(
    base_config: SweepConfig,
    fractions: tuple[float, ...] = (0.5, 0.4, 0.3, 0.2),
) -> pd.DataFrame:
    """Pilot several data difficulties without changing model or optimiser.

    A fraction is accepted only when the measured stable-generalisation time
    follows stable memorisation by at least ``min_grok_gap_steps``.  Results
    live in separate protocol directories and can be inspected independently.
    This is protocol calibration; final claims must use fresh held-out seeds.
    """
    rows: list[dict] = []
    original_protocol = base_config.protocol_name
    original_fraction = base_config.train_fraction
    original_fraction_by_n = base_config.train_fraction_by_n
    original_n_values = base_config.n_values
    base_config.train_fraction_by_n = None
    for n in original_n_values:
        base_config.n_values = (int(n),)
        for fraction in fractions:
            percentage = int(round(100 * fraction))
            base_config.protocol_name = f"{original_protocol}_calib_f{percentage:02d}"
            base_config.train_fraction = fraction
            paths = run_sweep(base_config)
            for run_dir in paths:
                metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
                completed = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
                rows.append({
                    "n": int(metadata["n"]), "fraction": fraction,
                    "seed": int(run_dir.name.removeprefix("seed_")),
                    "success": completed["success"],
                    "memo_step": completed.get("first_memorization_step"),
                    "generalization_step": completed.get("first_generalization_step"),
                    "gap_steps": completed.get("grok_gap_steps"),
                    "best_val_acc": completed.get("best_val_acc"),
                    "run_dir": str(run_dir),
                })
            fraction_rows = [row for row in rows if row["n"] == n and row["fraction"] == fraction]
            if fraction_rows and all(row["success"] for row in fraction_rows):
                print(f"S_{n}: accepted train_fraction={fraction}; "
                      f"gaps={[row['gap_steps'] for row in fraction_rows]}")
                break
    base_config.protocol_name = original_protocol
    base_config.train_fraction = original_fraction
    base_config.train_fraction_by_n = original_fraction_by_n
    base_config.n_values = original_n_values
    table = pd.DataFrame(rows)
    destination = Path(base_config.output_root) / f"{original_protocol}_calibration.csv"
    table.to_csv(destination, index=False)
    print(f"Calibration table written to {destination}")
    return table


def fractions_from_calibration(table: pd.DataFrame) -> dict[int, float]:
    """Return the first successful calibrated fraction for each n."""
    accepted: dict[int, float] = {}
    for n, group in table[table["success"]].groupby("n", sort=True):
        accepted[int(n)] = float(group.iloc[0]["fraction"])
    missing = sorted(set(map(int, table["n"].unique())) - set(accepted))
    if missing:
        raise ValueError(
            f"No genuine >=10k-gap regime was found for S_n={missing}. "
            "Extend the fraction grid or max_steps before the final sweep."
        )
    return accepted


if __name__ == "__main__":
    # Conservative smoke test.  Edit this block only for command-line runs;
    # Colab users should instantiate SweepConfig in a notebook cell.
    run_sweep(SweepConfig(n_values=(3,), seeds=(42,), min_steps=500, max_steps=500,
                          max_epochs=1))
