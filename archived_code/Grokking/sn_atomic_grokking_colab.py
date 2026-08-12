"""Grokking sweep on atomic Cayley-table prediction for symmetric groups.

This protocol intentionally follows the original successful S_5 experiment:
each permutation is an opaque group-element ID, and the target is the opaque ID
of the product.  Unlike the token-sequence protocol, it does not expose the
coordinate-wise shortcut c[k] = a[b[k]].

Recommended primary range: S_5,...,S_8.  S_3/S_4 are too small for a stable
long memorisation/generalisation gap; S_9 is optional and expensive; an atomic
S_10 output layer is impractical on a standard Colab T4 with AdamW.
"""

from __future__ import annotations

import gc
import json
import math
import random
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from grokking_model import GrokkingTransformer


@dataclass
class AtomicConfig:
    output_root: str = "/content/drive/MyDrive/grokking_sn_atomic"
    protocol_name: str = "atomic_native_v2_f30"
    n_values: tuple[int, ...] = (5, 6)
    max_atomic_n: int = 8
    seeds: tuple[int, ...] = (42,)
    train_fraction: float = 0.3
    # Native n!+1 embedding/output sizes reproduce the original S_5/S_6 task.
    # The Transformer core and all optimiser settings remain identical.
    fixed_vocabulary: bool = False
    # Full tables are used through S_5.  Larger groups use a deterministic,
    # disjoint sample because |S_n|^2 grows factorially.
    sampled_pairs_by_n: dict[int, int] | None = None
    batch_size: int = 256
    eval_batch_size: int = 1024
    monitor_pairs: int = 8_192
    diagnostic_every: int = 5_000
    max_epochs: int = 4_000
    min_steps: int = 30_000
    max_steps: int = 600_000
    log_every: int = 20
    checkpoint_every: int = 1_000
    learning_rate: float = 1e-3
    weight_decay: float = 2e-1
    betas: tuple[float, float] = (0.9, 0.98)
    d_model: int = 128
    d_mlp: int = 512
    d_head: int = 32
    n_heads: int = 4
    target_train_acc: float = 0.99
    target_val_acc: float = 0.95
    patience_logs: int = 10
    required_gap_steps: int = 10_000
    post_grok_steps: int = 5_000
    num_workers: int = 2
    force_restart: bool = False
    # Kaggle inputs are read-only.  Attach the previous notebook output as a
    # Dataset and list its root here; checkpoints are copied to working space.
    resume_search_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sampled_pairs_by_n is None:
            # S_5: full 14,400 pairs.  Larger tasks need more coverage; these
            # are data-budget choices, not optimiser/model changes.
            self.sampled_pairs_by_n = {
                5: 14_400,
                6: math.factorial(6) ** 2,
                7: 500_000,
                8: 500_000,
                9: 500_000,
            }


def factorials(n: int) -> list[int]:
    values = [1]
    for k in range(1, n + 1):
        values.append(values[-1] * k)
    return values


def unrank_permutation(rank: int, n: int, facts: list[int] | None = None) -> list[int]:
    facts = factorials(n) if facts is None else facts
    remaining = list(range(n))
    result = []
    for width in range(n, 0, -1):
        index, rank = divmod(rank, facts[width - 1])
        result.append(remaining.pop(index))
    return result


def rank_permutation(permutation: list[int], facts: list[int] | None = None) -> int:
    n = len(permutation)
    facts = factorials(n) if facts is None else facts
    remaining = list(range(n))
    rank = 0
    for index, value in enumerate(permutation):
        position = remaining.index(value)
        rank += position * facts[n - index - 1]
        remaining.pop(position)
    return rank


def product_id(left_id: int, right_id: int, n: int, facts: list[int]) -> int:
    left = unrank_permutation(left_id, n, facts)
    right = unrank_permutation(right_id, n, facts)
    return rank_permutation([left[right[k]] for k in range(n)], facts)


def make_atomic_split(n: int, fraction: float, requested_pairs: int, seed: int) -> tuple[torch.Tensor, ...]:
    group_size = math.factorial(n)
    total_pairs = group_size * group_size
    sampled_pairs = min(total_pairs, requested_pairs)
    rng = random.Random(seed)
    pair_ids = rng.sample(range(total_pairs), sampled_pairs)
    facts = factorials(n)
    examples = torch.empty((sampled_pairs, 3), dtype=torch.long)
    for row, pair_id in enumerate(pair_ids):
        left_id, right_id = divmod(pair_id, group_size)
        examples[row, 0] = left_id
        examples[row, 1] = right_id
        examples[row, 2] = product_id(left_id, right_id, n, facts)
    train_count = max(1, min(sampled_pairs - 1, round(fraction * sampled_pairs)))
    equals = group_size

    def pack(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([rows[:, 0], rows[:, 1], torch.full_like(rows[:, 0], equals)], dim=1)
        return x, rows[:, 2]

    train_x, train_y = pack(examples[:train_count])
    val_x, val_y = pack(examples[train_count:])
    metadata = {
        "n": n, "group_size": group_size, "vocabulary_size": group_size + 1,
        "total_pairs": total_pairs, "sampled_pairs": sampled_pairs,
        "train_pairs": len(train_x), "validation_pairs": len(val_x),
        "train_fraction": fraction, "split_seed": seed,
        "task": "opaque_group_id_cayley_product",
    }
    return train_x, train_y, val_x, val_y, metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_weight_norm(model: nn.Module) -> float:
    return float(torch.sqrt(sum(parameter.detach().pow(2).sum() for parameter in model.parameters())).cpu())


class FixedVocabularyGrokkingTransformer(GrokkingTransformer):
    """Same parameter shapes for every S_n, with an active vocabulary mask."""

    def __init__(self, fixed_vocab: int, active_vocab: int, **kwargs) -> None:
        super().__init__(d_vocab=fixed_vocab, **kwargs)
        self.active_vocab = active_vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.pos_embed(x)
        x = self.block(x)
        # Computing only active logits avoids an unnecessary S_8-sized output
        # tensor in smaller tasks while parameter shapes remain identical.
        return (x[:, -1] @ self.unembed.W_U[:, :self.active_vocab])


def active_weight_norm(model: GrokkingTransformer, active_vocab: int | None = None) -> float:
    """Norm of shared core plus input/output rows active for the current S_n."""
    active = getattr(model, "active_vocab", active_vocab)
    if active is None:
        raise ValueError("active vocabulary size is required")
    total = torch.zeros((), device=model.unembed.W_U.device)
    for name, parameter in model.named_parameters():
        if name == "embed.W_E":
            total = total + parameter[:, :active].detach().pow(2).sum()
        elif name == "unembed.W_U":
            total = total + parameter[:, :active].detach().pow(2).sum()
        else:
            total = total + parameter.detach().pow(2).sum()
    return float(torch.sqrt(total).cpu())


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device, non_blocking=True)
        yb = y[start:start + batch_size].to(device, non_blocking=True)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return loss_sum / len(x), correct / len(x)


def save_state(run_dir: Path, step: int, model: nn.Module, optimizer: torch.optim.Optimizer,
               logs: list[dict], state: dict) -> None:
    payload = {"next_step": step + 1, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "logs": logs, **state}
    torch.save(payload, run_dir / "checkpoint.pt")
    pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)


def import_resume_artifacts(run_dir: Path, config: AtomicConfig, n: int, seed: int) -> None:
    """Copy a matching checkpoint from an attached Kaggle Dataset if present."""
    if (run_dir / "checkpoint.pt").exists() or config.force_restart:
        return
    suffix = Path(config.protocol_name) / f"S_{n}" / f"seed_{seed}"
    for root_text in config.resume_search_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        candidates = [path for path in root.rglob("checkpoint.pt")
                      if tuple(path.parent.parts[-3:]) == tuple(suffix.parts[-3:])]
        if not candidates:
            continue
        source_dir = max(candidates, key=lambda path: path.stat().st_mtime).parent
        for filename in ("checkpoint.pt", "training_log.csv", "metadata.json", "COMPLETED.json"):
            source = source_dir / filename
            if source.exists():
                shutil.copy2(source, run_dir / filename)
        print(f"Imported resume artifacts from {source_dir}")
        return


def train_atomic(n: int, seed: int, config: AtomicConfig, device: torch.device) -> Path:
    if n < 5:
        raise ValueError("S_3/S_4 are excluded from the primary long-gap protocol")
    if n > config.max_atomic_n:
        raise ValueError(f"S_{n} exceeds fixed atomic vocabulary S_{config.max_atomic_n}")
    run_dir = Path(config.output_root) / config.protocol_name / f"S_{n}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    import_resume_artifacts(run_dir, config, n, seed)
    completed_path = run_dir / "COMPLETED.json"
    if completed_path.exists() and not config.force_restart:
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("success"):
            print(f"Skipping completed genuine-grokking run: {run_dir}")
            return run_dir

    set_seed(seed)
    requested = config.sampled_pairs_by_n[n]
    train_x, train_y, val_x, val_y, data_meta = make_atomic_split(
        n, config.train_fraction, requested, seed
    )
    steps_per_epoch = math.ceil(len(train_x) / min(config.batch_size, len(train_x)))
    max_steps = min(config.max_steps, max(config.min_steps, steps_per_epoch * config.max_epochs))
    active_vocab = data_meta["vocabulary_size"]
    if config.fixed_vocabulary:
        model_vocab = math.factorial(config.max_atomic_n) + 1
        model = FixedVocabularyGrokkingTransformer(
            fixed_vocab=model_vocab, active_vocab=active_vocab,
            d_model=config.d_model, d_mlp=config.d_mlp,
            d_head=config.d_head, num_heads=config.n_heads, n_ctx=3,
        ).to(device)
    else:
        model_vocab = active_vocab
        model = GrokkingTransformer(
            d_vocab=model_vocab, d_model=config.d_model, d_mlp=config.d_mlp,
            d_head=config.d_head, num_heads=config.n_heads, n_ctx=3,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=config.betas)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        **data_meta, "seed": seed, "protocol": config.protocol_name,
        "max_steps": max_steps, "steps_per_epoch": steps_per_epoch,
        "model": {"class": "GrokkingTransformer", "d_model": config.d_model,
                  "d_mlp": config.d_mlp, "d_head": config.d_head,
                  "n_heads": config.n_heads, "n_ctx": 3,
                  "parameter_count": parameter_count,
                  "fixed_vocabulary": config.fixed_vocabulary,
                  "model_vocabulary_size": model_vocab,
                  "active_vocabulary_size": active_vocab,
                  "max_atomic_n": config.max_atomic_n},
        "optimizer": {"class": "AdamW", "lr": config.learning_rate,
                      "weight_decay": config.weight_decay, "betas": config.betas},
        "grokking": {"train_threshold": config.target_train_acc,
                     "val_threshold": config.target_val_acc,
                     "required_gap_steps": config.required_gap_steps},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config.batch_size,
                        shuffle=True, pin_memory=device.type == "cuda",
                        num_workers=config.num_workers,
                        persistent_workers=config.num_workers > 0)
    iterator = iter(loader)
    logs: list[dict] = []
    start_step = 0
    state = {"memo_logs": 0, "gen_logs": 0, "memo_step": None,
             "generalization_step": None, "grok_step": None, "stop_step": None}
    checkpoint = run_dir / "checkpoint.pt"
    if checkpoint.exists() and not config.force_restart:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        logs = saved["logs"]
        start_step = saved["next_step"]
        for key in state:
            state[key] = saved.get(key, state[key])

    previous_gradient = None
    monitor_train_x = train_x[:config.monitor_pairs]
    monitor_train_y = train_y[:config.monitor_pairs]
    monitor_val_x = val_x[:config.monitor_pairs]
    monitor_val_y = val_y[:config.monitor_pairs]
    progress = tqdm(range(start_step, max_steps), desc=f"Atomic S_{n}, seed {seed}")
    for step in progress:
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, yb = next(iterator)
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        if step % config.log_every == 0:
            gradient = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
            grad_norm = float(gradient.norm().cpu())
            grad_cosine = (0.0 if previous_gradient is None else
                           float(F.cosine_similarity(gradient, previous_gradient, dim=0).cpu()))
            previous_gradient = gradient.clone()
            batch_train_acc = float((logits.argmax(dim=1) == yb).float().mean().detach().cpu())
        optimizer.step()  # exactly one update

        if step % config.log_every == 0:
            val_loss, val_acc = evaluate(
                model, monitor_val_x, monitor_val_y, config.eval_batch_size, device
            )
            full_train_loss = np.nan
            full_train_acc = np.nan
            if batch_train_acc >= config.target_train_acc or val_acc >= config.target_val_acc:
                full_train_loss, full_train_acc = evaluate(
                    model, monitor_train_x, monitor_train_y, config.eval_batch_size, device
                )
            active_norm = active_weight_norm(model, active_vocab)
            logs.append({
                "step": step, "train_loss": float(loss.detach().cpu()),
                "full_train_loss": full_train_loss, "val_loss": val_loss,
                "train_acc": batch_train_acc, "full_train_acc": full_train_acc,
                "val_acc": val_acc, "weight_norm": model_weight_norm(model),
                "active_weight_norm": active_norm,
                "active_weight_norm_per_sqrt_parameter": active_norm / math.sqrt(parameter_count),
                "grad_norm": grad_norm,
                "embed_grad_norm": float(model.embed.W_E.grad.detach().norm().cpu()),
                "grad_cosine": grad_cosine,
            })
            progress.set_postfix(train=f"{loss.item():.3g}", val_acc=f"{val_acc:.3f}")

            if step % config.diagnostic_every == 0:
                diagnostic_train_loss, diagnostic_train_acc = evaluate(
                    model, monitor_train_x, monitor_train_y,
                    config.eval_batch_size, device,
                )
                logs[-1]["diagnostic_train_loss"] = diagnostic_train_loss
                logs[-1]["diagnostic_train_acc"] = diagnostic_train_acc
                print(f"Atomic S_{n} step={step}: monitor train loss="
                      f"{diagnostic_train_loss:.3f}, train acc={diagnostic_train_acc:.4f}, "
                      f"val acc={val_acc:.4f}")
            else:
                logs[-1]["diagnostic_train_loss"] = np.nan
                logs[-1]["diagnostic_train_acc"] = np.nan

            state["memo_logs"] = state["memo_logs"] + 1 if full_train_acc >= config.target_train_acc else 0
            if state["memo_step"] is None and state["memo_logs"] >= config.patience_logs:
                _, confirmed_train_acc = evaluate(
                    model, train_x, train_y, config.eval_batch_size, device
                )
                if confirmed_train_acc >= config.target_train_acc:
                    state["memo_step"] = step
                    print(f"Atomic S_{n}: stable memorization at {step} "
                          f"(full train acc={confirmed_train_acc:.4f})")
                else:
                    state["memo_logs"] = 0
            state["gen_logs"] = state["gen_logs"] + 1 if val_acc >= config.target_val_acc else 0
            if state["generalization_step"] is None and state["gen_logs"] >= config.patience_logs:
                confirmed_val_loss, confirmed_val_acc = evaluate(
                    model, val_x, val_y, config.eval_batch_size, device
                )
                if confirmed_val_acc >= config.target_val_acc:
                    state["generalization_step"] = step
                    gap = None if state["memo_step"] is None else step - state["memo_step"]
                    if gap is not None and gap >= config.required_gap_steps:
                        state["grok_step"] = step
                        state["stop_step"] = min(max_steps, step + config.post_grok_steps)
                        print(f"Atomic S_{n}: genuine grokking at {step}, gap={gap}, "
                              f"full val acc={confirmed_val_acc:.4f}")
                    else:
                        state["stop_step"] = step + 1
                        print(f"Atomic S_{n}: early generalization, gap={gap}; rejecting run")
                else:
                    state["gen_logs"] = 0
                    print(f"Atomic S_{n}: monitor crossed threshold but full val acc="
                          f"{confirmed_val_acc:.4f}; continuing")

        if (step + 1) % config.checkpoint_every == 0 or step + 1 == max_steps:
            save_state(run_dir, step, model, optimizer, logs, state)
        if state["stop_step"] is not None and step + 1 >= state["stop_step"]:
            save_state(run_dir, step, model, optimizer, logs, state)
            break

    final_val_loss, final_val_acc = evaluate(
        model, val_x, val_y, config.eval_batch_size, device
    )
    gap = (None if state["memo_step"] is None or state["generalization_step"] is None
           else state["generalization_step"] - state["memo_step"])
    completed_path.write_text(json.dumps({
        "completed": True, "success": state["grok_step"] is not None,
        "outcome": (
            "genuine_grokking" if state["grok_step"] is not None else
            "generalization_before_memorization" if state["generalization_step"] is not None and state["memo_step"] is None else
            "early_generalization" if state["generalization_step"] is not None else
            "memorized_without_generalization" if state["memo_step"] is not None else
            "not_memorized"
        ),
        "memo_step": state["memo_step"],
        "generalization_step": state["generalization_step"],
        "grok_gap_steps": gap, "grok_step": state["grok_step"],
        "best_val_acc": max((row["val_acc"] for row in logs), default=None),
        "final_full_val_loss": final_val_loss,
        "final_full_val_acc": final_val_acc,
        "final_step": logs[-1]["step"] if logs else start_step,
    }, indent=2), encoding="utf-8")
    del model, optimizer, loader, train_x, train_y, val_x, val_y
    torch.cuda.empty_cache()
    gc.collect()
    return run_dir


def run_atomic_sweep(config: AtomicConfig) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    root = Path(config.output_root) / config.protocol_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    results = []
    for n in config.n_values:
        for seed in config.seeds:
            results.append(train_atomic(n, seed, config, device))
    return results


def calibrate_atomic_fraction(
    base_config: AtomicConfig,
    fractions: tuple[float, ...] = (0.25, 0.22, 0.20, 0.18),
) -> pd.DataFrame:
    """Find a data fraction with a measured genuine grokking gap.

    Model, optimizer, seed and pair universe are kept fixed.  Each fraction
    receives a separate protocol directory and a fresh initialization with the
    same seed.  The first successful fraction is returned, but it remains a
    calibration result and must be confirmed on unused seeds.
    """
    if len(base_config.n_values) != 1 or len(base_config.seeds) != 1:
        raise ValueError("Calibration requires exactly one n and one pilot seed")
    rows: list[dict] = []
    base_protocol = base_config.protocol_name
    for fraction in fractions:
        config = deepcopy(base_config)
        config.train_fraction = float(fraction)
        config.protocol_name = f"{base_protocol}_f{round(100 * fraction):02d}"
        # Never import a checkpoint from another fraction/protocol.
        paths = run_atomic_sweep(config)
        run_dir = paths[0]
        completed = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        row = {
            "n": metadata["n"], "seed": metadata["seed"], "fraction": fraction,
            "train_pairs": metadata["train_pairs"],
            "validation_pairs": metadata["validation_pairs"],
            "success": completed["success"], "outcome": completed["outcome"],
            "memo_step": completed["memo_step"],
            "generalization_step": completed["generalization_step"],
            "gap_steps": completed["grok_gap_steps"],
            "best_monitor_val_acc": completed["best_val_acc"],
            "final_full_val_acc": completed["final_full_val_acc"],
            "run_dir": str(run_dir),
        }
        rows.append(row)
        print("Calibration result:", row)
        pd.DataFrame(rows).to_csv(
            Path(base_config.output_root) / f"{base_protocol}_fraction_calibration.csv",
            index=False,
        )
        if completed["success"]:
            print(f"Accepted pilot fraction={fraction}; confirm it on fresh seeds")
            break
    return pd.DataFrame(rows)


def calibrate_atomic_weight_decay(
    base_config: AtomicConfig,
    weight_decays: tuple[float, ...] = (0.10, 0.05, 0.02, 0.01),
) -> pd.DataFrame:
    """Locate the stationary-AdamW regime between memorisation and early generalisation.

    Dataset, fraction, architecture, learning rate and seed remain fixed.  Each
    decay gets a fresh run and its own checkpoint directory.  Calibration is
    performed on one pilot seed; the selected decay must subsequently be
    confirmed for every target n on unused seeds.
    """
    if len(base_config.n_values) != 1 or len(base_config.seeds) != 1:
        raise ValueError("Weight-decay calibration requires exactly one n and one pilot seed")
    rows: list[dict] = []
    base_protocol = base_config.protocol_name
    for decay in weight_decays:
        config = deepcopy(base_config)
        config.weight_decay = float(decay)
        tag = str(decay).replace(".", "p")
        config.protocol_name = f"{base_protocol}_wd{tag}"
        run_dir = run_atomic_sweep(config)[0]
        completed = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        row = {
            "n": metadata["n"], "seed": metadata["seed"],
            "train_fraction": metadata["train_fraction"],
            "weight_decay": decay, "train_pairs": metadata["train_pairs"],
            "validation_pairs": metadata["validation_pairs"],
            "success": completed["success"], "outcome": completed["outcome"],
            "memo_step": completed["memo_step"],
            "generalization_step": completed["generalization_step"],
            "gap_steps": completed["grok_gap_steps"],
            "best_monitor_val_acc": completed["best_val_acc"],
            "final_full_val_acc": completed["final_full_val_acc"],
            "run_dir": str(run_dir),
        }
        rows.append(row)
        print("Weight-decay calibration result:", row)
        pd.DataFrame(rows).to_csv(
            Path(base_config.output_root) / f"{base_protocol}_weight_decay_calibration.csv",
            index=False,
        )
        if completed["success"]:
            print(f"Accepted pilot weight_decay={decay}; confirm it on S_5/S_6 with fresh seeds")
            break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_atomic_sweep(AtomicConfig(n_values=(5,), seeds=(42,), min_steps=10,
                                  max_steps=10, max_epochs=1, num_workers=0))
