"""Modular addition, and the prompt-and-split machinery every algebraic task shares.

Each task turns a binary operation on a finite set into next-token prediction: the prompt
is ``[a, b, =]`` and the target is ``a * b``, so the vocabulary is the set itself plus the
one ``=`` symbol. ``a + b (mod p)`` is the abelian case of appendix O; ``groups.py`` adds
composition in ``S_n`` on top of the same two helpers.

**The random stream is one stream, and it is short.** The builder seeds the global torch
generator and then draws *exactly one* ``torch.randperm`` for the split. The model is
constructed afterwards and continues that same stream, so the split, the initial weights
and the mini-batch order are one chain: an extra draw anywhere in here changes the initial
weights of every published run, and initial weights decide whether these runs generalise
at all. That is also why the trajectory sketch takes its hashes from NumPy and restores
the torch state around its forward passes -- see ``actdim.sketch.probe``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class Task:
    """A materialised task: the split tensors plus what the model needs to size itself."""

    name: str
    X_train: torch.Tensor        # (n_train, 3) int64 prompts [a, b, =]
    Y_train: torch.Tensor        # (n_train,)   int64 targets
    X_val: torch.Tensor
    Y_val: torch.Tensor
    num_classes: int             # |set|: the modulus p, or the group order n!
    num_total: int               # pairs before the split

    @property
    def vocab_size(self) -> int:
        """``num_classes`` symbols plus ``=``, whose id is ``num_classes``."""
        return self.num_classes + 1

    @property
    def n_ctx(self) -> int:
        return int(self.X_train.shape[1])

    def __repr__(self) -> str:
        return (f"Task({self.name}, classes={self.num_classes}, pairs={self.num_total}, "
                f"train={len(self.X_train)}, val={len(self.X_val)})")


def split(name: str, a: torch.Tensor, b: torch.Tensor, answers: torch.Tensor,
          num_classes: int, fraction: float, device: Any) -> Task:
    """Assemble ``[a, b, =] -> answer`` and cut a uniform random train/validation split.

    Draws exactly one ``torch.randperm``. See the module docstring for why the count
    matters.
    """
    equals = torch.full_like(a, num_classes)
    prompts = torch.stack([a, b, equals], dim=1).to(device)
    answers = answers.to(device)

    num_total = len(prompts)
    num_train = int(fraction * num_total)
    if not 0 < num_train < num_total:
        raise ValueError(
            f"fraction={fraction} leaves {num_train}/{num_total} training pairs; pick a "
            f"fraction that yields a non-empty split on both sides")

    order = torch.randperm(num_total)
    return Task(
        name=name,
        X_train=prompts[order[:num_train]],
        Y_train=answers[order[:num_train]],
        X_val=prompts[order[num_train:]],
        Y_val=answers[order[num_train:]],
        num_classes=num_classes,
        num_total=num_total,
    )


def sample_indices(population: int, size: int, seed: Optional[int]) -> np.ndarray:
    """A sorted random subset of ``range(population)``, drawn from NumPy.

    Deliberately not torch: subsampling the product set must not shift the global torch
    stream the split and the initialisation share, or ``max_pairs`` would silently change
    the initial weights.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(population, size=size, replace=False))


def modular_addition(p: int = 113, fraction: float = 0.3, seed: Optional[int] = 42,
                     device: Any = "cpu", max_pairs: Optional[int] = None) -> Task:
    """``a + b (mod p)`` over all ``p^2`` pairs."""
    if seed is not None:
        torch.manual_seed(seed)

    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    answers = (a + b) % p

    if max_pairs is not None and max_pairs < len(a):
        keep = torch.from_numpy(sample_indices(len(a), max_pairs, seed))
        a, b, answers = a[keep], b[keep], answers[keep]

    return split(f"mod_add_p{p}", a, b, answers, p, fraction, device)


def from_config(config: Any, device: Any = "cpu") -> Task:
    """Build the task a run configuration names.

    Imported lazily inside the function so that ``groups`` -- which pulls in the
    permutation algebra -- is not loaded by a run that only does modular arithmetic.
    """
    from . import groups

    builders = {"modular_addition": modular_addition, "symmetric_group": groups.symmetric_group}
    extra = {"modular_addition": ("p",), "symmetric_group": ("n",)}
    if config.task not in builders:
        raise KeyError(f"unknown task '{config.task}'. Available: {', '.join(builders)}")
    kwargs = {name: getattr(config, name) for name in extra[config.task]}
    return builders[config.task](
        fraction=config.fraction,
        seed=config.seed,
        device=device,
        max_pairs=config.max_pairs,
        **kwargs,
    )
