"""Algorithmic grokking tasks.

Every task turns a binary operation on a finite set into next-token prediction:
the prompt is ``[a, b, =]`` and the target is ``a * b``, so the vocabulary is the
set itself plus one ``=`` token.  ``a + b (mod p)`` is the abelian baseline;
composition in ``S_n`` is the non-abelian escalation of Sec. 4.2 of the paper.

Reproducibility note
--------------------
Like the original ``grokking_utils.py``, the builders call ``torch.manual_seed``
themselves and then draw exactly one ``torch.randperm`` for the train/val split.
The model is constructed *after* the task, so its initialisation continues the
same RNG stream -- changing the number of draws made here silently changes the
initial weights.  Do not add torch RNG calls to these functions.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .groups import MAX_FULL_PAIRS, SymmetricGroup


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
    def vocab_size(self):
        """``num_classes`` symbols plus the ``=`` token, whose id is ``num_classes``."""
        return self.num_classes + 1

    @property
    def n_ctx(self):
        return self.X_train.shape[1]

    def __repr__(self):
        return (f"Task({self.name}, classes={self.num_classes}, pairs={self.num_total}, "
                f"train={len(self.X_train)}, val={len(self.X_val)})")


def _split(name, a, b, answers, num_classes, fraction, device):
    """Assemble ``[a, b, =] -> answer`` prompts and cut a random train/val split.

    Draws exactly one ``torch.randperm`` -- see the module docstring.
    """
    equals = torch.full_like(a, num_classes)
    prompts = torch.stack([a, b, equals], dim=1).to(device)
    answers = answers.to(device)

    num_total = len(prompts)
    num_train = int(fraction * num_total)
    if not 0 < num_train < num_total:
        raise ValueError(
            f"fraction={fraction} leaves {num_train}/{num_total} training pairs; "
            f"pick a fraction that yields a non-empty split on both sides"
        )

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


def modular_addition(p=113, fraction=0.3, seed=42, device="cpu", max_pairs=None):
    """``a + b (mod p)`` over all ``p^2`` pairs.  The task of Sec. 4.1."""
    if seed is not None:
        torch.manual_seed(seed)

    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    answers = (a + b) % p

    if max_pairs is not None and max_pairs < len(a):
        keep = torch.from_numpy(_sample_indices(len(a), max_pairs, seed))
        a, b, answers = a[keep], b[keep], answers[keep]

    return _split(f"mod_add_p{p}", a, b, answers, p, fraction, device)


def symmetric_group(n=5, fraction=0.5, seed=42, device="cpu", max_pairs=None):
    """Composition ``a . b`` in ``S_n`` over all ``(n!)^2`` pairs.  Sec. 4.2.

    ``max_pairs`` samples the product set instead of materialising it, which is
    what makes ``n >= 7`` (25M+ pairs) representable at all.
    """
    if seed is not None:
        torch.manual_seed(seed)

    group = SymmetricGroup(n)
    order = group.order
    pairs = order * order

    if max_pairs is not None and max_pairs < pairs:
        flat = _sample_indices(pairs, max_pairs, seed)
        a_ids, b_ids = np.divmod(flat, order)
    elif pairs > MAX_FULL_PAIRS:
        raise ValueError(
            f"S_{n} has {order} elements -> {pairs} pairs, above the {MAX_FULL_PAIRS} "
            f"guard (and far past a trainable dataset). Pass max_pairs=... to sample "
            f"the product set."
        )
    else:
        a_ids = np.repeat(np.arange(order), order)
        b_ids = np.tile(np.arange(order), order)

    answers = group.compose(a_ids, b_ids)
    return _split(
        f"S_{n}",
        torch.from_numpy(np.ascontiguousarray(a_ids)),
        torch.from_numpy(np.ascontiguousarray(b_ids)),
        torch.from_numpy(answers),
        order,
        fraction,
        device,
    )


def _sample_indices(population, size, seed):
    """A sorted random subset of ``range(population)``, drawn from NumPy's RNG.

    Deliberately *not* torch: sampling must not shift the global torch RNG stream
    that the train/val split and the weight initialisation share.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(population, size=size, replace=False))


TASKS = {
    "modular_addition": modular_addition,
    "symmetric_group": symmetric_group,
}

_TASK_ARGS = {
    "modular_addition": ("p",),
    "symmetric_group": ("n",),
}


def from_config(config, device="cpu"):
    """Build the task described by a :class:`~grok.config.RunConfig`."""
    if config.task not in TASKS:
        raise KeyError(f"unknown task '{config.task}'. Available: {', '.join(TASKS)}")
    kwargs = {name: getattr(config, name) for name in _TASK_ARGS[config.task]}
    return TASKS[config.task](
        fraction=config.fraction,
        seed=config.seed,
        device=device,
        max_pairs=config.max_pairs,
        **kwargs,
    )
