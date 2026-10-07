"""Composition in the symmetric group, and the algebra it needs.

The non-abelian task of appendix O: the prompt is a pair of permutations and the target is
their composite. The algebra below is NumPy only, so the Cayley table can be checked
without a deep-learning install, and the task builder borrows ``Task`` and ``split`` from
``modular`` rather than repeating them -- one split rule, one place it can be wrong.

Elements of ``S_n`` are addressed by their **lexicographic rank**, the position of the
permutation in ``itertools.permutations(range(n))``. That is the numbering the published
``S_5`` logs were produced with, so element ids stay comparable with them.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Optional

import numpy as np
import torch

from .modular import Task, sample_indices, split

MAX_FULL_PAIRS = 4_000_000
"""Refuse to materialise a Cayley table larger than this without ``max_pairs``.

``|S_6|^2 = 518,400`` fits comfortably; ``|S_7|^2 = 25,401,600`` does not, and is far past
the point where the full product set is a trainable dataset anyway.
"""

_CHUNK = 1 << 18
"""Pairs composed per block, to bound peak memory on the larger groups."""


def permutations(n: int) -> np.ndarray:
    """All ``n!`` permutations of ``0..n-1`` as an ``(n!, n)`` array, lexicographic."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def rank(perms: np.ndarray) -> np.ndarray:
    """Lexicographic rank of each row, the inverse of :func:`permutations`.

    The factorial number system (Lehmer code): the rank of ``p`` is
    ``sum_i c_i (n-1-i)!`` with ``c_i = #{j > i : p[j] < p[i]}``. Vectorised over rows, so
    it costs ``O(n^2)`` array operations however many permutations are ranked and needs no
    table of size ``n!``.
    """
    perms = np.asarray(perms, dtype=np.int64)
    n = perms.shape[1]
    factorials = np.array([math.factorial(k) for k in range(n)], dtype=np.int64)
    ranks = np.zeros(len(perms), dtype=np.int64)
    for i in range(n):
        smaller = (perms[:, i + 1:] < perms[:, i:i + 1]).sum(axis=1)
        ranks += smaller * factorials[n - 1 - i]
    return ranks


class SymmetricGroup:
    """``S_n`` with elements addressed by lexicographic rank."""

    def __init__(self, n: int):
        self.n = int(n)
        self.perms = permutations(self.n)
        self.order = len(self.perms)

    def __repr__(self) -> str:
        return f"SymmetricGroup(n={self.n}, order={self.order})"

    @property
    def identity(self) -> int:
        """Id of the identity permutation, always 0 in lexicographic order."""
        return 0

    def compose(self, left: Any, right: Any) -> np.ndarray:
        """Ids of ``a . b`` elementwise, with ``b`` applied first: ``c[k] = a[b[k]]``.

        The order follows the notebooks that produced the published logs. Reversing it
        gives a group that is isomorphic but whose labels differ, and the ``S_5`` logs
        would stop being comparable.
        """
        left = np.asarray(left, dtype=np.int64).ravel()
        right = np.asarray(right, dtype=np.int64).ravel()
        if left.shape != right.shape:
            raise ValueError(f"left/right length mismatch: {left.shape} vs {right.shape}")

        out = np.empty(len(left), dtype=np.int64)
        for start in range(0, len(left), _CHUNK):
            block = slice(start, start + _CHUNK)
            composed = np.take_along_axis(self.perms[left[block]], self.perms[right[block]],
                                          axis=1)
            out[block] = rank(composed)
        return out

    def table(self) -> np.ndarray:
        """The full ``(order, order)`` Cayley table, ``table[i, j] = i . j``."""
        pairs = self.order * self.order
        if pairs > MAX_FULL_PAIRS:
            raise ValueError(
                f"S_{self.n} has {self.order} elements -> {pairs} pairs, above the "
                f"{MAX_FULL_PAIRS} guard. Sample the product set instead (max_pairs=...).")
        left = np.repeat(np.arange(self.order), self.order)
        right = np.tile(np.arange(self.order), self.order)
        return self.compose(left, right).reshape(self.order, self.order)


def minimal_faithful_dimension(n: int) -> int:
    """Dimension of the smallest faithful irreducible representation of ``S_n``.

    ``n - 1`` for every ``n >= 2``, the standard representation. The two one-dimensional
    representations are unfaithful, and for ``S_4`` the two-dimensional one factors through
    ``S_4 / V_4 = S_3``, so it is unfaithful too. This is the algebraic floor a
    representation-counting account would predict; nothing in the estimator uses it, and it
    is here because the ``S_5`` runs are where that prediction would be read.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 if n == 1 else n - 1


def symmetric_group(n: int = 5, fraction: float = 0.5, seed: Optional[int] = 42,
                    device: Any = "cpu", max_pairs: Optional[int] = None) -> Task:
    """Composition ``a . b`` in ``S_n`` over all ``(n!)^2`` pairs.

    ``max_pairs`` samples the product set instead of materialising it, which is what makes
    ``n >= 7`` representable at all. The sampling is NumPy's, for the reason given in
    ``modular.sample_indices``.
    """
    if seed is not None:
        torch.manual_seed(seed)

    group = SymmetricGroup(n)
    order = group.order
    pairs = order * order

    if max_pairs is not None and max_pairs < pairs:
        flat = sample_indices(pairs, max_pairs, seed)
        a_ids, b_ids = np.divmod(flat, order)
    elif pairs > MAX_FULL_PAIRS:
        raise ValueError(
            f"S_{n} has {order} elements -> {pairs} pairs, above the {MAX_FULL_PAIRS} "
            f"guard (and far past a trainable dataset). Pass max_pairs=... to sample the "
            f"product set.")
    else:
        a_ids = np.repeat(np.arange(order), order)
        b_ids = np.tile(np.arange(order), order)

    answers = group.compose(a_ids, b_ids)
    return split(
        f"S_{n}",
        torch.from_numpy(np.ascontiguousarray(a_ids)),
        torch.from_numpy(np.ascontiguousarray(b_ids)),
        torch.from_numpy(answers),
        order,
        fraction,
        device,
    )
