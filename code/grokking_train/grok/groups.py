"""Finite-group algebra for the composition tasks -- NumPy only, no torch.

Kept free of torch so the algebra can be unit-tested (and the Cayley table
inspected) without a deep-learning install.

Elements of ``S_n`` are identified with their **lexicographic rank**, i.e. the
position of the permutation in ``list(itertools.permutations(range(n)))``.  That
is the numbering the original ``get_s5_composition_data`` used, so element ids
are comparable with the published ``S_5`` logs.
"""

import itertools
import math

import numpy as np

MAX_FULL_PAIRS = 4_000_000
"""Refuse to materialise a Cayley table larger than this without ``max_pairs``.

``|S_6|^2 = 518 400`` fits comfortably; ``|S_7|^2 = 25 401 600`` does not (and is
far past the point where the full product set is a trainable dataset anyway).
"""

_CHUNK = 1 << 18
"""Pairs composed per block, to bound peak memory on the larger groups."""


def permutations(n):
    """All ``n!`` permutations of ``0..n-1`` as an ``(n!, n)`` array, lexicographic."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def rank(perms):
    """Lexicographic rank of each row of ``perms`` -- the inverse of :func:`permutations`.

    Uses the factorial number system (Lehmer code): the rank of ``p`` is
    ``sum_i c_i (n-1-i)!`` where ``c_i = #{j > i : p[j] < p[i]}``.  Vectorised over
    rows, so it costs ``O(n^2)`` array ops regardless of how many permutations are
    ranked, and needs no ``n!``-sized lookup table.
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
    """The symmetric group ``S_n`` with elements addressed by lexicographic rank."""

    def __init__(self, n):
        self.n = int(n)
        self.perms = permutations(self.n)
        self.order = len(self.perms)

    def __repr__(self):
        return f"SymmetricGroup(n={self.n}, order={self.order})"

    @property
    def identity(self):
        """Id of the identity permutation (always 0 in lexicographic order)."""
        return 0

    def compose(self, left, right):
        """Ids of ``a . b`` for ``a = perms[left]``, ``b = perms[right]``, elementwise.

        Composition follows the original notebooks: ``c[k] = a[b[k]]``, i.e. ``b``
        is applied first.
        """
        left = np.asarray(left, dtype=np.int64).ravel()
        right = np.asarray(right, dtype=np.int64).ravel()
        if left.shape != right.shape:
            raise ValueError(f"left/right length mismatch: {left.shape} vs {right.shape}")

        out = np.empty(len(left), dtype=np.int64)
        for start in range(0, len(left), _CHUNK):
            block = slice(start, start + _CHUNK)
            composed = np.take_along_axis(self.perms[left[block]], self.perms[right[block]], axis=1)
            out[block] = rank(composed)
        return out

    def table(self):
        """The full ``(order, order)`` Cayley table, ``table[i, j] = i . j``."""
        pairs = self.order * self.order
        if pairs > MAX_FULL_PAIRS:
            raise ValueError(
                f"S_{self.n} has {self.order} elements -> {pairs} pairs, above the "
                f"{MAX_FULL_PAIRS} guard. Sample the product set instead "
                f"(max_pairs=... on the task builder)."
            )
        left = np.repeat(np.arange(self.order), self.order)
        right = np.tile(np.arange(self.order), self.order)
        return self.compose(left, right).reshape(self.order, self.order)


def minimal_faithful_dimension(n):
    """Dimension of the smallest faithful irreducible representation of ``S_n``.

    ``n - 1`` for every ``n >= 2`` -- the standard representation.  The two 1-D
    representations (trivial and sign) are unfaithful, and for ``S_4`` the 2-D
    representation factors through ``S_4 / V_4 = S_3`` so it is unfaithful too.

    This is the algebraic floor the paper predicts the post-grokking attractor
    dimension should plateau at: 4 for ``S_5``, 5 for ``S_6``.  See Sec. 4.2 of
    ``icomp_article/grokking_en.tex``.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 if n == 1 else n - 1
