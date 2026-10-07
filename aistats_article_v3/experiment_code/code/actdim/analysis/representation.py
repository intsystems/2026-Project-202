"""The representation of appendix M in closed form, and what is outside it.

Two questions about the quadratic perceptron that no training run answers, and neither
needs a gradient step.

**What does the solution look like when it exists?** For the tasks with a periodic
solution the weights can be written down (Eqs. 6-7 of arXiv:2301.02679) and scored on
the full ``p x p`` table. That settles the amplitude, and with it the normalisation
convention: the closed form has no free scale, so the network either emits a clean
one-hot delta or it does not. It is the one check a training run cannot make. A wrong
constant in the forward pass still produces a curve, at a different learning rate, and
nothing in the log says so.

It also supplies the reference values appendix M reads the measured order parameter
against. Three numbers can be read off the learned representation and they are not
commensurable: the *mode count* ``(p+1)/2``, which is the dimension the closed form
fixes; the *order parameter*, the mean inverse participation ratio of the first layer's
Fourier spectrum, which is 1.000 for a solution periodic in the raw operand and about
``1/p`` at random initialisation; and the *effective rank*, the participation ratio of
the singular values, which is 148.8 for the ``p = 97`` closed form and reads 139.1 at
random initialisation, within seven per cent of it before any training. Agreement with
the third is therefore evidence of nothing, and the second is only interpretable against
its own task's reference, which for two of appendix M's tasks sits near the floor.

**Is there a solution at all?** Hypothesis 5.1 of arXiv:2406.03495 says the network
generalises on ``P(n1, n2)`` exactly when ``P = h(g1(n1) + g2(n2)) mod p``. For the
three learnable polynomials the decomposition is known and exhibiting one settles
existence. For the three perturbed ones the claim is that *none* exists, which is a
statement about the ``p x p`` table alone and is decidable. ``decompose`` decides it by
two independent arguments -- a cheap one-sided multiset certificate and a complete
backtracking search -- and returns both, so the cheap one is cross-checked rather than
trusted. That proof is the substance of the article's claim that ``g_p3x`` reaches 73
per cent validation accuracy while being provably outside the class the architecture can
represent.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models.perceptron import (QuadraticPerceptron, effective_rank, fourier_ipr,
                                 weight_spectra)
from ..tasks import arithmetic, polynomials

# -- the closed-form weights ----------------------------------------------------

ARITHMETIC_CASES: Dict[str, Tuple[Optional[List[Callable]], Optional[Callable]]] = {
    #  task      inner (f1, f2)                          outer h
    "add": (None, None),
    "sub": ([lambda v: v, lambda v: -v], None),
    "sq_sum": ([lambda v: v ** 2, lambda v: v ** 2], None),
    "sum_sq": (None, lambda t: t ** 2),
}
"""The arithmetic tasks Sec. 3 gives a periodic solution for.

``mul``, ``mix_quad`` and ``no_grok`` are absent because no closed form is given for
them, and appendix M prints a dash rather than borrowing another task's reference.
"""

POLYNOMIAL_DECOMPOSITIONS: Dict[str, Tuple[Callable, Callable, Callable]] = {
    # Degrees stay under 5 and p under 100, so int64 cannot overflow here and the
    # object-dtype arithmetic the task table needs is unnecessary.
    "p1": (lambda n: 4 * n, lambda n: n ** 2, lambda t: t ** 3),
    "p2": (lambda n: 2 * n, lambda n: 3 * n, lambda t: t ** 4),
    "p3": (lambda n: 5 * n ** 3, lambda n: 2 * n ** 4, lambda t: t ** 2),
}
"""The known ``(g1, g2, h)`` of the three learnable polynomials, all arithmetic mod p."""


def mode_count(p: int) -> int:
    """Distinct Fourier frequencies the generalising solution uses: ``(p+1)/2``.

    49 at ``p = 97``. This is the only one of appendix M's three numbers that is a
    dimension in the sense the closed form fixes.
    """
    return (p + 1) // 2


def amplitude(p: int, n_vars: int = 2) -> float:
    """The one amplitude that makes the closed form a solution: ``A = (2 D)^(1/3)``.

    Substituting Eqs. (6)-(7) into Eq. (4) leaves ``A^3 / (2 D)`` on the diagonal, so
    the amplitude is fixed rather than free: 7.29 at ``p = 97``, which puts the weight
    norm about 5.2 times the ``N(0, 1)`` initialisation, the same order as the 3.7 a
    trained network reaches.
    """
    return (2.0 * n_vars * p) ** (1.0 / 3.0)


def build_weights(p: int, width: int, n_vars: int = 2,
                  inner: Optional[Sequence[Callable]] = None,
                  outer: Optional[Callable] = None,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Claim I and Claim II weights for ``h(f1(n) + f2(m)) mod p``.

        W1[k, n]     = A cos(2 pi k f1(n) / p + psi1_k)
        W1[k, p + m] = A cos(2 pi k f2(m) / p + psi2_k)
        W2[h(t), k] += A cos(-2 pi k t / p - psi1_k - psi2_k)

    ``inner`` is the pair ``(f1, f2)`` applied to the operands before the cosine and
    ``outer`` is ``h`` applied to the sum; both default to the identity, which gives
    plain modular addition. Frequencies run over ``0 .. (p-1)/2`` and are cycled when
    the width exceeds that range, with independent phases per neuron, which is what
    suppresses the spurious cross terms.

    The readout is written as a *forward* map -- row ``h(t)`` accumulates the frequency
    content of ``t`` -- rather than by inverting ``h`` as Eq. (19) does. That is what
    lets a non-invertible ``h`` work at all: every preimage of an output index
    contributes to it constructively, so ``(n + m)^2 mod p`` reaches 100 per cent here
    rather than the 51 per cent of Sec. 3.2, which comes from picking one branch of the
    square root.
    """
    rng = np.random.default_rng(seed)
    d_in = n_vars * p
    freqs = np.arange(0, p // 2 + 1)
    k = np.resize(freqs, width)
    psi = rng.uniform(-np.pi, np.pi, size=(n_vars, width))

    f = list(inner) if inner is not None else [lambda v: v] * n_vars
    h = outer or (lambda t: t)

    amp = amplitude(p, n_vars)
    w1 = np.zeros((width, d_in))
    for v in range(n_vars):
        vals = np.asarray(f[v](np.arange(p))) % p
        w1[:, v * p:(v + 1) * p] = amp * np.cos(
            2 * np.pi * np.outer(k, vals) / p + psi[v][:, None])

    t = np.arange(p)
    rows = np.asarray(h(t)) % p
    contrib = amp * np.cos(-2 * np.pi * np.outer(k, t) / p - psi.sum(axis=0)[:, None])
    w2 = np.zeros((p, width))
    np.add.at(w2, rows, contrib.T)
    return w1, w2


def has_closed_form(task: str) -> bool:
    return task in ARITHMETIC_CASES or task in POLYNOMIAL_DECOMPOSITIONS


def closed_form(task: str, p: int, width: int,
                seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """The closed-form weights for any task that has one, from either family."""
    if task in ARITHMETIC_CASES:
        inner, outer = ARITHMETIC_CASES[task]
        return build_weights(p, width, 2, inner, outer, seed=seed)
    if task in POLYNOMIAL_DECOMPOSITIONS:
        g1, g2, h = POLYNOMIAL_DECOMPOSITIONS[task]
        return build_weights(p, width, 2, [g1, g2], h, seed=seed)
    raise KeyError(f"no closed form is given for {task!r}. Known: "
                   f"{sorted(set(ARITHMETIC_CASES) | set(POLYNOMIAL_DECOMPOSITIONS))}")


def label_function(task: str, p: int) -> Callable[..., np.ndarray]:
    """The label function of a task, from whichever table defines it."""
    try:
        return arithmetic.get(task)
    except KeyError:
        return polynomials.evaluator(task, p)


def score_weights(w1: np.ndarray, w2: np.ndarray, p: int, task: str,
                  n_vars: int = 2) -> Dict[str, float]:
    """Score a weight pair on the whole ``p ** n_vars`` table.

    The whole table and not a split: the closed form is not fitted to anything, so
    holding examples out would measure nothing. ``mean_peak`` is the mean output at the
    correct class, which is the reading that fixes the amplitude and hence the
    normalisation convention -- it is 1.0 when the convention is right and off by the
    cube of the error when it is not.
    """
    import torch

    grids = np.meshgrid(*[np.arange(p) for _ in range(n_vars)], indexing="ij")
    operands = [g.reshape(-1) for g in grids]
    labels = np.asarray(label_function(task, p)(*operands), dtype=np.int64) % p
    m = labels.size
    x = np.zeros((m, n_vars * p), dtype=np.float64)
    for v, col in enumerate(operands):
        x[np.arange(m), v * p + col] = 1.0

    model = QuadraticPerceptron(p, w1.shape[0], n_vars, "quadratic",
                                dtype=torch.float64)
    with torch.no_grad():
        model.W1.copy_(torch.as_tensor(w1, dtype=torch.float64))
        model.W2.copy_(torch.as_tensor(w2, dtype=torch.float64))
        out = model(torch.as_tensor(x, dtype=torch.float64))
        y = torch.as_tensor(labels, dtype=torch.long)
        target = torch.zeros_like(out)
        target[torch.arange(y.shape[0]), y] = 1.0
        mse = float(((out - target) ** 2).mean())
        acc = float((out.argmax(1) == y).double().mean())
        peak = float(out[torch.arange(y.shape[0]), y].mean())
    norm = (math.sqrt((w1 ** 2).sum() + (w2 ** 2).sum())
            / math.sqrt(w1.size + w2.size))
    return {"acc": acc, "mse": mse, "mean_peak": peak, "weight_norm": norm}


def reference(task: str, p: int, width: int = 500, seed: int = 0,
              score: bool = False) -> Optional[Dict[str, Any]]:
    """Appendix M's own-reference numbers for one task, or ``None`` if it has none.

    ``order_parameter`` is the Fourier inverse participation ratio of the *first*
    operand block of the closed-form first layer, which is the column appendix M reads
    the measured value against. It is 1.000 when the representation is periodic in the
    raw operand and near the floor when it is not: ``n^2 + m^2`` and
    ``(5 n1^3 + 2 n2^4)^2`` encode a nonlinear function of the index, so their own
    reference is 0.052 and 0.062 at ``p = 97``. Against those a measured 0.044 indicates
    convergence, while against the 1.000 that modular addition sets the same value would
    say nothing had been learned.

    A task with no closed form returns ``None``. Nothing substitutes another task's
    reference for it.
    """
    if not has_closed_form(task):
        return None
    w1, w2 = closed_form(task, p, width, seed=seed)
    spectra = weight_spectra(w1, w2, p, 2)
    blocks = [fourier_ipr(w1[:, v * p:(v + 1) * p]) for v in range(2)]
    out: Dict[str, Any] = {
        "task": task, "p": p, "width": width,
        "mode_count": mode_count(p),
        "order_parameter": blocks[0],
        "order_parameter_blocks": blocks,
        "effective_rank": spectra["erank_w1"],
        "amplitude": amplitude(p, 2),
        "weight_norm": spectra["w1_norm"],
    }
    if score:
        out.update({f"analytic_{k}": v
                    for k, v in score_weights(w1, w2, p, task).items()})
    return out


def initialisation_reference(p: int, width: int = 500, n_vars: int = 2,
                             seed: int = 0) -> Dict[str, float]:
    """The same two spectra at random initialisation, which is the floor to read against.

    Appendix M quotes 0.041 for the order parameter and 139.1 for the effective rank at
    ``p = 97``, width 500. The second is within seven per cent of the closed form's
    148.8, which is why appendix M declines to treat agreement there as evidence.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((width, n_vars * p))
    return {"order_parameter": fourier_ipr(w1[:, :p]),
            "effective_rank": effective_rank(np.linalg.svd(w1, compute_uv=False))}


# -- deciding whether a decomposition exists ------------------------------------

@dataclass(frozen=True)
class Decomposition:
    """Whether ``P(n1, n2) = h(g1(n1) + g2(n2)) mod p``, and every argument that bears
    on it.

    ``exists`` is ``True`` when a decomposition was exhibited or found, ``False`` when
    non-existence was proved, and ``None`` when the search ran out of nodes and nothing
    was decided -- which is a third answer and not a negative.
    """

    name: str
    p: int
    exists: Optional[bool]
    shape: Tuple[int, int]
    reasons: Tuple[str, ...] = ()
    nodes: int = 0
    seconds: float = 0.0
    verified: bool = False
    certificate: Optional[str] = None
    searched: Optional[bool] = None

    @property
    def verdict(self) -> str:
        return {True: "exists", False: "none", None: "undecided"}[self.exists]

    def summary(self) -> Dict[str, Any]:
        """A flat record, for a table."""
        return {"name": self.name, "p": self.p, "verdict": self.verdict,
                "exists": self.exists, "rows": self.shape[0], "cols": self.shape[1],
                "verified": self.verified, "certificate": self.certificate,
                "searched": self.searched, "nodes": self.nodes,
                "seconds": round(self.seconds, 3),
                "reasons": "; ".join(self.reasons)}


def _reduce(t: np.ndarray) -> np.ndarray:
    """Collapse duplicate rows and duplicate columns.

    ``g1(n1) = g1(n1')`` forces rows ``n1`` and ``n1'`` to be equal, and conversely two
    equal rows can always be given the same ``g1`` value without changing any entry. So
    a decomposition of the deduplicated table with *injective* ``g1`` and ``g2`` exists
    exactly when one of the original table does, and injectivity is what gives the
    search something to prune with.
    """
    return np.unique(np.unique(t, axis=0).T, axis=0).T


def _multiset_certificate(m: np.ndarray, p: int) -> Optional[str]:
    """A cheap one-sided proof of non-existence: the reason, or ``None``.

    If the table has ``p`` distinct columns then ``g2`` is injective on them and hence a
    bijection of ``Z_p``, so row ``n1`` runs over ``{h(g1(n1) + b) : b in Z_p}``, the
    image of ``h`` over all of ``Z_p`` -- the same multiset of values for every row. Two
    rows with different value multisets therefore admit no decomposition at all. The
    transposed statement tests the columns. Nothing follows when the test passes, which
    is why the complete search is run as well.
    """
    def uniform(a: np.ndarray) -> bool:
        return len({tuple(np.bincount(r, minlength=p)) for r in a}) == 1

    rows, cols = m.shape
    if cols == p and not uniform(m):
        return "p distinct columns, rows have unequal value multisets"
    if rows == p and not uniform(m.T):
        return "p distinct rows, columns have unequal value multisets"
    return None


def _search(m: np.ndarray, p: int, budget: int) -> Tuple[Optional[bool], int]:
    """Complete backtracking search for ``(g1, g2, h)`` on the reduced table.

    The variables are the labels ``a[r] = g1(r)`` and ``b[c] = g2(c)``, both injective
    after ``_reduce``; ``h`` is a partial array that every assignment writes into and
    checks against, so the invariant ``m[r][c] == h(a[r] + b[c])`` holds over all
    assigned pairs at all times. Two normalisations cost nothing: translating ``g1`` and
    ``g2`` by constants absorbed into ``h`` fixes ``a[0] = b[0] = 0``, and the
    automorphism ``s -> u s`` of ``Z_p`` then rescales ``b[1]``, nonzero by injectivity,
    to 1. What remains is exhaustive, so returning ``False`` is a proof of
    non-existence.

    Returns ``(verdict, nodes)``, the verdict being ``None`` when the node budget ran
    out, in which case nothing was decided either way.
    """
    n_rows, n_cols = m.shape
    # Most distinct values first: those rows and columns constrain h the hardest.
    r_ord = np.argsort([-np.unique(m[i]).size for i in range(n_rows)], kind="stable")
    c_ord = np.argsort([-np.unique(m[:, j]).size for j in range(n_cols)], kind="stable")
    m = m[np.ix_(r_ord, c_ord)]

    h = np.full(p, -1, dtype=np.int64)
    alpha = np.full(n_rows, -1, dtype=np.int64)
    beta = np.full(n_cols, -1, dtype=np.int64)
    used_a = np.zeros(p, dtype=bool)
    used_b = np.zeros(p, dtype=bool)
    grid = np.arange(p)
    nodes = 0

    def assign(is_row: bool, i: int, v: int, trail: List[int]) -> bool:
        if is_row:
            alpha[i], used_a[v] = v, True
            pairs = [((v + beta[c]) % p, m[i, c]) for c in np.flatnonzero(beta >= 0)]
        else:
            beta[i], used_b[v] = v, True
            pairs = [((alpha[r] + v) % p, m[r, i]) for r in np.flatnonzero(alpha >= 0)]
        for s, want in pairs:
            if h[s] < 0:
                h[s] = want
                trail.append(s)
            elif h[s] != want:
                return False
        return True

    def retract(is_row: bool, i: int, v: int, trail: List[int]) -> None:
        for s in trail:
            h[s] = -1
        if is_row:
            alpha[i], used_a[v] = -1, False
        else:
            beta[i], used_b[v] = -1, False

    def domains():
        """Smallest live domain as ``(size, is_row, index, values)``; ``None`` if one is
        empty."""
        best = None
        done = True
        for is_row in (True, False):
            todo = np.flatnonzero((alpha if is_row else beta) < 0)
            if not todo.size:
                continue
            done = False
            fixed = np.flatnonzero((beta if is_row else alpha) >= 0)
            ok = np.repeat((~used_a if is_row else ~used_b)[None, :], todo.size, axis=0)
            if fixed.size:
                other = (beta if is_row else alpha)[fixed]
                seen = h[(grid[:, None] + other[None, :]) % p]
                want = m[np.ix_(todo, fixed)] if is_row else m[np.ix_(fixed, todo)].T
                ok &= ((seen < 0)[None] | (seen[None] == want[:, None, :])).all(2)
            sizes = ok.sum(1)
            j = int(sizes.argmin())
            if sizes[j] == 0:
                return None, False
            if best is None or sizes[j] < best[0]:
                best = (int(sizes[j]), is_row, int(todo[j]), np.flatnonzero(ok[j]))
        return best, done

    def descend() -> bool:
        """One node: propagate every forced label, then branch on the smallest domain."""
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise TimeoutError
        forced: List[Tuple[bool, int, int, List[int]]] = []
        try:
            while True:
                best, done = domains()
                if done:
                    return True
                if best is None:
                    return False
                if best[0] > 1:
                    break
                _, is_row, i, vals = best
                trail: List[int] = []
                forced.append((is_row, i, int(vals[0]), trail))
                if not assign(is_row, i, int(vals[0]), trail):
                    return False
            _, is_row, i, vals = best
            for v in vals:
                trail = []
                if assign(is_row, i, int(v), trail) and descend():
                    return True
                retract(is_row, i, int(v), trail)
            return False
        finally:
            # The verdict is the only output, so the state is always unwound.
            for is_row, i, v, trail in reversed(forced):
                retract(is_row, i, v, trail)

    root: List[int] = []
    assign(True, 0, 0, root)
    assign(False, 0, 0, root)
    if n_cols > 1 and not assign(False, 1, 1, root):
        return False, nodes
    try:
        return descend(), nodes
    except TimeoutError:
        return None, nodes


def verify_decomposition(name: str, p: int) -> bool:
    """Check the known ``(g1, g2, h)`` on every one of the ``p^2`` entries.

    Exhibiting one decomposition settles existence, so a pass here is a complete proof
    that the polynomial has the Hypothesis 5.1 form and nothing further is needed.
    """
    if name not in POLYNOMIAL_DECOMPOSITIONS:
        return False
    g1, g2, h = POLYNOMIAL_DECOMPOSITIONS[name]
    n1, n2 = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    s = (np.asarray(g1(n1)) + np.asarray(g2(n2))) % p
    return bool((np.asarray(h(s)) % p == polynomials.table(name, p)).all())


def decompose(name: str, p: int, budget: int = 40_000) -> Decomposition:
    """Decide ``P = h(g1 + g2)`` for one polynomial at one modulus.

    Three sound arguments, cheapest first: a verified decomposition proves existence,
    the multiset certificate proves non-existence, and the complete search proves
    either. All that apply are run, so the cheap ones are cross-checked rather than
    trusted, and every one of them is reported. ``budget = 0`` drops the search, which
    leaves the certificate as the only argument for non-existence.

    A disagreement between two of them would be a bug in one of them, so
    ``exists`` is taken from the strongest available argument and the reasons carry the
    rest for comparison.
    """
    table = polynomials.table(name, p)
    m = _reduce(table)
    started = time.time()
    exists: Optional[bool] = None
    reasons: List[str] = []
    verified = verify_decomposition(name, p)
    if verified:
        exists = True
        reasons.append("(g1, g2, h) verified on all p^2 entries")

    certificate = _multiset_certificate(m, p)
    if certificate is not None:
        if exists is True:
            raise AssertionError(
                f"{name} at p={p}: a decomposition was verified on the full table and "
                f"the multiset certificate claims none exists ({certificate}). One of "
                f"the two is wrong.")
        exists = False
        reasons.append(f"multiset certificate: {certificate}")

    searched: Optional[bool] = None
    nodes = 0
    if budget:
        searched, nodes = _search(m, p, budget)
        reasons.append({True: "complete search: exists",
                        False: "complete search: none",
                        None: f"complete search undecided at {budget} nodes"}[searched])
        if searched is not None and exists is not None and searched != exists:
            raise AssertionError(
                f"{name} at p={p}: the complete search and the cheaper argument "
                f"disagree ({searched} against {exists}). One of them is wrong.")
        if exists is None:
            exists = searched

    return Decomposition(name=name, p=p, exists=exists, shape=tuple(m.shape),
                         reasons=tuple(reasons), nodes=nodes,
                         seconds=time.time() - started, verified=verified,
                         certificate=certificate, searched=searched)


def decompose_all(p: int, budget: int = 40_000,
                  names: Optional[Sequence[str]] = None) -> Dict[str, Decomposition]:
    """Decide every polynomial of appendix O at one modulus, in table order."""
    names = names or tuple(polynomials.POLYNOMIALS)
    return {name: decompose(name, p, budget) for name in names}
