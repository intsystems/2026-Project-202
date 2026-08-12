"""The modular polynomials of arXiv:2406.03495, App. C, Table 2.

The table is three polynomials, each in two versions. The *learnable* version has the
form of Hypothesis 5.1,

    P(n1, n2) = h( g1(n1) + g2(n2) )  mod p                            (Eq. 14)

which the quadratic perceptron can represent by the periodic construction of Eq. (2):
the first layer encodes ``g1(n1)`` and ``g2(n2)`` as phases, the quadratic activation
adds them, and the readout applies ``h``. The *perturbed* version adds one extra
monomial that cannot be absorbed into any ``h(g1 + g2)``, and the network then fits the
training set perfectly and fails to generalise.

That pair structure is what makes these runs worth collecting, and it is why appendix O
carries all six. The two members of a pair differ by a single low-order term, train
identically, and both reach 100 per cent training accuracy, so anything that separates
them in the logs is a property of the solution found rather than of the optimisation
difficulty or the data budget. ``actdim.analysis.representation`` settles the
representability half without a gradient step.

Test accuracy reported by the source paper (Adam, lr 5e-3, weight decay 5.0, N = 5000,
alpha 0.5):

    (4 n1 + n2^2)^3       mod 97  100%  |  + n1 n2   ->   2.27%
    (2 n1 + 3 n2)^4       mod 97  100%  |  - n1^2    ->   3.93%
    (5 n1^3 + 2 n2^4)^2   mod 97  100%  |  - n2      ->  72.32%
    (4 n1 + n2^2)^3       mod 23  100%  |  + n1 n2   ->   1.89%
    (2 n1 + 3 n2)^4       mod 23  100%  |  - n1^2    ->   7.17%
    (5 n1^3 + 2 n2^4)^2   mod 23  100%  |  - n2      ->   2.64%

The 72.32 per cent entry is the one the source paper does not comment on, and appendix
O reproduces it at 73 per cent rather than smoothing it.

Powers are taken with Python integers and reduced only at the end. ``int64`` would in
fact hold these -- the largest intermediate is ``(5*96^3 + 2*96^4)^2``, about 3.0e16
against ``int64``'s 9.2e18 -- but the margin is two orders of magnitude and depends on
both the modulus and the coefficients, so a larger ``p`` or a higher degree would
overflow silently and produce wrong labels rather than an error. Object dtype removes
the question.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np


def _obj(a: Any) -> np.ndarray:
    """Python-int arithmetic on an integer array: exact at any degree."""
    return np.asarray(a, dtype=object)


def _base1(n1: Any, n2: Any) -> np.ndarray:
    return (4 * _obj(n1) + _obj(n2) ** 2) ** 3


def _base2(n1: Any, n2: Any) -> np.ndarray:
    return (2 * _obj(n1) + 3 * _obj(n2)) ** 4


def _base3(n1: Any, n2: Any) -> np.ndarray:
    return (5 * _obj(n1) ** 3 + 2 * _obj(n2) ** 4) ** 2


POLYNOMIALS: Dict[str, Tuple[Callable[..., np.ndarray], bool]] = {
    # key           expression                                        learnable?
    "p1": (lambda n1, n2: _base1(n1, n2), True),
    "p1x": (lambda n1, n2: _base1(n1, n2) + _obj(n1) * _obj(n2), False),
    "p2": (lambda n1, n2: _base2(n1, n2), True),
    "p2x": (lambda n1, n2: _base2(n1, n2) - _obj(n1) ** 2, False),
    "p3": (lambda n1, n2: _base3(n1, n2), True),
    "p3x": (lambda n1, n2: _base3(n1, n2) - _obj(n2), False),
}

EXPRESSIONS: Dict[str, str] = {
    "p1": "(4 n1 + n2^2)^3",
    "p1x": "(4 n1 + n2^2)^3 + n1 n2",
    "p2": "(2 n1 + 3 n2)^4",
    "p2x": "(2 n1 + 3 n2)^4 - n1^2",
    "p3": "(5 n1^3 + 2 n2^4)^2",
    "p3x": "(5 n1^3 + 2 n2^4)^2 - n2",
}

LEARNABLE: Tuple[str, ...] = ("p1", "p2", "p3")
PERTURBED: Tuple[str, ...] = ("p1x", "p2x", "p3x")

PAPER_TEST_ACC: Dict[Tuple[int, str], float] = {
    (97, "p1"): 1.0000, (97, "p1x"): 0.0227,
    (97, "p2"): 1.0000, (97, "p2x"): 0.0393,
    (97, "p3"): 1.0000, (97, "p3x"): 0.7232,
    (23, "p1"): 1.0000, (23, "p1x"): 0.0189,
    (23, "p2"): 1.0000, (23, "p2x"): 0.0717,
    (23, "p3"): 1.0000, (23, "p3x"): 0.0264,
}
"""Table 2 verbatim, so a comparison reports the gap rather than an impression."""


def evaluator(name: str, p: int) -> Callable[..., np.ndarray]:
    """The label function, reduced mod ``p`` and returned as int64.

    Unlike the arithmetic tasks, this closes over ``p``: anything that changes the
    modulus of a run must rebuild the evaluator, or the labels stay at the old one.
    """
    if name not in POLYNOMIALS:
        raise KeyError(f"unknown polynomial {name!r}. Known: {sorted(POLYNOMIALS)}")
    fn, _ = POLYNOMIALS[name]
    return lambda n1, n2: np.asarray(fn(n1, n2) % p, dtype=np.int64)


def is_learnable(name: str) -> bool:
    return POLYNOMIALS[name][1]


def describe(name: str, p: int) -> str:
    kind = "learnable" if is_learnable(name) else "perturbed"
    return (f"{EXPRESSIONS[name]} mod {p} -- {kind} "
            f"(paper: {PAPER_TEST_ACC[(p, name)]:.2%} test accuracy)")


def table(name: str, p: int) -> np.ndarray:
    """The full ``p x p`` answer table."""
    n1, n2 = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return evaluator(name, p)(n1.reshape(-1), n2.reshape(-1)).reshape(p, p)


def distinct_outputs(name: str, p: int) -> Dict[str, float]:
    """How many residues the polynomial can produce, and the largest class share.

    A polynomial with a small image caps accuracy from below for reasons that have
    nothing to do with grokking: ``(2 n1 + 3 n2)^4 mod p`` only ever lands on fourth
    powers. Chance for a constant predictor is the majority share, not ``1/p``, and
    appendix O's ``g_p3x`` row is the one that needs the distinction.
    """
    y = table(name, p).reshape(-1)
    counts = np.bincount(y, minlength=p)
    return {"n_distinct": int((counts > 0).sum()),
            "majority_share": float(counts.max() / counts.sum())}
