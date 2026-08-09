"""The modular polynomials of arXiv:2406.03495, Appendix C, Table 2.

The table is six polynomials, each in two versions.  The *learnable* version has the
form of Hypothesis 5.1,

    P(n1, n2) = h( g1(n1) + g2(n2) )  mod p                       (Eq. 14)

which the two-layer MLP can represent by the periodic construction of Eq. (2): the
first layer encodes ``g1(n1)`` and ``g2(n2)`` as phases, the quadratic activation
adds them, and the readout applies ``h``.  The *perturbed* version adds one extra
monomial that cannot be absorbed into any ``h(g1 + g2)``, and the paper reports the
network then fails to generalise while still fitting the training set perfectly.

That pair structure is what makes these runs worth collecting.  The two members of a
pair differ by a single low-order term, are trained identically, and both reach 100%
training accuracy -- so anything that distinguishes them in the logs is a property of
the *solution found*, not of the optimisation difficulty or the data budget.

Paper's reported test accuracy (Adam, lr 5e-3, weight decay 5.0, N = 5000, alpha 0.5):

    (4 n1 + n2^2)^3            mod 97  100%   |  + n1 n2   ->   2.27%
    (2 n1 + 3 n2)^4            mod 97  100%   |  - n1^2    ->   3.93%
    (5 n1^3 + 2 n2^4)^2        mod 97  100%   |  - n2      ->  72.32%
    (4 n1 + n2^2)^3            mod 23  100%   |  + n1 n2   ->   1.89%
    (2 n1 + 3 n2)^4            mod 23  100%   |  - n1^2    ->   7.17%
    (5 n1^3 + 2 n2^4)^2        mod 23  100%   |  - n2      ->   2.64%

Chance is 1/p: 1.03% at p = 97, 4.35% at p = 23.  The 72.32% entry is the one
outlier the paper does not comment on; it is reproduced here rather than smoothed.

Powers are taken with Python integers and reduced only at the end.  ``int64`` would in
fact hold these -- the largest intermediate is ``(5*96^3 + 2*96^4)^2 ~ 3.0e16`` against
``int64``'s 9.2e18 -- but the margin is only two orders of magnitude and it depends on
both the modulus and the coefficients, so a larger ``p`` or a higher degree would
overflow silently and produce wrong labels rather than an error.  Object dtype removes
the question.
"""

from __future__ import annotations

import numpy as np


def _obj(a):
    """Python-int arithmetic on an integer array: exact for any degree."""
    return np.asarray(a, dtype=object)


# --- the three base polynomials, in the paper's notation ---------------------

def _base1(n1, n2):
    return (4 * _obj(n1) + _obj(n2) ** 2) ** 3


def _base2(n1, n2):
    return (2 * _obj(n1) + 3 * _obj(n2)) ** 4


def _base3(n1, n2):
    return (5 * _obj(n1) ** 3 + 2 * _obj(n2) ** 4) ** 2


POLYNOMIALS = {
    # key            expression                                     learnable?
    "p1":  (lambda n1, n2: _base1(n1, n2),                          True),
    "p1x": (lambda n1, n2: _base1(n1, n2) + _obj(n1) * _obj(n2),    False),
    "p2":  (lambda n1, n2: _base2(n1, n2),                          True),
    "p2x": (lambda n1, n2: _base2(n1, n2) - _obj(n1) ** 2,          False),
    "p3":  (lambda n1, n2: _base3(n1, n2),                          True),
    "p3x": (lambda n1, n2: _base3(n1, n2) - _obj(n2),               False),
}

EXPRESSIONS = {
    "p1":  "(4 n1 + n2^2)^3",
    "p1x": "(4 n1 + n2^2)^3 + n1 n2",
    "p2":  "(2 n1 + 3 n2)^4",
    "p2x": "(2 n1 + 3 n2)^4 - n1^2",
    "p3":  "(5 n1^3 + 2 n2^4)^2",
    "p3x": "(5 n1^3 + 2 n2^4)^2 - n2",
}

LEARNABLE = ("p1", "p2", "p3")
PERTURBED = ("p1x", "p2x", "p3x")

PAPER_TEST_ACC = {
    (97, "p1"): 1.0000, (97, "p1x"): 0.0227,
    (97, "p2"): 1.0000, (97, "p2x"): 0.0393,
    (97, "p3"): 1.0000, (97, "p3x"): 0.7232,
    (23, "p1"): 1.0000, (23, "p1x"): 0.0189,
    (23, "p2"): 1.0000, (23, "p2x"): 0.0717,
    (23, "p3"): 1.0000, (23, "p3x"): 0.0264,
}
"""Table 2 verbatim, so ``compare.py`` can report the gap instead of a vibe."""


def evaluator(name, p):
    """The label function, reduced mod ``p`` and returned as int64."""
    if name not in POLYNOMIALS:
        raise KeyError(f"unknown polynomial '{name}'. Known: {sorted(POLYNOMIALS)}")
    fn, _ = POLYNOMIALS[name]
    return lambda n1, n2: np.asarray(fn(n1, n2) % p, dtype=np.int64)


def is_learnable(name):
    return POLYNOMIALS[name][1]


def distinct_outputs(name, p):
    """How many residues the polynomial can produce.

    A polynomial whose image is small caps accuracy from below in a way that has
    nothing to do with grokking -- ``(2 n1 + 3 n2)^4 mod p`` only ever lands on
    fourth powers.  Chance for a constant predictor is the largest class share, not
    ``1/p``, and ``compare.py`` uses this to say which is which.
    """
    n1, n2 = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    y = evaluator(name, p)(n1.reshape(-1), n2.reshape(-1))
    counts = np.bincount(y, minlength=p)
    return dict(n_distinct=int((counts > 0).sum()),
                majority_share=float(counts.max() / counts.sum()))
