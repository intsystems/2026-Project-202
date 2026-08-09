"""Registry for the modular-polynomial runs.

Two arms, because the paper and the request want different things and both are cheap:

``f_*``  **paper-faithful** -- Adam, lr 5e-3, weight decay 5.0, N = 5000, alpha 0.5,
         which is what App. C states.  Its job is to show that the grokkable /
         non-grokkable split of Table 2 reproduces at all.
``g_*``  **Gromov no-weight-decay** -- full-batch GD, wd = 0, N = 500, the setup of
         arXiv:2301.02679 Sec. 2.  This is the arm the dimension analysis wants: with
         wd = 5.0 the weight norm is being dragged by the regulariser, and any
         trajectory statistic read off it is partly a statistic of the regulariser.

The training fraction differs between the two primes on purpose.  Sec. 4 of the
arithmetic paper says the critical fraction ``alpha_c`` grows as ``p`` shrinks, and a
direct check here (add mod 23, N = 200, GD, 60 000 steps) put ``alpha_c`` for p = 23
between 0.5 and 0.7: at 0.5 test accuracy never left chance, at 0.7 it grokked at
step 750.  Running p = 23 at alpha = 0.5 under plain GD would therefore produce a
non-grokking log for a reason that has nothing to do with the polynomial, and would
make the learnable and perturbed arms indistinguishable for the wrong reason.
"""

from __future__ import annotations

from _core import Config
from runs import gd_lr
import polynomials as P

FAITHFUL = dict(optimizer="adam", lr=5e-3, weight_decay=5.0, width=5000,
                fraction=0.5, max_steps=8_000, log_every=5, obs_every=200,
                n_snapshots=15)
"""App. C verbatim, except the step budget and the batch rule, which it does not state.

BROKEN AS WRITTEN, and kept only to document why.  These numbers belong to the Doshi
parametrisation, where the scale is folded into the init and the output is O(1) at step
zero.  Dropped onto the arithmetic paper's parametrisation -- ``N(0,1)`` init, ``1/(DN)``
in the forward pass -- the L2 term Adam adds is ``wd * w ~ 5`` against a task gradient of
``2/(p^3 N) ~ 4e-10``, so the weights are driven to zero before the task is seen: measured
``|W| = 0.000`` by step 5 000, loss pinned at ``1/p``.  Reproducing this arm needs the
other parametrisation implemented, not its hyperparameters copied.  See
``report.md`` Result 4."""

NOWD = dict(optimizer="gd", weight_decay=0.0, width=500,
            max_steps=100_000, log_every=10, obs_every=100, n_snapshots=21)
"""Gromov's no-regularisation setup.  The rate comes from ``gd_lr(p, width)`` rather
than a constant: it has to fall by 75x between p=97 and p=23 or the smaller modulus
diverges on the first step."""

ALPHA = {97: 0.5, 23: 0.8}
"""alpha = 0.5 at p = 97 is the paper's value and sits well above alpha_c ~ 0.29.
alpha = 0.8 at p = 23 is above the 0.5-0.7 bracket measured for that modulus."""


def _cfg(prefix, name, p, **kw):
    base = dict(key=f"{prefix}_{name}_p{p}", task=name, p=p, n_vars=2,
                description=f"{P.EXPRESSIONS[name]} mod {p} -- "
                            f"{'learnable' if P.is_learnable(name) else 'perturbed'}"
                            f" (paper: {P.PAPER_TEST_ACC[(p, name)]:.2%} test acc)",
                activation="quadratic", batch_size=None)
    base.update(kw)
    return Config(**base)


RUNS = {}
for _p in (97, 23):
    for _name in P.POLYNOMIALS:
        RUNS[f"f_{_name}_p{_p}"] = _cfg("f", _name, _p, **FAITHFUL)
        RUNS[f"g_{_name}_p{_p}"] = _cfg("g", _name, _p, fraction=ALPHA[_p],
                                        lr=gd_lr(_p, NOWD["width"]), **NOWD)

GROUPS = {
    "faithful97": tuple(f"f_{n}_p97" for n in P.POLYNOMIALS),
    "faithful23": tuple(f"f_{n}_p23" for n in P.POLYNOMIALS),
    "faithful": tuple(f"f_{n}_p{p}" for p in (97, 23) for n in P.POLYNOMIALS),
    "nowd97": tuple(f"g_{n}_p97" for n in P.POLYNOMIALS),
    "nowd23": tuple(f"g_{n}_p23" for n in P.POLYNOMIALS),
    "nowd": tuple(f"g_{n}_p{p}" for p in (97, 23) for n in P.POLYNOMIALS),
    # one learnable / perturbed pair per base polynomial: the minimum that still
    # carries the contrast, for when GPU time is short.
    "pairs97": ("g_p1_p97", "g_p1x_p97", "g_p2_p97", "g_p2x_p97",
                "g_p3_p97", "g_p3x_p97"),
}


def get(key):
    if key in RUNS:
        return RUNS[key]
    raise KeyError(f"unknown run '{key}'. Known: {sorted(RUNS)}")


def expand(names):
    out = []
    for name in names:
        for key in GROUPS.get(name, (name,)):
            if key not in out:
                out.append(key)
    return out
