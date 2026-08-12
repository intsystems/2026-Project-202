"""The registry of configurations this folder is meant to produce logs for.

Two families, and the contrast between them is the whole point:

``a_*``  the tasks Sec. 3 gives a periodic solution for -- these grok.
``x_*``  the tasks Sec. 4 / App. C report as not learnable -- these must *not* grok,
         and a run that quietly generalises here would falsify the reproduction
         rather than improve it.

Everything is full-batch GD with ``weight_decay = 0``: the paper's claim is that no
explicit regularisation is necessary (Sec. 2), and the intrinsic-dimension analysis
downstream wants a trajectory that is not being dragged by a decay term.

The learning rate is not stated in the paper.  ``LR`` below is what ``lr_sweep.py``
selected; see ``report.md`` for the sweep it came from.
"""

from __future__ import annotations

from gromov import Config
import tasks

LR_REF, P_REF, N_REF = 1.0e5, 97, 500
"""Full-batch GD learning rate at the reference point, from lr_sweep.py.

The paper gives none.  At p=97, N=500, alpha=0.5 the sweep put 1e4 and 3e4 below the
band (train accuracy 12% and 93% after 30 000 steps, never memorising) and 1e5 inside
it, memorising at step 9 180 and grokking at 12 000 -- which is Fig. 0's timescale
(train 100% at ~8 000-10 000, test 100% by ~20 000) reproduced without having been
fitted to it.
"""


def gd_lr(p, width):
    """Scale the rate to a different modulus or width.

    At initialisation the gradient of the mean-field parametrisation is
    ``dL/dW2 ~ -2 / (p^3 N)``: the ``1/(DN)`` prefactor of Eq. (4) and the ``1/p``
    variance of the pre-activations each contribute, and the batch size cancels.  The
    target weight scale ``A = (2D)^(1/3)`` moves only as ``p^(1/3)``, so holding the
    step *size* fixed means ``lr ~ p^3 N``.

    Without this a rate tuned at p=97 diverges immediately at p=23 -- the gradient
    there is 75x larger.  Checked against a local run: the formula maps p=97, N=500,
    lr=1e5 onto p=23, N=200, lr=533, and the measured band at that point was
    1e3-3e4, so the prediction is inside it to within a factor of two.
    """
    return LR_REF * (p / P_REF) ** 3 * (width / N_REF)


LR = LR_REF

STEPS = 100_000
"""Long enough that the dimension analysis has 10 000 logged samples at stride 10,
which is the sample count ``dimension_recovery`` froze its window size against."""


def _cfg(key, task, **kw):
    base = dict(key=key, task=task, description=tasks.DESCRIPTIONS[task],
                p=97, width=500, fraction=0.5, optimizer="gd", lr=LR,
                weight_decay=0.0, max_steps=STEPS, batch_size=None,
                log_every=10, obs_every=100, n_snapshots=21)
    base.update(kw)
    return Config(**base)


RUNS = {
    # --- grokkable ----------------------------------------------------------
    "a_add":     _cfg("a_add", "add"),
    "a_sub":     _cfg("a_sub", "sub"),
    "a_sq_sum":  _cfg("a_sq_sum", "sq_sum"),
    "a_sum_sq":  _cfg("a_sum_sq", "sum_sq"),
    "a_mul":     _cfg("a_mul", "mul"),
    # --- reported not to grok ----------------------------------------------
    "x_mix_quad": _cfg("x_mix_quad", "mix_quad"),
    "x_no_grok":  _cfg("x_no_grok", "no_grok"),
    # --- controls -----------------------------------------------------------
    # alpha below the critical fraction: the same task that groks at 0.5 must fail
    # here, which separates "the architecture cannot represent it" from "the data
    # were not enough".
    "c_add_lowalpha": _cfg("c_add_lowalpha", "add", fraction=0.2),
    # alpha high enough that Sec. 4 says mix_quad finally generalises.
    "c_mix_quad_hi":  _cfg("c_mix_quad_hi", "mix_quad", fraction=0.95),
    # second seed of the headline run, to tell run-to-run spread from a real effect.
    "a_add_s1":  _cfg("a_add_s1", "add", init_seed=2, split_seed=421),
    # BROKEN, and kept only as documentation, exactly like ``runs_poly.FAITHFUL``.
    # The intent was Fig. 5's AdamW setup, to give the dimension analysis a case where
    # the weight norm *is* being pulled. But wd=8.0 is a Doshi-parametrisation number:
    # here decoupled decay removes 8% of the norm per step against a task gradient of
    # order 2/(p^3 N), so the equilibrium norm is ~0.1 against the ~4.5 the task needs.
    # Measured at p=23, N=200: |W| 1.006 -> 0.00000 by step 500, accuracy at chance
    # for 3000 steps. Reproducing Fig. 5 needs the other parametrisation implemented,
    # not its hyperparameters copied. Deliberately absent from every group below.
    "r_add_adamw": _cfg("r_add_adamw", "add", optimizer="adamw",
                        lr=1e-2, weight_decay=8.0, max_steps=20_000),
}

GROUPS = {
    "grok": ("a_add", "a_sub", "a_sq_sum", "a_sum_sq", "a_mul"),
    "nogrok": ("x_mix_quad", "x_no_grok"),
    "controls": ("c_add_lowalpha", "c_mix_quad_hi", "a_add_s1"),
    "core": ("a_add", "a_mul", "x_no_grok", "x_mix_quad"),
}


def get(key):
    if key in RUNS:
        return RUNS[key]
    raise KeyError(f"unknown run '{key}'. Known: {sorted(RUNS)}")


def expand(names):
    """Resolve a list of run keys and/or group names to an ordered, deduped list."""
    out = []
    for name in names:
        for key in GROUPS.get(name, (name,)):
            if key not in out:
                out.append(key)
    return out
