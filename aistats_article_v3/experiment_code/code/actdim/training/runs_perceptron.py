"""The perceptron rows of appendix O, and what else the archived campaign defined.

Thirteen rows of the article's run inventory are perceptron runs, and they are the
thirteen keys of ``INVENTORY``. All of them are full-batch gradient descent at
``p = 97``, width 500, half the pairs held out, no regularisation of any kind, at the
one learning rate ``gd_lr`` gives for that point. Two families, and the contrast between
them is the point of the set:

``a_*``  the arithmetic tasks Sec. 3 of arXiv:2301.02679 gives a periodic solution for,
         which grok.
``x_*``  the tasks Sec. 4 and App. C report as not learnable, which must *not* grok. A
         run that quietly generalised here would falsify the reproduction rather than
         improve it.
``g_*``  the polynomial pairs of arXiv:2406.03495, App. C: three that have the
         representable form and three perturbed by one monomial that does not. The two
         members of a pair share every hyperparameter and both reach 100 per cent
         training accuracy, so anything that separates them is a property of the
         solution found. ``actdim.analysis.representation`` proves the perturbed three
         are outside the representable class without training anything.

Three things this file records that the article's own inventory gets wrong or leaves
out. They are marked at the entries they concern: ``a_sum_sq`` ran 46,000 steps and not
the 100,000 its budget column claims; the runs that carry the trajectory sketch are the
four of ``SKETCHED`` and not the table's first four rows; and two run families the
archived registry documented as broken are carried in ``BROKEN``, out of every group, so
that nobody spends a GPU hour rediscovering why they do not work.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..tasks import arithmetic, polynomials
from .perceptron import PerceptronConfig

# -- the learning rate the source paper never states ----------------------------

LR_REF, P_REF, N_REF = 1.0e5, 97, 500
"""Full-batch rate at the reference point, from this study's own sweep.

At ``p = 97``, ``N = 500``, ``alpha = 0.5`` the sweep put 1e4 and 3e4 below the band --
training accuracy 12 and 93 per cent after 30,000 steps, never memorising -- and 1e5
inside it, memorising at step 9,180 and generalising at 12,000. That is the source
paper's own timescale reproduced without having been fitted to it.
"""


def gd_lr(p: int, width: int) -> float:
    """Scale the rate to another modulus or width.

    At initialisation the gradient of the mean-field parametrisation is
    ``dL/dW2 ~ -2 / (p^3 N)``: the ``1/(D N)`` prefactor and the ``1/p`` variance of the
    pre-activations each contribute, and the batch size cancels. The target weight scale
    ``A = (2 D)^(1/3)`` moves only as ``p^(1/3)``, so holding the step size fixed means
    ``lr ~ p^3 N``.

    Without this a rate tuned at ``p = 97`` diverges on the first step at ``p = 23``,
    where the gradient is 75 times larger.
    """
    return LR_REF * (p / P_REF) ** 3 * (width / N_REF)


STEPS = 100_000
"""The budget of appendix O's perceptron rows.

Long enough that the trajectory analysis has 10,000 logged samples at a stride of ten,
which is the sample count the estimator's window size was frozen against.
"""

ALPHA: Dict[int, float] = {97: 0.5, 23: 0.8}
"""Training fraction by modulus.

0.5 at ``p = 97`` is the source paper's value and sits well above the critical fraction
of about 0.29. At ``p = 23`` the critical fraction is larger -- a direct check put it
between 0.5 and 0.7 -- so a run there at 0.5 would fail for a reason that has nothing to
do with the polynomial, and would make the learnable and perturbed arms
indistinguishable for the wrong reason.
"""


def _arith(key: str, task: str, **kw: Any) -> PerceptronConfig:
    base: Dict[str, Any] = dict(
        key=key, task=task, description=arithmetic.describe(task), p=97, width=500,
        fraction=ALPHA[97], optimizer="gd", lr=gd_lr(97, 500), weight_decay=0.0,
        max_steps=STEPS, batch_size=None, log_every=10, obs_every=100, n_snapshots=21)
    base.update(kw)
    return PerceptronConfig(**base)


def _poly(key: str, name: str, p: int, **kw: Any) -> PerceptronConfig:
    base: Dict[str, Any] = dict(
        key=key, task=name, p=p, n_vars=2, description=polynomials.describe(name, p),
        activation="quadratic", width=500, fraction=ALPHA[p], optimizer="gd",
        lr=gd_lr(p, 500), weight_decay=0.0, max_steps=STEPS, batch_size=None,
        log_every=10, obs_every=100, n_snapshots=21)
    base.update(kw)
    return PerceptronConfig(**base)


# -- the runs -------------------------------------------------------------------

RUNS: Dict[str, PerceptronConfig] = {
    # -- appendix O, the five arithmetic rows that generalise ------------------
    "a_add": _arith("a_add", "add"),
    "a_mul": _arith("a_mul", "mul"),
    "a_sub": _arith("a_sub", "sub"),
    "a_sq_sum": _arith("a_sq_sum", "sq_sum"),
    # 46,000 steps, not the 100,000 appendix O's budget column states. The run was
    # stopped early and its log ends at step 46,000; it had generalised at 8,150, so no
    # claim in the article moves, but the inventory is wrong here and the budget
    # recorded is the one that ran. Anything that reproduces the row must use this
    # number or it will not reproduce the log.
    "a_sum_sq": _arith("a_sum_sq", "sum_sq", max_steps=46_000),
    # -- appendix O, the two arithmetic rows that must not generalise ----------
    "x_mix_quad": _arith("x_mix_quad", "mix_quad"),
    "x_no_grok": _arith("x_no_grok", "no_grok"),
    # -- controls, defined by the archived campaign and never completed --------
    # Below the critical fraction, so the task that groks at 0.5 must fail here, which
    # separates "the architecture cannot represent it" from "the data were not enough".
    "c_add_lowalpha": _arith("c_add_lowalpha", "add", fraction=0.2),
    # Above the fraction at which Sec. 4 says mix_quad finally generalises.
    "c_mix_quad_hi": _arith("c_mix_quad_hi", "mix_quad", fraction=0.95),
    # A second seed of the headline run, to tell run-to-run spread from a real effect.
    "a_add_s1": _arith("a_add_s1", "add", init_seed=2, split_seed=421),
}

# -- appendix O, the six polynomial rows, and the second modulus ----------------
# The article's inventory names the p = 97 rows without a suffix, so they keep those
# names here. The p = 23 arm is real and complete but is cited nowhere in the article,
# and carries its modulus in its key.
for _name in polynomials.POLYNOMIALS:
    RUNS[f"g_{_name}"] = _poly(f"g_{_name}", _name, 97)
    RUNS[f"g_{_name}_p23"] = _poly(f"g_{_name}_p23", _name, 23)


INVENTORY: Tuple[str, ...] = (
    "a_add", "a_mul", "a_sub", "a_sq_sum", "a_sum_sq",
    "x_mix_quad", "x_no_grok",
    "g_p1", "g_p1x", "g_p2", "g_p2x", "g_p3", "g_p3x",
)
"""The thirteen perceptron rows of appendix O, in the order the table prints them."""

PAPER_MILESTONES: Dict[str, Dict[str, Optional[float]]] = {
    "a_add": {"memorise": 9_230, "generalise": 11_760, "final_val_acc": 1.00},
    "a_mul": {"memorise": 9_100, "generalise": 11_560, "final_val_acc": 1.00},
    "a_sub": {"memorise": 9_110, "generalise": 11_540, "final_val_acc": 1.00},
    "a_sq_sum": {"memorise": 8_440, "generalise": 10_470, "final_val_acc": 1.00},
    "a_sum_sq": {"memorise": 7_350, "generalise": 8_190, "final_val_acc": 1.00},
    "x_mix_quad": {"memorise": 9_600, "generalise": None, "final_val_acc": 0.01},
    "x_no_grok": {"memorise": 9_860, "generalise": None, "final_val_acc": 0.01},
    "g_p1": {"memorise": 7_470, "generalise": 12_290, "final_val_acc": 1.00},
    "g_p1x": {"memorise": 10_140, "generalise": None, "final_val_acc": 0.01},
    "g_p2": {"memorise": 6_470, "generalise": 7_680, "final_val_acc": 1.00},
    "g_p2x": {"memorise": 9_730, "generalise": None, "final_val_acc": 0.02},
    "g_p3": {"memorise": 5_970, "generalise": 6_600, "final_val_acc": 1.00},
    "g_p3x": {"memorise": 9_420, "generalise": None, "final_val_acc": 0.74},
}
"""What appendix O reports for each row, so a re-run can be compared against the table.

A generalisation step of ``None`` is the table's dash: the criterion was not met inside
the budget. Such a row is a censored observation and not an established negative.

These were the article's numbers transcribed, and for one campaign they were the only
source: no committed file recorded what a perceptron run reached, so the table was checked
against a copy of itself. Both training experiments promote a ``milestones.csv`` now, and
`actdim.tables.check_runs` compares the rows against those and against this copy, so the
transcription going stale is reported rather than believed. The values above are the
committed ones, which differ from the published campaign by ten to two hundred and seventy
steps.
"""

SKETCHED: Tuple[str, ...] = ("a_add", "x_no_grok", "g_p1", "g_p1x")
"""The four runs repeated with the trajectory sketch attached.

The article's text says the top four rows of the table, which would be ``a_add``,
``a_mul``, ``a_sub`` and ``a_sq_sum``. The campaign that ran carries these four instead:
one generalising and one non-generalising arithmetic run, and one matched polynomial
pair. That is the better set for the comparison, since it holds the label function as
the only difference within a pair, but it is not what the text describes.

Each of the four was run twice, once at full batch and once with mini-batches of 512;
the batch rule is the only difference between the two campaigns, which is why the run
record carries it as a string that cannot be null.
"""

# -- the run families that do not work ------------------------------------------

_ADAMW_REASON = (
    "AdamW at weight_decay = 8.0. The number belongs to the Doshi parametrisation, "
    "where the normalisation is folded into the initialisation. Under the "
    "parametrisation this package implements, decoupled decay removes 8 per cent of "
    "the norm per step against a task gradient of order 2/(p^3 N), so the equilibrium "
    "norm is about 0.1 against the 4.5 the task needs. Measured at p = 23, N = 200: "
    "the norm fell from 1.006 to zero by step 500 and accuracy stayed at chance for "
    "3,000 steps. Reproducing that figure needs the other parametrisation written, not "
    "its hyperparameters copied.")

_FAITHFUL_REASON = (
    "The source paper's App. C setting verbatim: Adam, lr 5e-3, weight decay 5.0, "
    "N = 5000. Broken for the same reason as r_add_adamw. The L2 term Adam adds is "
    "wd * w, about 5, against a task gradient of about 4e-10, so the weights are driven "
    "to zero before the task is seen: the norm measured 0.000 by step 5,000 with the "
    "loss pinned at 1/p. Two of these were run and their dead logs were kept; the "
    "article cites neither.")

FAITHFUL: Dict[str, Any] = dict(optimizer="adam", lr=5e-3, weight_decay=5.0, width=5000,
                                fraction=0.5, max_steps=8_000, log_every=5,
                                obs_every=200, n_snapshots=15)

BROKEN: Dict[str, str] = {
    "r_add_adamw": _ADAMW_REASON,
}

RUNS["r_add_adamw"] = _arith("r_add_adamw", "add", optimizer="adamw", lr=1e-2,
                             weight_decay=8.0, max_steps=20_000)

for _p in (97, 23):
    for _name in polynomials.POLYNOMIALS:
        _key = f"f_{_name}" if _p == 97 else f"f_{_name}_p{_p}"
        RUNS[_key] = _poly(_key, _name, _p, **FAITHFUL)
        BROKEN[_key] = _FAITHFUL_REASON


# -- groups ---------------------------------------------------------------------

GROUPS: Dict[str, Tuple[str, ...]] = {
    "inventory": INVENTORY,
    "grok": ("a_add", "a_mul", "a_sub", "a_sq_sum", "a_sum_sq"),
    "nogrok": ("x_mix_quad", "x_no_grok"),
    "arith": ("a_add", "a_mul", "a_sub", "a_sq_sum", "a_sum_sq",
              "x_mix_quad", "x_no_grok"),
    "poly": ("g_p1", "g_p1x", "g_p2", "g_p2x", "g_p3", "g_p3x"),
    "poly23": tuple(f"g_{n}_p23" for n in polynomials.POLYNOMIALS),
    # One learnable and one perturbed run per base polynomial, plus the arithmetic
    # pair: the matched comparisons, which are what the sketch campaign measures.
    "pairs": ("a_add", "x_no_grok", "g_p1", "g_p1x", "g_p2", "g_p2x", "g_p3", "g_p3x"),
    "sketched": SKETCHED,
    "controls": ("c_add_lowalpha", "c_mix_quad_hi", "a_add_s1"),
}

# A broken run must never arrive through a group. Checked here rather than trusted,
# because the archived registry kept its broken entries in the same dictionary as the
# working ones and nothing but a comment stopped a group from naming one.
for _group, _keys in GROUPS.items():
    _bad = [k for k in _keys if k in BROKEN]
    if _bad:
        raise ValueError(f"group {_group!r} names broken runs: {_bad}")
    _missing = [k for k in _keys if k not in RUNS]
    if _missing:
        raise ValueError(f"group {_group!r} names unknown runs: {_missing}")


def get(key: str) -> PerceptronConfig:
    """The config for one key. Broken runs resolve, so they can be inspected."""
    if key not in RUNS:
        raise KeyError(f"unknown run {key!r}. Known: {sorted(RUNS)}")
    return RUNS[key]


def why_broken(key: str) -> Optional[str]:
    """Why this run does not work, or ``None`` if it does."""
    return BROKEN.get(key)


def expand(names: Iterable[str], allow_broken: bool = False) -> List[str]:
    """Resolve keys and group names to an ordered list with no repeats.

    A broken run is refused with its reason rather than run. Passing
    ``allow_broken=True`` runs one anyway, which is what reproducing the failure needs.
    """
    out: List[str] = []
    for name in names:
        for key in GROUPS.get(name, (name,)):
            if key not in RUNS:
                raise KeyError(f"unknown run or group {name!r}. "
                               f"Groups: {sorted(GROUPS)}")
            if key in BROKEN and not allow_broken:
                raise KeyError(f"{key} is a known-broken configuration and is in no "
                               f"group. {BROKEN[key]} Pass allow_broken=True to run it "
                               f"anyway.")
            if key not in out:
                out.append(key)
    return out
