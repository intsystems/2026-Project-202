"""The twelve scalar observers of appendix B, defined once.

An observer is a scalar a training run either already logs or could log without changing
anything: a loss, a norm, a gradient statistic, a fixed projection, a function-space
quantity. The article records twelve of them on the image-data system, in five families,
and every claim about "which observer works" is a claim about this list.

The archived tree kept the list in five places. ``dynamics.OBSERVERS`` had sixteen,
computed inline in the simulation loop; ``dynamics.OBSERVER_FAMILY`` had the families;
``e2_rank_sweep.SWEEP_OBS`` had the twelve the article reports, and ``e3_transitions`` and
``e4_controls`` imported their panel *from that experiment module*, so an experiment was
acting as a configuration file; ``calibration_k20`` and ``e10_ceiling_sweep`` each had a
different panel of eight and four. The panels below are selections from the one registry,
so a name can be added or dropped in one place and every experiment follows.

The four names the article drops -- ``w_fro_sq``, ``c_proj2``, ``c_proj3``, ``fn_proj2`` --
are near-duplicates of one that is kept, and each costs a full pass of the estimator. They
are not registered here because no published panel contains them. One archived comment
justified dropping ``w_fro`` on the grounds that it returns "bit-identical" estimates to
``c_norm``; it does not. ``w_fro`` is a monotone *nonlinear* function of ``c_norm``, the
two differ by up to 9.3e-5 over the published ranks, and the article keeps both because
appendix B lists both.

Nothing here writes a file, and no observer holds state: each is a function of one step's
:class:`Observation`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .linalg import orthonormal, unit
from .runtime.determinism import rng as stream_rng

#: The five families of appendix B, in the order its table lists them.
FAMILIES: Tuple[str, ...] = ("loss", "norm", "gradient", "projection", "function")


# ----------------------------------------------------------------- what they read

@dataclass(frozen=True)
class Probe:
    """The fixed held-out set the function-space observers are read on.

    Drawn once before training and never redrawn, so that a change in a function-space
    observer is a change in the function and not a change in where it was measured.
    """

    labels: np.ndarray
    classes: int = 10

    @property
    def index(self) -> np.ndarray:
        return np.arange(len(self.labels))

    @property
    def true_mask(self) -> np.ndarray:
        return np.eye(self.classes)[self.labels] > 0


@dataclass(frozen=True)
class Directions:
    """The projection directions and the rotation, drawn once from a fixed stream.

    ``gradient`` and ``parameter`` live in the ``k``-dimensional subspace, ``function`` in
    the flattened probe-logit space. ``rotation`` is the fixed orthogonal change of
    coordinates the invariance control applies; it is the identity otherwise, and it is
    what makes ``c_proj1`` the one observer that is *not* invariant to a rotation of the
    parameters -- which is the point of recording it.
    """

    gradient: np.ndarray
    parameter: np.ndarray
    function: np.ndarray
    rotation: np.ndarray


def directions(k: int, function_dim: int, seed: int, rotate: bool = False) -> Directions:
    """Draw the fixed directions for one run.

    The draws are made in the archived order and at the archived shapes -- a ``(k, 3)``
    frame of which the first column is used, two function-space rows of which the first is
    used -- so that a re-run reproduces the published projections. Only the first of each
    is read: the other two projections were near-duplicates and the article drops them.
    """
    generator = stream_rng(seed, "observer_directions")
    gradient = unit(generator.standard_normal(k))
    parameter = orthonormal((k, 3), generator)[:, 0]
    function = generator.standard_normal((2, function_dim))
    function /= np.linalg.norm(function, axis=1, keepdims=True)
    rotation = (orthonormal((k, k), stream_rng(seed, "rotation")) if rotate
                else np.eye(k))
    return Directions(gradient=gradient, parameter=parameter, function=function[0],
                      rotation=rotation)


@dataclass
class Observation:
    """Everything the twelve observers are functions of, at one step.

    Built once per step by the system and handed to whichever panel is being recorded, so
    that the expensive quantities -- the full-batch gradient, the probe logits -- are
    computed once however many observers read them.
    """

    step_loss: float          # the loss on the minibatch the step was taken on
    full_loss: float          # the loss on the whole training set
    probe_loss: float         # cross-entropy on the probe set
    coordinate: np.ndarray    # c, the coordinate in the k-dimensional subspace
    gradient: np.ndarray      # g = d loss / d c, full batch
    logits: np.ndarray        # L, the probe logits, (m, classes)
    head_norm: float          # ||theta_0||, the frozen head at initialisation
    probe: Probe
    directions: Directions


# ----------------------------------------------------------------- the definitions

def _loss_step(o: Observation) -> float:
    return float(o.step_loss)


def _loss_full(o: Observation) -> float:
    return float(o.full_loss)


def _loss_probe(o: Observation) -> float:
    return float(o.probe_loss)


def _w_fro(o: Observation) -> float:
    return float(np.sqrt(o.head_norm ** 2 + float(o.coordinate @ o.coordinate)))


def _c_norm(o: Observation) -> float:
    return float(np.linalg.norm(o.coordinate))


def _fn_fro(o: Observation) -> float:
    return float(np.linalg.norm(o.logits))


def _g_fro(o: Observation) -> float:
    return float(np.linalg.norm(o.gradient))


def _g_proj(o: Observation) -> float:
    return float(o.directions.gradient @ o.gradient)


def _c_proj1(o: Observation) -> float:
    return float(o.directions.parameter @ (o.directions.rotation @ o.coordinate))


def _fn_proj1(o: Observation) -> float:
    return float(o.directions.function @ o.logits.ravel())


def _margin(o: Observation) -> float:
    best_other = np.max(np.where(o.probe.true_mask, -np.inf, o.logits), axis=1)
    return float((o.logits[o.probe.index, o.probe.labels] - best_other).mean())


def _acc_probe(o: Observation) -> float:
    return float((o.logits.argmax(axis=1) == o.probe.labels).mean())


@dataclass(frozen=True)
class Observer:
    """One scalar observer: its name in every result file, and what it is."""

    name: str
    title: str        # appendix B's name for it
    family: str
    definition: str   # appendix B's definition of it
    fn: Callable[[Observation], float]
    state_only: bool = True

    def __call__(self, observation: Observation) -> float:
        return self.fn(observation)


def _register(*observers: Observer) -> Dict[str, Observer]:
    table: Dict[str, Observer] = {}
    for observer in observers:
        if observer.family not in FAMILIES:
            raise ValueError(f"{observer.name}: unknown family {observer.family!r}")
        if observer.name in table:
            raise ValueError(f"duplicate observer: {observer.name}")
        table[observer.name] = observer
    return table


#: The twelve, in the order appendix B's table lists them.
REGISTRY: Dict[str, Observer] = _register(
    Observer("loss_step", "instantaneous loss", "loss",
             "the loss on the current minibatch", _loss_step,
             # It contains the instantaneous drive weights, so it is not a function of the
             # optimiser state and it survives the zero-learning-rate test of section 5.3.
             # Kept in the panel precisely so that the contamination is visible.
             state_only=False),
    Observer("loss_full", "full-batch loss", "loss",
             "the loss on the whole training set", _loss_full),
    Observer("loss_probe", "probe loss", "loss",
             "cross-entropy on the probe set", _loss_probe),
    Observer("w_fro", "parameter norm", "norm",
             "(||theta_0||^2 + ||c||^2)^(1/2)", _w_fro),
    Observer("c_norm", "subspace norm", "norm", "||c||_2", _c_norm),
    Observer("fn_fro", "function-space norm", "norm", "||L||_F", _fn_fro),
    Observer("g_fro", "gradient norm", "gradient", "||g||_2", _g_fro),
    Observer("g_proj", "gradient projection", "gradient", "<u, g>", _g_proj),
    Observer("c_proj1", "fixed parameter projection", "projection", "<v, R c>", _c_proj1),
    Observer("fn_proj1", "function-space projection", "function", "<a, vec L>", _fn_proj1),
    Observer("margin", "margin", "function",
             "mean probe logit gap, true class less best other", _margin),
    Observer("acc_probe", "probe accuracy", "function",
             "accuracy on the probe set", _acc_probe),
)

#: The panel of section 5.4: all twelve.
PAPER_TWELVE: Tuple[str, ...] = tuple(REGISTRY)

#: The eleven that are functions of the optimiser state alone. An observer outside this
#: set can pass a recovery test without ever having read the optimiser, which is what
#: requirement 4 exists to catch.
STATE_ONLY: Tuple[str, ...] = tuple(n for n, o in REGISTRY.items() if o.state_only)

#: The twenty-direction calibration of appendix C: eight observers, at least one per
#: family. The two loss observers other than ``loss_full`` are dropped there because the
#: run is long and ``loss_step`` fails the silence control.
K20_PANEL: Tuple[str, ...] = ("w_fro", "c_norm", "g_fro", "g_proj", "c_proj1",
                              "fn_fro", "fn_proj1", "loss_full")

#: The four of those the twenty-direction configuration was selected on.
K20_CALIBRATION: Tuple[str, ...] = ("w_fro", "c_norm", "g_fro", "c_proj1")

#: The ceiling sweep of appendix R stores the same eight, so that its check arm can
#: reproduce the published median, and scores four -- one per family other than loss.
CEILING_STORED: Tuple[str, ...] = K20_PANEL
CEILING_PANEL: Tuple[str, ...] = ("c_norm", "g_fro", "c_proj1", "fn_proj1")

#: The Theiler contrast of appendix P: four, one per family, including the parameter norm
#: a run ordinarily logs.
THEILER_PANEL: Tuple[str, ...] = ("w_fro", "c_proj1", "g_fro", "loss_probe")


# ----------------------------------------------------------------- selection

def get(name: str) -> Observer:
    if name not in REGISTRY:
        raise KeyError(f"no such observer: {name}. Known: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def select(names: Optional[Tuple[str, ...]] = None) -> Tuple[Observer, ...]:
    """The observers of a panel, in the order given, checking every name exists."""
    return tuple(get(name) for name in (PAPER_TWELVE if names is None else names))


def by_family(family: str) -> Tuple[str, ...]:
    if family not in FAMILIES:
        raise KeyError(f"no such family: {family}. Known: {', '.join(FAMILIES)}")
    return tuple(name for name, o in REGISTRY.items() if o.family == family)


def record(observation: Observation,
           names: Optional[Tuple[str, ...]] = None) -> Dict[str, float]:
    """Evaluate a panel on one step."""
    return {o.name: o.fn(observation) for o in select(names)}


def table() -> Tuple[Dict[str, str], ...]:
    """Appendix B's table, as rows: name, family, definition.

    The article's table and this registry are the same object; if they disagree, one of
    them is wrong and the test that compares them says which.
    """
    return tuple({"name": o.title, "key": o.name, "family": o.family,
                  "definition": o.definition} for o in REGISTRY.values())
