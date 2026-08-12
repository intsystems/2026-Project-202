"""Online regression against a moving teacher, linear and logistic.

The second and third rungs. A ``D``-dimensional model is trained by full-batch gradient
descent on one-hot inputs, and exactly ``r`` of the ``D`` targets move quasiperiodically
while the rest are constant and already fitted. There is a real optimiser here, taking a
real gradient step against a real objective; what is arranged is only which part of the
problem is still moving.

Why the active dimension is ``r``. With one-hot inputs the objective decouples: coordinate
``i`` sees only ``w_i`` and target ``y_i``, so the update is
``w_i <- (1 - eta) w_i + eta y_i`` in the linear case -- an independent one-pole filter per
coordinate. A filter does not change the dimension of the set its input traces, so the
weight trajectory closes onto the same ``r``-torus as the drive, at attenuated amplitudes.
The ``D - r`` fitted coordinates sit at their targets and contribute nothing.

``link="logistic"`` replaces the squared loss with binary cross-entropy through a sigmoid,
so both the prediction and the gradient are nonlinear, the response gain becomes
state-dependent, and the estimator is asked the same question about a curved problem. The
targets are probabilities whose drivers live in logit space, which is where they can be
made independent.

Both are decoupled by coordinate, which is what makes the sharp diagonal ground-truth check
available here: if two drive frequencies came close to a resonance the weight covariance
would stop being diagonal and the check would fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.special import expit

from ..runtime.determinism import rng as stream_rng
from .drive import DriveConfig, build_drive
from .spec import GroundTruth, Simulation, excitation, register, standardise

LINKS: Tuple[str, ...] = ("identity", "logistic")


@dataclass(frozen=True)
class RegressionConfig:
    """One online-regression system. ``k`` targets move; ``dimension - k`` are fitted."""

    k: int = 4
    dimension: int = 64
    link: str = "identity"
    window: int = 4000
    burn: int = 1000
    eta: float = 0.2
    jitter: float = 1e-6
    drive: DriveConfig = field(default_factory=lambda: DriveConfig(
        cycles_per_window=1200.0, window=4000, amp_scale=0.10,
        amp_low=0.5, amp_span=0.5))

    @property
    def active_dimension(self) -> float:
        return float(self.k)

    @property
    def length(self) -> int:
        return self.window + self.burn


#: The logistic arm drives logits, not probabilities, and needs an amplitude large enough
#: for the sigmoid to bend. These are the archived constants for that arm, as a config.
LOGISTIC_DRIVE = DriveConfig(cycles_per_window=1200.0, window=4000, amp_scale=1.2,
                             amp_low=0.7, amp_span=0.3)


def _baseline(config: RegressionConfig, seed: int) -> np.ndarray:
    """Where the untouched coordinates sit.

    In the linear arm the baseline is held away from zero, for the same reason the diagonal
    matrix holds its offsets away from zero: a norm read at zero is not injective. In the
    logistic arm the baseline is a logit, and zero is the interesting place to be.
    """
    generator = stream_rng(seed, "regression_baseline")
    if config.link == "logistic":
        return 0.4 * generator.standard_normal(config.dimension)
    base = 1.0 + 0.25 * generator.standard_normal(config.dimension)
    base[np.abs(base) < 0.5] = 0.5
    return base


def trajectory(config: RegressionConfig, seed: int):
    """Run the optimiser, returning the weight, gradient and prediction histories."""
    if config.link not in LINKS:
        raise ValueError(f"unknown link {config.link!r}. Known: {', '.join(LINKS)}")
    n, k = config.length, int(config.k)
    baseline = _baseline(config, seed)
    drive = build_drive(config.drive, k, seed)

    target = np.tile(baseline, (n, 1))
    target[:, :k] += drive.amplitudes * drive.waves(n)
    if config.link == "logistic":
        target = expit(target)

    weights = np.empty_like(target)
    gradients = np.empty_like(target)
    predictions = np.empty_like(target)
    losses = np.empty(n)
    w = baseline.copy()
    for step in range(n):
        if config.link == "logistic":
            p = expit(w)
            gradient = p - target[step]
            losses[step] = float(np.sum(np.logaddexp(0.0, w) - target[step] * w))
        else:
            p = w
            gradient = w - target[step]
            losses[step] = 0.5 * float(gradient @ gradient)
        weights[step], gradients[step], predictions[step] = w, gradient, p
        w = w - config.eta * gradient

    keep = slice(config.burn, None)
    return weights[keep], gradients[keep], predictions[keep], losses[keep], drive


def observe(weights: np.ndarray, gradients: np.ndarray, predictions: np.ndarray,
            losses: np.ndarray, link: str, seed: int) -> Dict[str, np.ndarray]:
    """The scalars a run of this kind would log."""
    dimension = weights.shape[1]
    weight_direction = stream_rng(seed, "observer_weight").standard_normal(dimension)
    gradient_direction = stream_rng(seed, "observer_gradient").standard_normal(dimension)
    series = {
        "weight_fro": np.linalg.norm(weights, axis=1),
        "weight_trace": weights.sum(axis=1),
        "weight_projection": weights @ weight_direction,
        "gradient_fro": np.linalg.norm(gradients, axis=1),
        "gradient_projection": gradients @ gradient_direction,
        "loss": losses,
    }
    if link == "logistic":
        series["probability_fro"] = np.linalg.norm(predictions, axis=1)
    return series


def logistic_config(**overrides) -> RegressionConfig:
    """The logistic arm's settings, which differ from the linear arm's in four places."""
    settings = dict(link="logistic", burn=1500, eta=1.0, drive=LOGISTIC_DRIVE)
    settings.update(overrides)
    return RegressionConfig(**settings)


@register("regression.linear", "Online linear regression", RegressionConfig,
          paper="sec:ladder")
def simulate(config: RegressionConfig, seed: int = 0) -> Simulation:
    """Train the regressor and read it through six -- or, under the sigmoid, seven -- scalars."""
    weights, gradients, predictions, losses, drive = trajectory(config, seed)
    series = standardise(observe(weights, gradients, predictions, losses,
                                 config.link, seed), seed, config.jitter)
    measured, checks = excitation(weights[:, :config.k], config.k, diagonal=True)
    measured["resonance_margin"] = drive.margin
    info = {"link": config.link, "k": config.k, "dimension": config.dimension,
            "eta": config.eta, "window": config.window, "burn": config.burn,
            **drive.report()}
    return Simulation(series=series,
                      truth=GroundTruth(active_dimension=config.active_dimension,
                                        measured=measured, checks=checks),
                      info=info)


# The two arms are one construction under two links, so they are one function under two
# ids rather than two near-identical modules.
register("regression.logistic", "Online logistic regression", RegressionConfig,
         paper="sec:ladder")(simulate)
