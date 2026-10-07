"""An oscillating diagonal matrix: the first rung, with no optimiser and no feedback.

``W(t) = diag(b_1 + d_1(t), ..., b_D + d_D(t))`` with ``b_i`` nonzero and ``r`` of the
coordinates oscillating, ``d_i(t) = a_i sin(2 pi f_i t + phi_i)``. There is no learning
here at all, which is why a negative result on this system would be decisive: nothing about
an optimiser could be blamed for it.

Why the active dimension is ``r``. The state of the moving part is the vector of phases,
and at integer sampling times the closure of the orbit is the closure of the subgroup that
``(f_1, ..., f_r)`` generates in the ``r``-torus. That closure is the whole torus -- a
smooth compact manifold of dimension ``r`` -- exactly when ``1, f_1, ..., f_r`` are
rationally independent, which :mod:`actdim.systems.drive` arranges and measures.

Why ``b_i`` must be nonzero. The squared Frobenius norm is
``sum b_i^2 + 2 sum b_i d_i(t) + sum d_i(t)^2``, so for small amplitudes the linear term
dominates and the norm is close to a generic linear functional of the oscillators. At
``b_i = 0`` it would be ``sum d_i^2``, which is invariant under ``d -> -d`` and therefore
not injective on the torus: the observer would collapse the very geometry it is meant to
show.

The controls matter as much as the main mode. ``sync`` drives several coordinates from one
phase: the set traced has one degree of freedom rather than several and is no smoother for
it, so an estimator that returns ``r`` there is not being conservative, it is hallucinating.
``sync_phased`` is the harder version -- one angle, a distinct fixed phase per coordinate --
whose *covariance* rank is two while its *manifold* dimension is still one, which is the
clearest case of the two numbers coming apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..runtime.determinism import rng as stream_rng
from .drive import Drive, DriveConfig, build_drive
from .spec import GroundTruth, Simulation, excitation, register, standardise

#: How much a mode's measured effective rank may differ from the value the construction
#: predicts before the run is called unverified: the larger of this and two per cent of
#: the rank. Both are far tighter than any difference the estimator is asked to resolve.
RANK_TOLERANCE = 0.05
RANK_TOLERANCE_FRACTION = 0.02


@dataclass(frozen=True)
class MatrixConfig:
    """One oscillating diagonal matrix.

    ``mode`` selects the main construction or one of the controls; ``k`` is the number of
    moving coordinates, which is the active dimension only in ``quasiperiodic`` and
    ``noise_smooth``.
    """

    k: int = 4
    dimension: int = 64
    length: int = 4000
    mode: str = "quasiperiodic"
    snr: float = float("inf")
    projections: int = 4
    jitter: float = 1e-6
    drive: DriveConfig = field(default_factory=lambda: DriveConfig(
        cycles_per_window=1000.0, window=4000, amp_scale=0.1,
        amp_low=0.5, amp_span=0.5))

    @property
    def active_dimension(self) -> float:
        """The dimension of the closure of the sampled orbit -- not always ``k``."""
        return _MODES[self.mode].dimension(self.k)


@dataclass(frozen=True)
class _Mode:
    dimension_of: object
    rank_of: object
    diagonal_covariance: bool  # the driven block's covariance is diagonal, so its
                               # participation ratio is predicted by the amplitudes alone

    def dimension(self, k: int) -> float:
        return float(self.dimension_of(k))

    def rank(self, k: int) -> int:
        return int(self.rank_of(k))


_MODES: Dict[str, _Mode] = {
    # r independent phases: an r-torus.
    "quasiperiodic": _Mode(lambda k: float(k), lambda k: k, True),
    # k coordinates, one angle: a circle, whatever k is.
    "sync": _Mode(lambda k: 1.0, lambda k: 1, False),
    # one angle with a fixed phase per coordinate: still a circle, but an ellipse in the
    # plane two coordinates span, so the covariance rank is two and the dimension is one.
    "sync_phased": _Mode(lambda k: 1.0, lambda k: min(2, k), False),
    # a pure rescaling along a fixed direction: a curve, never revisited.
    "scale_monotone": _Mode(lambda k: 1.0, lambda k: 1, False),
    "scale_periodic": _Mode(lambda k: 1.0, lambda k: 1, False),
    # k white sequences: not a manifold at all, so the true dimension is the embedding one.
    # The diagonal check does not apply to either noise mode: independent draws still
    # correlate a little over a finite record, and for the smoothed mode the correlation
    # time makes that a real effect rather than a numerical one.
    "noise": _Mode(lambda k: float("inf"), lambda k: k, False),
    # k smooth random signals: a k-dimensional stochastic process, differentiable, so
    # neither the white-noise case nor a torus.
    "noise_smooth": _Mode(lambda k: float(k), lambda k: k, False),
}

MODES: Tuple[str, ...] = tuple(_MODES)


def _base(dimension: int, seed: int) -> np.ndarray:
    """Nonzero diagonal offsets, held away from zero for the reason above."""
    b = 1.0 + 0.25 * stream_rng(seed, "matrix_base").standard_normal(dimension)
    small = np.abs(b) < 0.5
    b[small] = 0.5 * np.sign(b[small] + 1e-12)
    return b


def trajectory(config: MatrixConfig, seed: int) -> Tuple[np.ndarray, Drive, np.ndarray]:
    """The ``(n, D)`` diagonal, the drive that made it, and the moving coordinates."""
    if config.mode not in _MODES:
        raise ValueError(f"unknown mode {config.mode!r}. Known: {', '.join(MODES)}")
    n, k = int(config.length), int(config.k)
    active = np.arange(k)
    diag = np.tile(_base(config.dimension, seed), (n, 1))
    t = np.arange(n, dtype=float)

    single = config.mode in ("sync", "sync_phased", "scale_periodic")
    drive = build_drive(config.drive, 1 if single else k, seed)
    amplitude = config.drive.amp_scale * (
        config.drive.amp_low
        + config.drive.amp_span * stream_rng(seed, "drive_amplitudes").random(k))
    rng = stream_rng(seed, "matrix_noise")

    if config.mode == "quasiperiodic":
        diag[:, active] += amplitude * drive.waves(n)
    elif config.mode == "sync":
        diag[:, active] += amplitude * drive.waves(n)[:, [0] * k]
    elif config.mode == "sync_phased":
        phases = stream_rng(seed, "matrix_phases").uniform(0.0, 2.0 * np.pi, k)
        diag[:, active] += amplitude * np.sin(
            2.0 * np.pi * drive.frequencies[0] * t[:, None] + phases)
    elif config.mode == "scale_monotone":
        diag = diag * (1.0 + 0.5 * (t / n))[:, None]
    elif config.mode == "scale_periodic":
        diag = diag * (1.0 + config.drive.amp_scale * drive.waves(n)[:, 0])[:, None]
    elif config.mode == "noise":
        diag[:, active] += amplitude * rng.standard_normal((n, k))
    elif config.mode == "noise_smooth":
        kernel = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2)
        kernel = kernel / kernel.sum()
        for j in range(k):
            smooth = np.convolve(rng.standard_normal(n), kernel, mode="same")
            diag[:, j] += amplitude[j] * smooth / (smooth.std() + 1e-12)

    if np.isfinite(config.snr):
        diag = diag + (config.drive.amp_scale / config.snr) * rng.standard_normal(diag.shape)
    return diag, drive, amplitude


def observe(diag: np.ndarray, k: int, seed: int, projections: int = 4) -> Dict[str, np.ndarray]:
    """Scalar series from the ``(n, D)`` diagonal.

    Three groups, and the grouping is a prediction made before the measurement, not a
    result. The functions that mix every moving coordinate with generic coefficients
    should read ``r``; those that see a strict subset should read the size of that subset,
    because the dimension of their *image* really is lower; those that see none of them are
    constants plus noise and no estimate is defined.
    """
    n, dimension = diag.shape
    generator = stream_rng(seed, "observer_projections")
    active = np.arange(k)
    inactive = np.setdiff1d(np.arange(dimension), active)
    out: Dict[str, np.ndarray] = {
        "norm_fro": np.linalg.norm(diag, axis=1),
        "norm_fro_sq": (diag ** 2).sum(axis=1),
        "trace": diag.sum(axis=1),
        "logdet": np.log(np.abs(diag) + 1e-12).sum(axis=1),
        "norm_l1": np.abs(diag).sum(axis=1),
    }
    for j in range(projections):
        out[f"proj_rand{j}"] = diag @ generator.standard_normal(dimension)
    if len(active):
        out["coord_active0"] = diag[:, active[0]].copy()
        half = active[:max(1, len(active) // 2)]
        weights = np.zeros(dimension)
        weights[half] = generator.standard_normal(len(half))
        out["proj_half_active"] = diag @ weights
    if len(inactive):
        out["coord_inactive"] = diag[:, inactive[0]].copy()
        weights = np.zeros(dimension)
        weights[inactive] = generator.standard_normal(len(inactive))
        out["proj_inactive_only"] = diag @ weights
    return out


def image_dimension(name: str, k: int) -> float:
    """What the estimator should return on this observer -- not always ``k``.

    An observer that sees a strict subset of the angles has an image of lower dimension,
    and a low reading there is the estimator being right about a lossy observer.
    """
    if name == "coord_active0":
        return 1.0
    if name == "proj_half_active":
        return float(max(1, k // 2))
    if name in ("coord_inactive", "proj_inactive_only"):
        return float("nan")  # constant plus noise: no manifold
    return float(k)


#: What each observer is expected to do, declared before the sweep runs.
EXPECTED: Dict[str, str] = {
    "norm_fro": "good", "norm_fro_sq": "good", "trace": "good", "logdet": "good",
    "norm_l1": "good", "proj_rand0": "good", "proj_rand1": "good",
    "proj_rand2": "good", "proj_rand3": "good",
    "coord_active0": "bad_by_construction",     # one angle: the image is a circle
    "proj_half_active": "bad_by_construction",  # floor(k/2) angles
    "coord_inactive": "degenerate",             # constant plus noise
    "proj_inactive_only": "degenerate",
}


def ground_truth(config: MatrixConfig, diag: np.ndarray, drive: Drive) -> GroundTruth:
    """Confirm on the recorded trajectory that the construction did what it claims.

    The moving block is checked against the rank the mode predicts, which is the rank of
    the *covariance* and not always the dimension of the orbit: ``sync_phased`` traces an
    ellipse, so its covariance rank is two while its manifold dimension is one. Where the
    driven coordinates do not interact the sharper diagonal check applies as well.
    """
    mode = _MODES[config.mode]
    expected = mode.rank(config.k)
    measured, checks = excitation(diag[:, :config.k], expected,
                                  diagonal=mode.diagonal_covariance)
    if expected == 1:
        # A single excited direction has effective rank one exactly; the floor the shared
        # check applies is far too generous to say anything here.
        checks["effective_rank"] = abs(measured["effective_rank"] - 1.0) <= RANK_TOLERANCE
    measured["resonance_margin"] = drive.margin
    measured["realised_band"] = drive.report()["realised_band"]
    return GroundTruth(active_dimension=config.active_dimension,
                       measured=measured, checks=checks)


@register("matrix", "An oscillating diagonal matrix", MatrixConfig, paper="sec:matrix")
def simulate(config: MatrixConfig, seed: int = 0) -> Simulation:
    """Record the diagonal, read it through every observer, and check the truth."""
    diag, drive, _ = trajectory(config, seed)
    series = standardise(observe(diag, config.k, seed, config.projections), seed,
                         config.jitter)
    info = {"mode": config.mode, "k": config.k, "dimension": config.dimension,
            "length": config.length, "snr": config.snr, **drive.report(),
            "image_dimension": {name: image_dimension(name, config.k) for name in series},
            "expected": dict(EXPECTED)}
    return Simulation(series=series, truth=ground_truth(config, diag, drive), info=info)


def simulate_transition(config: MatrixConfig, schedule: Sequence[int], seed: int = 0,
                        segment: int = 1200, ramp: int = 0,
                        k_max: Optional[int] = None) -> Simulation:
    """Concatenate regimes of known dimension into one non-stationary record.

    Each oscillator keeps its own frequency and phase throughout, so a change of dimension
    is an amplitude switching on or off and not a discontinuity in the series. With
    ``ramp > 0`` the switch is linear over that many samples and the truth is undefined
    inside it: an oscillator whose amplitude is below the noise floor is not a degree of
    freedom the data contains, whatever the nominal schedule says.
    """
    schedule = tuple(int(s) for s in schedule)
    k_max = int(k_max or max(schedule))
    n = segment * len(schedule)
    drive = build_drive(config.drive, k_max, seed)
    diag = np.tile(_base(config.dimension, seed), (n, 1))
    amplitude = config.drive.amp_scale * (
        config.drive.amp_low
        + config.drive.amp_span * stream_rng(seed, "drive_amplitudes").random(k_max))

    envelope = np.zeros((n, k_max))
    for index, k in enumerate(schedule):
        envelope[index * segment:(index + 1) * segment, :k] = 1.0
    if ramp > 0:
        kernel = np.ones(ramp) / ramp
        for j in range(k_max):
            envelope[:, j] = np.convolve(envelope[:, j], kernel, mode="same")

    diag[:, :k_max] += amplitude * envelope * drive.waves(n)
    if np.isfinite(config.snr):
        diag = diag + (config.drive.amp_scale / config.snr) * stream_rng(
            seed, "matrix_noise").standard_normal(diag.shape)

    series = standardise(observe(diag, k_max, seed, config.projections), seed,
                         config.jitter)
    truth = np.repeat(np.asarray(schedule, dtype=float), segment)
    info = {"schedule": schedule, "segment": segment, "ramp": ramp, "k_max": k_max,
            "truth_series": truth, **drive.report()}
    return Simulation(
        series=series,
        truth=GroundTruth(active_dimension=float(np.median(truth)),
                          measured={"resonance_margin": drive.margin},
                          checks={}),
        info=info)
