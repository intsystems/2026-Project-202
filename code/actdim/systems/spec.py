"""The contract every system module obeys, and the epilogue they all share.

A system module exposes three things and nothing else:

* a frozen config dataclass, whose ``active_dimension`` says what the construction fixes;
* ``simulate(config, seed) -> Simulation``, returning named scalar series and a ground
  truth;
* module-level prose saying why the active dimension is what it claims to be.

The ground truth is not a label. Requirement 1 asks that the constructed dimension be
confirmed on the recorded trajectory, so :class:`GroundTruth` carries the measurements as
well as the claim, and :meth:`GroundTruth.failures` names any that disagree. The archived
tree verified this in two of its six systems; in the other four the claim ``true_dim = k``
appeared only in a comment.

Nothing here writes a file. A system returns arrays; the experiment module decides what to
store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..linalg import RANK_TOL as TRAJECTORY_RANK_TOL_DEFAULT
from ..linalg import participation_ratio, spectrum
from ..runtime.determinism import rng as stream_rng

#: Measurement noise added to every recorded series, as a fraction of its own spread.
#: The archived experiments spelled this ``1e-6 * standard_normal`` five times and
#: ``1 / obs_snr`` with ``obs_snr`` of 1e4 or 1e6 twice more.
DEFAULT_JITTER = 1e-6


def standardise(series: Dict[str, np.ndarray], seed: int,
                jitter: float = DEFAULT_JITTER) -> Dict[str, np.ndarray]:
    """Centre each series, scale it to unit spread, and add a little measurement noise.

    Standardising costs nothing -- the estimate is scale invariant -- and it makes the
    signal-to-noise ratio the same number for every observer and every rank. Without it the
    comparison is rigged: at fixed per-coordinate noise a single oscillator moves a norm
    least, so rank one has the worst ratio in the sweep and reads *higher* than rank two,
    purely as a noise-floor artefact. That inversion is real and this removes it.

    The jitter breaks exact ties between delay vectors, which would otherwise put a
    neighbour distance on the numerical floor and make the estimate meaningless. Each
    series draws from its own named stream, so adding an observer does not move the noise
    on the others.
    """
    out: Dict[str, np.ndarray] = {}
    for name, values in series.items():
        x = np.asarray(values, dtype=float)
        spread = float(x.std())
        z = (x - x.mean()) / spread if spread > 0.0 else np.zeros_like(x)
        noise = stream_rng(seed, "observation_noise:" + name).standard_normal(x.size)
        out[name] = z + jitter * noise
    return out


#: A check exists where a claim exists, and nowhere else.
#:
#: Every system claims that its ``r`` directions are all excited, so every system is
#: checked for a covariance rank of ``r`` and for a weakest direction that is not a
#: rounding error next to the strongest.
#:
#: Only the two image-data systems claim more than that. They equalise the per-mode
#: response gain -- by a measured mixing matrix in one case and by QR-whitening the
#: function-space Jacobian in the other -- so that the ``r`` directions are excited
#: *comparably* and the effective rank equals ``r``; appendix F tabulates it. Those get the
#: effective-rank check too, at ``equalised=True``.
#:
#: The five synthetic rungs make no such claim: their amplitudes are drawn unequal and
#: their response gains vary across the drive band, so their effective rank sits below
#: ``r`` by construction and asserting otherwise would be asserting something false. It is
#: measured and reported instead, and what it comes to is worth knowing: 0.85 r for the
#: diagonal matrix and both regressions, 0.46 r for the decoder and 0.23 r for the
#: parameter subspace at r = 20. The archived code measured none of them.
EXCITATION_FLOOR = 0.9
DIRECTION_FLOOR = 5e-3

#: Where the driven coordinates do not interact -- the diagonal matrix, and regression on
#: one-hot inputs, where the update is coordinate-wise -- a sharper check is available: the
#: effective rank computed from the eigenvalues must equal the one computed from the
#: variances alone. They differ only if two directions have coupled, which is what a
#: near-resonance between two drive frequencies would look like.
DIAGONAL_TOLERANCE = 0.05
DIAGONAL_TOLERANCE_FRACTION = 0.02


def excitation(block: np.ndarray, expected_rank: int, diagonal: bool = False,
               equalised: bool = False, tol: float = TRAJECTORY_RANK_TOL_DEFAULT
               ) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Measure a driven block and say whether it carries the rank it was built to carry.

    Returns the measurements and the named checks, which the system puts into its
    :class:`GroundTruth`. This is the one place the question "did the construction do what
    it claims" is answered, so a system cannot quietly answer it differently.
    """
    expected_rank = int(expected_rank)
    variances = spectrum(block)
    singular = np.sqrt(variances)
    if singular.size == 0 or singular[0] <= 0.0:
        return ({"covariance_rank": 0.0, "effective_rank": float("nan"),
                 "direction_ratio": 0.0, "excited_fraction": 0.0},
                {"covariance_rank": expected_rank == 0, "direction_ratio": False})
    rank = int(np.sum(singular > singular[0] * tol))
    # The ratio is taken at the *expected* direction, not at the last one: a block with
    # more columns than the construction excites has zeros beyond that point by design,
    # and comparing against them would fail every control on purpose.
    weakest = (float(singular[expected_rank - 1] / singular[0])
               if 0 < expected_rank <= singular.size else 0.0)
    measured = {"covariance_rank": float(rank),
                "effective_rank": participation_ratio(variances),
                "direction_ratio": weakest}
    measured["excited_fraction"] = measured["effective_rank"] / max(expected_rank, 1)
    checks = {
        "covariance_rank": rank == expected_rank,
        "direction_ratio": weakest >= DIRECTION_FLOOR,
    }
    if equalised:
        checks["effective_rank"] = (
            measured["effective_rank"] >= EXCITATION_FLOOR * expected_rank)
    if diagonal:
        independent = participation_ratio(np.var(np.asarray(block, dtype=float), axis=0))
        limit = max(DIAGONAL_TOLERANCE, DIAGONAL_TOLERANCE_FRACTION * expected_rank)
        measured["independent_effective_rank"] = independent
        checks["independent"] = abs(measured["effective_rank"] - independent) <= limit
    return measured, checks


@dataclass(frozen=True)
class GroundTruth:
    """What the construction fixes, and the measurements that confirm it.

    ``active_dimension`` is the number the estimator is asked to recover. ``measured``
    holds the quantities that check it -- effective ranks, functional ranks, the resonance
    margin -- and ``checks`` records, per named check, whether the measurement agreed.
    """

    active_dimension: float
    measured: Dict[str, float] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)

    def failures(self) -> List[str]:
        return sorted(name for name, ok in self.checks.items() if not ok)

    @property
    def verified(self) -> bool:
        return not self.failures()

    def require(self) -> "GroundTruth":
        """Raise if a construction did not excite the rank it claims.

        Called by the experiment, not by the library, so that a diagnostic run can still
        look at a trajectory whose truth failed.
        """
        bad = self.failures()
        if bad:
            detail = ", ".join(f"{name}={self.measured.get(name, float('nan')):.4g}"
                               for name in bad if name in self.measured)
            raise ValueError(
                f"the construction claims active dimension {self.active_dimension:g} but "
                f"{', '.join(bad)} disagree" + (f" ({detail})" if detail else ""))
        return self


@dataclass(frozen=True)
class Simulation:
    """One run of one system: the scalar series an observer sees, and the truth."""

    series: Dict[str, np.ndarray]
    truth: GroundTruth
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.series)

    @property
    def length(self) -> int:
        return int(len(next(iter(self.series.values())))) if self.series else 0

    def __getitem__(self, name: str) -> np.ndarray:
        return self.series[name]


# ----------------------------------------------------------------- the catalogue

@dataclass(frozen=True)
class SystemEntry:
    """One constructed system, as the article's ladder lists it."""

    id: str
    title: str
    config: type
    simulate: Callable[..., Simulation]
    paper: str = ""


SYSTEMS: Dict[str, SystemEntry] = {}

#: The six systems of section 5, in the order table 3 lists them. The image-data system of
#: section 5.4 appears twice: once as the parameter subspace scored in rows six and seven,
#: once as the function-space variant scored beside it.
LADDER: Tuple[str, ...] = ("matrix", "regression.linear", "regression.logistic",
                           "decoder", "subspace", "digits_function", "digits_parameter")


def register(id: str, title: str, config: type, paper: str = "") -> Callable:
    """Register the decorated ``simulate`` under a stable id."""

    def decorate(fn: Callable[..., Simulation]) -> Callable[..., Simulation]:
        if id in SYSTEMS:
            raise ValueError(f"duplicate system id: {id}")
        SYSTEMS[id] = SystemEntry(id=id, title=title, config=config, simulate=fn, paper=paper)
        return fn

    return decorate


#: Systems named by ``LADDER`` that have no module yet. Listed rather than silently
#: absent: a ladder row missing from the catalogue is a row of the article that cannot be
#: regenerated, and it should be visible in ``actdim list`` and in the catalogue itself
#: rather than surfacing as an import error three frames down.
NOT_PORTED: Dict[str, str] = {}


def load() -> Dict[str, SystemEntry]:
    """Import every system module that exists, populating the catalogue."""
    from importlib import import_module

    for name in ("matrix", "regression", "decoder", "subspace", "digits_function",
                 "digits_parameter"):
        if name in NOT_PORTED:
            continue
        import_module(f"{__package__}.{name}")
    return SYSTEMS


def missing() -> Dict[str, str]:
    """Ladder rows with no implementation, and where the archived one is."""
    load()
    return {name: NOT_PORTED[name] for name in LADDER
            if name in NOT_PORTED or (name not in SYSTEMS and name in NOT_PORTED)}


def get(id: str) -> SystemEntry:
    if id not in SYSTEMS:
        load()
    if id not in SYSTEMS:
        raise KeyError(f"no such system: {id}")
    return SYSTEMS[id]


def series_frame(simulation: Simulation) -> Any:
    """The series as a DataFrame, for an experiment that wants to store them.

    Imported lazily: a system does not need pandas to run.
    """
    import pandas as pd

    return pd.DataFrame(simulation.series)


def truth_row(simulation: Simulation, **extra: Any) -> Dict[str, Any]:
    """One flat row recording what a run's construction achieved."""
    row: Dict[str, Any] = {"active_dimension": simulation.truth.active_dimension,
                           "verified": simulation.truth.verified}
    row.update({f"measured_{k}": v for k, v in simulation.truth.measured.items()})
    row.update({f"check_{k}": v for k, v in simulation.truth.checks.items()})
    row.update(extra)
    return row
