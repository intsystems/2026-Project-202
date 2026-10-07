"""Every parameter that can move an estimate, in one frozen object.

There is one configuration object and it is frozen, because in the archived tree the
estimator's settings were spread over three places: a dataclass, two module-level constants,
and a mutable module global. The global was ``THEILER_CAP``, and three worker scripts wrote
to it from inside their own processes -- one of them permanently, for the life of the worker.
That cap sets a published number: on the decaying-transient arm the autocorrelation rule
asks for roughly 1600 samples and gets 150, so the estimate reported there is the value at
the cap and not at the rule. A number that consequential cannot live in a variable that any
importer can assign to, so it is a field here.

Only *estimator* parameters belong in this object. The archived calibration grid mixed system
parameters (the drive period, the learning rate) in with the estimator's, so choosing a
configuration chose the data as well as the measurement. Keeping the two kinds apart in
different objects makes that mistake hard to repeat.

The dither's seed is deliberately not a field. It varies per series in the calibration
pipelines -- each trajectory is dithered with its own simulation seed -- and folding it in
would mean rebuilding the configuration for every series, at which point "the frozen
configuration" would no longer be one object. It is an argument to the scoring functions,
and the runtime layer derives it from the run's base seed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

#: The distance floor and the log-ratio-sum floor of appendix A. They are defaults of the
#: configuration rather than constants so that a test can raise them and see the degeneracy
#: machinery fire, but no experiment changes them.
FLOOR_DISTANCE = 1e-8
FLOOR_RATIO_SUM = 1e-5


@dataclass(frozen=True)
class EstimatorConfig:
    """The estimator as it will be run.

    ``tau`` is an integer lag, or ``"acorr"`` to measure one per window as a quarter of the
    autocorrelation time. A fixed lag is not neutral: on a period-400 torus the estimate at
    r = 2/4/6 reads 1.84/2.26/2.31 at ``tau=1`` and 3.45/18.1/20.3 at ``tau=20``, so a claim
    that the estimate saturates is a claim about the lag unless the lag adapts.

    ``theiler`` is ``0`` (no exclusion), an integer lag, ``"embedding"`` for the span of one
    delay vector, or ``"autocorr"`` for the larger of that span and the autocorrelation time.
    Without an exclusion an oversampled trajectory returns the dimension of its own tangent
    line rather than of the set it fills.

    ``window`` and ``stride`` describe how a record is divided, not how a window is scored.
    Three pipelines override them and no other field; see ``actdim.frozen``.
    """

    max_E: int = 20
    tau: Union[int, str] = 1
    k_neighbors: int = 5
    theiler: Union[int, str] = "embedding"
    theiler_cap: int = 150
    window: int = 6000
    stride: int = 500
    dither: float = 1e-9
    floor_distance: float = FLOOR_DISTANCE
    floor_ratio_sum: float = FLOOR_RATIO_SUM
    #: A window is marked degenerate when either floor is reached by more than this fraction
    #: of its denominator -- of the neighbour distances, or of the per-point sums.
    degenerate_fraction: float = 0.01
    #: Band counts for the spectral participation ratio; 0 is the native FFT resolution.
    #: A free parameter, so it was calibrated on the same split and by the same objective as
    #: the estimator's own parameters, or the comparison between them would have been rigged.
    spectral_bins: Tuple[int, ...] = (64, 256, 1024, 0)
    #: TwoNN discards this top fraction of the empirical CDF before fitting, where the tail
    #: is thin and the linear fit is dominated by a handful of points.
    twonn_discard: float = 0.1
    twonn_min_points: int = 20

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **changes: Any) -> "EstimatorConfig":
        """A copy with fields changed. The object is frozen, so this is how it is varied."""
        return dataclasses.replace(self, **changes)

    def tag(self) -> str:
        """A short, stable identifier for a configuration, for a filename or a column."""
        return (f"E{self.max_E}_t{self.tau}_k{self.k_neighbors}"
                f"_th{self.theiler}_W{self.window}")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "EstimatorConfig":
        """Build from a stored mapping, rejecting anything this class does not know.

        A silently ignored key is how a stored configuration comes to differ from the
        configuration that was actually used, so an unknown one is an error and names itself.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"unknown estimator parameter(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(known))}"
            )
        values = dict(values)
        if "spectral_bins" in values:
            values["spectral_bins"] = tuple(values["spectral_bins"])
        return cls(**values)


#: The defaults, as a named object, so a caller can say what it started from.
DEFAULT = EstimatorConfig()

#: The statistics that come out of the neighbour search, and are therefore the only ones a
#: degenerate window invalidates. See ``actdim.estimator.windows.summarise``.
NEIGHBOUR_BASED = ("MG", "LB", "TwoNN")
