"""The admissibility diagnostics, and the one summary rule that depends on them.

Two of the three diagnostics separate the two ways the estimator fails. The identifiability
ratio grows when the estimate is a property of the embedding space rather than of the data,
which is the stochastic regime; the trend-crossing count is near zero on a decaying transient,
where the ratio looks fine for the wrong reason. The third, the degeneracy indicator,
identifies no regime and is tested here for what it controls: which statistics a summary may
still report.
"""
from __future__ import annotations

import numpy as np
import pytest

from actdim.estimator import diagnostics, windows
from actdim.estimator.config import NEIGHBOUR_BASED, EstimatorConfig

CFG = EstimatorConfig(max_E=6, tau=28, k_neighbors=20, theiler="autocorr",
                      window=8000, stride=4000)
PERIOD = 80.0 * np.sqrt(2)
GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0


def torus(n: int = 8000) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return np.sin(2 * np.pi * t / PERIOD) + np.sin(2 * np.pi * GOLDEN * t / PERIOD + 0.4)


def noise(n: int = 8000) -> np.ndarray:
    return np.random.default_rng(3).standard_normal(n)


def transient(n: int = 8000) -> np.ndarray:
    return np.exp(-np.arange(n, dtype=np.float64) / 2000.0)


def recurrent_exactly(n: int = 20000) -> np.ndarray:
    """A sinusoid whose period divides the sampling interval: every delay vector repeats."""
    return np.sin(2 * np.pi * np.arange(n, dtype=np.float64) / 100.0)


# -- the identifiability ratio -------------------------------------------------

def test_the_ratio_is_near_one_where_a_dimension_is_resolvable():
    assert diagnostics.identifiability_ratio(torus(), CFG) == pytest.approx(1.0, abs=0.1)


def test_the_ratio_grows_where_the_estimate_is_a_property_of_the_embedding():
    """White noise fills whatever space it is embedded in, so doubling the space moves it."""
    assert diagnostics.identifiability_ratio(noise(), CFG) > 1.5


def test_the_ratio_refuses_a_denominator_it_cannot_divide_by():
    assert np.isnan(diagnostics.ratio(float("nan"), 2.0))
    assert np.isnan(diagnostics.ratio(0.0, 2.0))
    assert np.isnan(diagnostics.ratio(-1.0, 2.0))
    assert np.isnan(diagnostics.ratio(2.0, float("nan")))
    assert diagnostics.ratio(2.0, 3.0) == 1.5


# -- the trend-crossing count --------------------------------------------------

def test_the_count_is_large_on_a_recurrent_orbit_and_near_zero_on_a_transient():
    assert diagnostics.trend_crossings(torus()) > 100
    assert diagnostics.trend_crossings(transient()) <= 2
    assert diagnostics.trend_crossings(np.arange(8000, dtype=np.float64)) == 0


def test_the_count_refuses_a_window_it_cannot_fit_a_line_to():
    assert np.isnan(diagnostics.trend_crossings(np.array([1.0, np.nan, 3.0, 4.0])))
    assert np.isnan(diagnostics.trend_crossings(np.array([1.0, 2.0])))


def test_the_two_regimes_are_told_apart_only_by_using_both():
    """On a transient the ratio looks admissible and the count does not, which is the point.

    The estimate on a decaying transient is large -- the article reports about 29 against a
    true dimension of 1 -- and the ratio alone would not say so.
    """
    stochastic = diagnostics.diagnose(noise(), CFG)
    decaying = diagnostics.diagnose(transient(), CFG)
    recurrent = diagnostics.diagnose(torus(), CFG)

    # The ratio separates the stochastic regime from the other two, and nothing else.
    assert stochastic.identifiability_ratio > 1.5
    assert decaying.identifiability_ratio == pytest.approx(1.0, abs=0.1)
    assert recurrent.identifiability_ratio == pytest.approx(1.0, abs=0.1)

    # The count separates the transient from the other two, and nothing else.
    assert decaying.trend_crossings <= 2
    assert stochastic.trend_crossings > 100
    assert recurrent.trend_crossings > 100


# -- the degeneracy indicator --------------------------------------------------

def test_the_indicator_is_raised_when_the_floors_take_over():
    scored = windows.score(recurrent_exactly(8000), CFG)
    assert scored["degenerate"] is True
    assert scored["frac_floor"] > 0.99 and scored["frac_sumfloor"] > 0.99


def test_the_one_percent_rule_is_a_configuration_field_and_not_a_literal():
    """The same window, marked or not, according to the threshold it is read at."""
    x = recurrent_exactly(8000)
    assert windows.score(x, CFG)["degenerate"] is True
    forgiving = CFG.replace(degenerate_fraction=1.5)  # a fraction no window can exceed
    assert windows.score(x, forgiving)["degenerate"] is False


def test_a_wide_floor_marks_a_window_that_is_otherwise_clean():
    """Raising the distance floor above the data's own scale must raise the flag."""
    strict = CFG.replace(floor_distance=1e3)
    assert windows.score(torus(), strict)["degenerate"] is True
    assert windows.score(torus(), CFG)["degenerate"] is False


# -- what a degenerate window costs a summary ----------------------------------

def test_a_degenerate_window_loses_only_its_neighbour_based_statistics():
    """The fix to ``mg.summarise``.

    The archived summariser filtered every statistic by the flag. The roughness, the
    autocorrelation time and the spectral participation ratios never touch the neighbour
    search and are perfectly well defined on a window the floors ruined, so dropping them
    thinned the null columns exactly on the arms where the nulls decide the argument.
    """
    summary = windows.summarise(recurrent_exactly(), CFG)
    assert summary["n_windows"] > 1
    assert summary["frac_degenerate"] == 1.0

    for name in NEIGHBOUR_BASED:
        assert np.isnan(summary[name]), f"{name} survived a degenerate window"
    for name in ("roughness", "acorr", "specPR256", "specPR0", "PRdelay"):
        assert np.isfinite(summary[name]), f"{name} was dropped with the neighbour estimates"

    # The nulls are not merely finite; they are right. The reconstruction is a circle.
    assert summary["PRdelay"] == pytest.approx(2.0, abs=0.1)
    assert summary["specPR256"] == pytest.approx(1.0, abs=0.1)


def test_a_clean_record_keeps_everything():
    summary = windows.summarise(np.tile(torus(), 2), CFG)
    assert summary["frac_degenerate"] == 0.0
    for name in windows.statistic_names(CFG):
        assert np.isfinite(summary[name])
    assert summary["theiler_used"] == 140  # the delay span, under the cap


def test_a_sliding_trace_is_labelled_by_the_right_edge():
    x = np.tile(torus(), 2)
    right, traces = windows.sliding(x, CFG)
    assert right[0] == CFG.window - 1
    assert np.all(np.diff(right) == CFG.stride)
    assert len(traces["MG"]) == len(right)
    assert set(traces).issuperset(set(windows.statistic_names(CFG)))
