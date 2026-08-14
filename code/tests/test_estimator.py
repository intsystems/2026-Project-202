"""The estimator: agreement with the archived numbers, and the analytic cases.

The regression test is the important one. ``fixtures/k20_windows.npz`` holds three real
logged windows from the twenty-direction calibration run, standardised exactly as that
pipeline standardised them, and ``fixtures/k20_frozen_scores.csv`` holds the numbers the
archived implementation published for them. Scoring them here has to reproduce those numbers.

The analytic cases say what the numbers mean: a sinusoid traces a circle, a one-dimensional
set, and a sum of two incommensurate sinusoids traces a two-torus. Those are the only cases
where the answer is known in advance rather than measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from actdim import frozen
from actdim.estimator import companions, embedding, mle, surrogates, windows
from actdim.estimator.calibration import Calibration
from actdim.estimator.config import EstimatorConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: A window of 8000 samples, a lag of a quarter period and 20 neighbours: enough of the
#: circle for the neighbour statistic to see it, and a Theiler exclusion that is the delay
#: span rather than the cap.
ANALYTIC = EstimatorConfig(max_E=6, tau=28, k_neighbors=20, theiler="autocorr",
                           window=8000, stride=8000)
PERIOD = 80.0 * np.sqrt(2)  # irrational, so the orbit never closes on itself exactly
GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0


def series(n: int = 8000) -> np.ndarray:
    return np.arange(n, dtype=np.float64)


def sine(n: int = 8000, period: float = PERIOD) -> np.ndarray:
    return np.sin(2 * np.pi * series(n) / period)


def torus(n: int = 8000, period: float = PERIOD) -> np.ndarray:
    t = series(n)
    return np.sin(2 * np.pi * t / period) + np.sin(2 * np.pi * GOLDEN * t / period + 0.4)


# -- the regression against the archived implementation ------------------------

def archived_rows():
    import csv

    with (FIXTURES / "k20_frozen_scores.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


#: Everything the archived scorer published for these windows, and how closely the port has
#: to match it. The residual is the one difference the port makes deliberately: each window
#: is standardised on its own rather than inheriting the record's standardisation, which
#: rescales the 1e-9 dither by the window's spread. TwoNN gets more room because its fit is
#: on the ratio of the two smallest distances, where that rescaling lands hardest.
TOLERANCE = {"MG": 1e-10, "LB": 1e-10, "TwoNN": 1e-8, "PRdelay": 1e-10,
             "specPR64": 1e-10, "specPR256": 1e-10, "specPR1024": 1e-10, "specPR0": 1e-10,
             "roughness": 1e-12, "acorr": 0.0, "tau_used": 0.0, "theiler_used": 0.0}


@pytest.mark.parametrize("row", archived_rows(), ids=lambda r: r["key"])
def test_reproduces_the_archived_frozen_scores(row):
    cfg = frozen.twenty_direction()
    window = np.load(FIXTURES / "k20_windows.npz")[row["key"]]
    scored = windows.score(window, cfg, seed=int(row["seed"]))

    for name, tolerance in TOLERANCE.items():
        expected = float(row[name])
        assert abs(scored[name] - expected) <= tolerance, (
            f"{name}: {scored[name]!r} against the archived {expected!r}")
    assert scored["degenerate"] is False


def test_the_archived_windows_are_not_degenerate():
    """The published cells contain no degenerate window, and neither may these."""
    cfg = frozen.twenty_direction()
    store = np.load(FIXTURES / "k20_windows.npz")
    for row in archived_rows():
        estimate = mle.estimate(store[row["key"]], cfg, seed=int(row["seed"]))
        assert not estimate.degenerate
        assert estimate.floor_distance_fraction == 0.0
        assert estimate.floor_sum_fraction == 0.0


def test_summarise_of_a_single_window_record_is_that_window():
    cfg = frozen.twenty_direction()
    window = np.load(FIXTURES / "k20_windows.npz")["qp_r04_s00__w_fro"]
    summary = windows.summarise(window, cfg, seed=0)
    assert summary["n_windows"] == 1
    assert summary["MG"] == pytest.approx(windows.score(window, cfg, seed=0)["MG"], abs=0)


# -- the analytic cases --------------------------------------------------------

def test_a_sinusoid_measures_about_one():
    """A delay reconstruction of a sinusoid is a circle: a one-dimensional set."""
    estimate = mle.estimate(sine(), ANALYTIC, seed=0)
    assert estimate.MG == pytest.approx(1.0, abs=0.15)
    assert estimate.LB == pytest.approx(1.0, abs=0.2)
    assert not estimate.degenerate


def test_a_two_frequency_torus_measures_about_two():
    estimate = mle.estimate(torus(), ANALYTIC, seed=0)
    assert estimate.MG == pytest.approx(2.0, abs=0.25)
    assert not estimate.degenerate


def test_the_linear_null_separates_the_two_cases():
    """The companion has to agree, or the estimate is measuring something else."""
    assert companions.delay_participation_ratio(
        embedding.reconstruct(sine(), ANALYTIC).points) == pytest.approx(2.0, abs=0.2)
    assert companions.delay_participation_ratio(
        embedding.reconstruct(torus(), ANALYTIC).points) > 3.0


# -- degeneracy ----------------------------------------------------------------

def test_a_constant_series_is_marked_degenerate():
    scored = windows.score(np.ones(8000), ANALYTIC)
    assert scored["degenerate"] is True
    assert np.isnan(scored["MG"]) and np.isnan(scored["LB"]) and np.isnan(scored["TwoNN"])


def test_an_exactly_recurrent_series_is_marked_degenerate():
    """The first of the two silent defects the archived report records.

    A sinusoid whose period divides the sampling interval exactly returns to the same delay
    vector every period, so the distances hit the 1e-8 floor and the per-point sums hit the
    1e-5 one. The value that comes back is ``(N(m-1) - 1) / (N * 1e-5)`` -- about two million
    here -- and the archived kernel returned it with nothing attached. It is returned here
    too, per appendix A, but the flag says what it is.
    """
    estimate = mle.estimate(np.sin(2 * np.pi * series() / 100.0), ANALYTIC, seed=0)
    assert estimate.degenerate
    assert estimate.floor_distance_fraction > 0.99
    assert estimate.floor_sum_fraction > 0.99
    assert estimate.MG > 1e5  # not NaN, and not clamped: the flag carries the warning


def test_the_estimate_is_never_clamped_to_the_embedding_dimension():
    """The second silent defect: the archived kernel returned ``max_E`` above ``2 * max_E``.

    A straight ramp never returns near itself, so the neighbour statistic has nothing to work
    with and diverges. Reporting that as ``max_E`` would make the transient regime of the
    article -- where the estimate reaches about 29 against a true dimension of 1 -- look like
    a measurement.
    """
    estimate = mle.estimate(series(), ANALYTIC, seed=0)
    assert estimate.MG > 3 * ANALYTIC.max_E


# -- the Theiler exclusion -----------------------------------------------------

def test_the_theiler_rule_takes_the_larger_of_the_span_and_the_autocorrelation_time():
    cfg = ANALYTIC.replace(theiler_cap=10 ** 9)
    span = (cfg.max_E - 1) * cfg.tau

    # An oscillation this fast has an autocorrelation time of a few samples, so the span wins.
    fast = sine(period=8.0)
    assert embedding.autocorrelation_time(fast) < span
    assert embedding.resolve_theiler(cfg, fast, cfg.tau) == span

    # A ramp never decorrelates, so the autocorrelation time wins by a wide margin.
    ramp = series()
    assert embedding.autocorrelation_time(ramp) > span
    assert embedding.resolve_theiler(cfg, ramp, cfg.tau) == embedding.autocorrelation_time(ramp)


def test_the_theiler_cap_binds_and_is_a_configuration_field():
    """The cap sets a published number, so it cannot be a module global.

    In the archived tree it was one, and three worker scripts assigned to it -- one of them
    for the life of the worker. Here it travels with the configuration, and two configurations
    differing only in the cap give two different exclusions in the same process.
    """
    ramp = series()
    capped = ANALYTIC  # theiler_cap defaults to 150
    uncapped = ANALYTIC.replace(theiler_cap=10 ** 9)
    assert embedding.resolve_theiler(capped, ramp, capped.tau) == 150
    assert embedding.resolve_theiler(uncapped, ramp, uncapped.tau) > 1000
    assert mle.estimate(ramp, capped).theiler == 150
    assert mle.estimate(ramp, uncapped).theiler > 1000


# -- refusals, and never a RuntimeWarning --------------------------------------

def test_a_window_holding_a_non_finite_sample_has_no_statistics():
    x = sine()
    x[17] = np.nan
    scored = windows.score(x, ANALYTIC)
    assert scored["degenerate"] is True
    assert all(np.isnan(scored[name]) for name in windows.statistic_names(ANALYTIC))


def test_a_window_too_short_to_embed_has_no_statistics():
    scored = windows.score(sine(60), ANALYTIC)
    assert all(np.isnan(scored[name]) for name in windows.statistic_names(ANALYTIC))


def test_a_record_shorter_than_one_window_summarises_to_nothing():
    """No windows, no statistics, and no warning: the archived version took the mean of an
    empty array here and printed a RuntimeWarning into a captured table."""
    summary = windows.summarise(sine(100), ANALYTIC)
    assert summary["n_windows"] == 0
    assert np.isnan(summary["MG"]) and np.isnan(summary["frac_degenerate"])


def test_non_finite_input_does_not_reach_the_autocorrelation_scan():
    x = sine()
    x[3] = np.inf
    assert embedding.autocorrelation_time(x) == 0
    assert np.isnan(companions.roughness(x))
    assert np.isnan(companions.spectral_participation_ratio(x, 256))


# -- determinism ---------------------------------------------------------------

def test_the_dither_is_seeded_and_the_estimate_is_reproducible():
    x = torus()
    assert mle.estimate(x, ANALYTIC, seed=7).MG == mle.estimate(x, ANALYTIC, seed=7).MG
    moved = abs(mle.estimate(x, ANALYTIC, seed=7).MG - mle.estimate(x, ANALYTIC, seed=8).MG)
    assert 0 < moved < 1e-3  # a different dither, not a different measurement


# -- the configuration and the frozen files ------------------------------------

def test_the_configuration_rejects_a_parameter_it_does_not_know():
    with pytest.raises(ValueError, match="k_neighbours"):
        EstimatorConfig.from_dict({"max_E": 10, "k_neighbours": 20})


def test_the_frozen_configurations_are_the_ones_the_article_states():
    eight, twenty = frozen.eight_direction(), frozen.twenty_direction()
    assert (eight.max_E, eight.tau, eight.window, eight.stride) == (20, 4, 8000, 2000)
    assert (twenty.max_E, twenty.tau, twenty.window, twenty.stride) == (40, 16, 8000, 4000)
    for cfg in (eight, twenty):
        assert cfg.k_neighbors == 20
        # Which of the two rules the grid selected is not pinned: appendix C reports that
        # they return bit-identical values on the calibration logs, so the argmin picks
        # between two copies of one measurement and either name is the same setting. The
        # eight-direction selection returned "autocorr" before the recalibration and
        # "embedding" after it, with no cell of the grid changing value.
        assert cfg.theiler in ("autocorr", "embedding")
        assert cfg.dither == 1e-9
        assert cfg.theiler_cap == 150


def test_the_two_theiler_rules_agree_where_the_record_is_not_oversampled():
    """What makes the eight-direction selection between them arbitrary, which the test above
    relies on. The autocorrelation rule only widens the exclusion where the autocorrelation
    time exceeds the embedding span, and appendix C reports one to three samples against a
    span of seventy-six on these logs."""
    eight = frozen.eight_direction()
    span = (eight.max_E - 1) * eight.tau
    rng = np.random.default_rng(0)
    series = rng.standard_normal(8000)          # autocorrelation time of one sample
    for rule in ("autocorr", "embedding"):
        assert embedding.resolve_theiler(eight.replace(theiler=rule), series,
                                         eight.tau) == min(span, eight.theiler_cap)


def test_the_window_geometry_overrides_are_the_ones_appendix_c_tabulates():
    eight = frozen.eight_direction()
    assert frozen.constructed_geometry(eight, 26_000).stride == 3000
    assert frozen.constructed_geometry(eight, 30_000).stride == 3666
    training = frozen.training_log_geometry(eight, 12_000)
    assert (training.window, training.stride) == (4000, 1000)
    # An override moves the window geometry and nothing else.
    assert training.replace(window=eight.window, stride=eight.stride) == eight


def test_the_stored_calibration_is_rebuilt_and_not_refitted():
    cal = frozen.calibration("c_norm")
    assert cal.fitted and cal.n_points > 10
    assert cal.predict(1.0) == pytest.approx(1.9999455188206496)   # clipped below the knots
    assert cal.predict(1000.0) == pytest.approx(5.999806638694667)  # and above them
    values = cal.predict(np.array([2.0, 3.0, 4.0, np.nan]))
    assert np.all(np.diff(values[:3]) >= 0)
    assert np.isnan(values[3])


def test_a_calibration_refuses_to_predict_before_it_is_fitted():
    with pytest.raises(RuntimeError):
        Calibration("affine").predict(3.0)


def test_an_affine_calibration_recovers_a_linear_relation():
    estimates = np.array([1.0, 2.0, 3.0, 4.0])
    cal = Calibration("affine").fit(estimates, 2 * estimates + 1)
    assert cal.predict(5.0) == pytest.approx(11.0)


# -- surrogates ----------------------------------------------------------------

def test_a_surrogate_keeps_the_values_and_the_spectrum():
    x = torus()
    made = surrogates.iaaft(x, iters=20, rng=np.random.default_rng(0), match=False)
    assert np.allclose(np.sort(made), np.sort(x))  # the amplitude distribution, exactly
    power = lambda y: np.abs(np.fft.rfft(y - y.mean())) ** 2  # noqa: E731
    kept = power(made).sum() / power(x).sum()
    assert kept == pytest.approx(1.0, abs=0.05)
    assert not np.allclose(made, x)  # and the phases are gone


def test_endpoint_matching_trims_the_series_and_is_the_default():
    x = torus()
    matched = surrogates.iaaft(x, iters=5, rng=np.random.default_rng(0))
    whole = surrogates.iaaft(x, iters=5, rng=np.random.default_rng(0), match=False)
    assert len(matched) < len(x) == len(whole)
    assert len(surrogates.match_endpoints(x)) >= int(0.85 * len(x))


def test_the_iteration_count_is_an_argument():
    x = torus()
    few = surrogates.iaaft(x, iters=2, rng=np.random.default_rng(1))
    many = surrogates.iaaft(x, iters=100, rng=np.random.default_rng(1))
    assert not np.allclose(few, many)
