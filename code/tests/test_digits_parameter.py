"""The parameter-subspace system: three ranks that are not the same number.

The module exists to keep *available*, *functional* and *active* apart, and the article's
two strongest claims about it are that the second is measured rather than assumed and that
the third can be held at ``k`` only because the drive is equalised. Both are checked here,
along with the two controls that make the distinction visible: the preconditioner, without
which rank-``k`` forcing does not give a rank-``k`` trajectory, and the zero learning rate,
which says whether an observer is reading the optimiser or the drive.

Every test runs at the module's default configuration, which is deliberately small: the
experiments set the real geometry, as they do for every other system.
"""
from __future__ import annotations

import numpy as np
import pytest

from actdim.systems import digits_parameter as dp
from actdim.systems import drive, spec


@pytest.fixture(scope="module")
def recurrent():
    return dp.simulate(dp.DigitsParameterConfig(k=4), seed=0)


# -- the three ranks -------------------------------------------------------------------

def test_the_three_ranks_are_different_numbers(recurrent):
    """Available, functional and active. The package exists to keep them apart."""
    measured = recurrent.truth.measured
    assert measured["functional_rank"] == 10.0, "the functional rank is the available one"
    assert measured["trajectory_rank"] > 4.0, (
        "the hard rank of the trajectory is generically the number of available "
        "directions, whatever is forced -- that is the Krylov argument the module makes, "
        "and if it stops holding the ground-truth check here is checking the wrong thing")
    assert measured["trajectory_effective_rank"] == pytest.approx(4.0, abs=0.05)


def test_the_equalised_drive_excites_every_direction_comparably(recurrent):
    """Requirement 1's word *comparably*, which is what equalise_gains delivers.

    Appendix F tabulates a measured effective rank of exactly ``r`` for this arm. The
    mixing matrix is what makes that true: the forcing directions of ``r`` data groups are
    neither orthogonal nor of equal effect, and unmixed forcing gives an effective rank far
    below ``r``.
    """
    assert recurrent.truth.verified, recurrent.truth.failures()
    assert recurrent.truth.measured["direction_ratio"] > 0.8, (
        "the weakest of the four directions should be within a fifth of the strongest")


def test_without_the_preconditioner_the_trajectory_spreads():
    """The available-versus-active distinction, made by measurement rather than by claim.

    The stationary covariance of ``c_{t+1} = c_t - eta(H c_t + xi_t)`` is supported on the
    Krylov space of ``I - eta H`` over the forcing subspace, which is generically the whole
    of ``R^available``. Preconditioning by ``H^-1`` makes the dynamics isotropic and only
    then does rank-``k`` forcing give a rank-``k`` trajectory.
    """
    with_it = dp.simulate(dp.DigitsParameterConfig(k=4, mode="noise", drive_amp=0.0,
                                                   noise_amp=0.08), seed=0)
    without = dp.simulate(dp.DigitsParameterConfig(k=4, mode="noise", drive_amp=0.0,
                                                   noise_amp=0.08, precondition=False),
                          seed=0)
    assert with_it.truth.measured["effective_rank"] == pytest.approx(4.0, abs=0.1)
    assert without.truth.measured["effective_rank"] < with_it.truth.measured["effective_rank"]


# -- the silence control ---------------------------------------------------------------

def test_a_zero_learning_rate_freezes_every_state_only_observer(recurrent):
    """Requirement 4. Eleven of the twelve observers are functions of the optimiser state.

    With the learning rate at zero the coordinate never moves, so those eleven are
    constant to rounding. ``loss_step`` contains the instantaneous drive weights, so it
    keeps varying -- which is why the article keeps it in the panel: the contamination is
    meant to be visible rather than assumed away.
    """
    from actdim.observers import STATE_ONLY

    silent = dp.simulate(dp.DigitsParameterConfig(k=4, eta_zero=True), seed=0)
    assert silent.truth.active_dimension == 1.0

    for name in STATE_ONLY:
        ratio = float(silent[name].std()) / max(float(recurrent[name].std()), 1e-300)
        assert ratio < 1e-6, f"{name} still moves at a zero learning rate: ratio {ratio:g}"
    loud = float(silent["loss_step"].std()) / float(recurrent["loss_step"].std())
    assert loud > 1e-3, (
        "loss_step reads the drive directly and must survive the control; if it stops "
        "doing so the panel no longer shows the contamination it was kept to show")


# -- the modes -------------------------------------------------------------------------

def test_a_transient_has_active_dimension_one_whatever_the_rank():
    """A converging trajectory is a curve, and a curve is one-dimensional for every k."""
    for k in (2, 6):
        config = dp.DigitsParameterConfig(k=k, mode="gd", drive_amp=0.0, noise_amp=0.0,
                                          eta=0.006, precondition=False, burn=0,
                                          displacement=1.0)
        assert config.active_dimension == 1.0
        result = dp.simulate(config, seed=0)
        assert result.truth.checks == {}, (
            "a mode that claims no rank has nothing to confirm, and asserting one would "
            "be asserting something the construction does not say")
        assert np.isfinite(result.truth.measured["trajectory_effective_rank"])


def test_plain_mini_batch_descent_claims_no_rank():
    """Its noise rank is a property of the data, not a parameter of the experiment."""
    config = dp.DigitsParameterConfig(k=4, mode="batch", drive_amp=0.0, batch=64)
    assert np.isnan(config.active_dimension)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        dp.simulate(dp.DigitsParameterConfig(k=2, mode="whatever"), seed=0)


# -- the drive -------------------------------------------------------------------------

def test_the_drive_is_the_shared_one_and_stays_far_from_a_resonance(recurrent):
    """No second prime table, no second margin.

    The archived system module carried its own copy of both, with a different table and a
    different order, and the frequencies it produced ignored the seed. Rational
    independence is what makes the orbit close onto a torus, so the margin is reported with
    every run rather than checked once.
    """
    margin = recurrent.truth.measured["resonance_margin"]
    assert margin > 1e-3, f"the drive is close to a resonance: margin {margin:g}"
    assert recurrent.info["realised_band"] == pytest.approx(2.0, rel=0.15)
    # The frequencies come from the shared module and vary with the seed, which is the
    # correction of errata item 1.
    centre = drive.centre_for_octave(dp.F_FAST, 2.0)
    assert not np.allclose(drive.frequencies(4, centre, seed=0),
                           drive.frequencies(4, centre, seed=1))


# -- reproducibility -------------------------------------------------------------------

def test_two_runs_of_one_seed_are_bit_identical():
    """Every stream is derived from the base seed by a named rule, so a re-run repeats."""
    config = dp.DigitsParameterConfig(k=2)
    first = dp.simulate(config, seed=3)
    second = dp.simulate(config, seed=3)
    for name in first.series:
        assert np.array_equal(first[name], second[name]), name


def test_the_system_is_registered_under_its_ladder_name():
    entry = spec.get("digits_parameter")
    assert entry.config is dp.DigitsParameterConfig
    assert "digits_parameter" not in spec.NOT_PORTED
    assert set(spec.load()) == set(spec.LADDER)


def test_the_two_published_configurations_differ_only_where_the_article_says():
    ten, twenty = dp.ten_direction(), dp.twenty_direction()
    assert (ten.available, twenty.available) == (10, 20)
    assert ten.groups == 12 and twenty.groups == 24
    assert ten.length == 30_000
    assert ten.mode == twenty.mode == "qp"


# -- the schedules ---------------------------------------------------------------------

def test_a_rank_schedule_changes_the_number_of_excited_directions():
    """The change-detection experiment of appendix E rests on this and nothing else."""
    config = dp.DigitsParameterConfig(k=6, window=4000, burn=500)
    schedule = np.full(config.length, 6, dtype=int)
    schedule[config.length // 2:] = 2
    _, coordinates, _, _, _, _ = dp.trajectory(config, 0, dp.Schedules(rank=schedule))

    from actdim.linalg import TRAJECTORY_RANK_TOL, rank_report

    half = len(coordinates) // 2
    high = rank_report(coordinates[:half], tol=TRAJECTORY_RANK_TOL).effective_rank
    low = rank_report(coordinates[half:], tol=TRAJECTORY_RANK_TOL).effective_rank
    assert high > low + 2.0, f"the schedule did not lower the rank: {high:.2f} -> {low:.2f}"


def test_an_observer_gain_scales_the_fluctuation_and_not_the_mean():
    """A ramp on the raw series would inject a trend, not a gain.

    For the loss, the parameter norm and the accuracy the mean is orders of magnitude
    larger than the fluctuation, so scaling the series itself would be read as the estimate
    failing to be scale invariant when what changed was a dominant trend.
    """
    config = dp.DigitsParameterConfig(k=2, window=2000, burn=200)
    plain = dp.simulate(config, seed=0)
    scaled = dp.simulate(config, seed=0,
                         schedules=dp.Schedules(observer_gain=np.full(config.length, 3.0)))
    for name in ("w_fro", "loss_full"):
        assert scaled[name].mean() == pytest.approx(plain[name].mean(), rel=1e-9)
        assert scaled[name].std() == pytest.approx(3.0 * plain[name].std(), rel=1e-9)
