"""Section 6's experiments: the pieces of them that a run must not be trusted to get right.

Three of these tests exist because the port had to reimplement something the estimator
already does, and one because it had to reimplement something the archived tree did four
times. In each case the duplicate is pinned against the original rather than believed.

The rest check the two rules the section's conclusions rest on: that the exclusion is a
field of the configuration and not a global, so the capped and the uncapped rule can be
measured side by side; and that two embedding dimensions of one cell see the same series,
which the archived seeding did not give them and without which the identifiability ratio
is not the quantity it is described as.
"""
from __future__ import annotations

import numpy as np
import pytest

from actdim import frozen
from actdim.estimator import mle, windows
from actdim.estimator.config import EstimatorConfig
from actdim.experiments import validity
from actdim.systems import synthetic


@pytest.fixture(scope="module")
def torus():
    """A four-torus, long enough for one frozen window."""
    state, _ = synthetic.quasiperiodic(4, 9000, seed=0, f0=1 / 16.0)
    return synthetic.observe(state, 0, "generic")


@pytest.fixture(scope="module")
def transient():
    """A decaying exponential: a curve, never revisited, of intrinsic dimension one."""
    t = np.arange(9000, dtype=float)
    return np.exp(-t / 3000.0)


# -- the duplicated pooling ------------------------------------------------------------

CONFIG = EstimatorConfig(max_E=20, tau=4, k_neighbors=20, theiler="autocorr",
                         window=8000, stride=8000, dither=1e-9)


@pytest.mark.parametrize("exclusion", [0, 5, 20, 150])
def test_the_shared_query_reproduces_the_estimator_exactly(torus, exclusion):
    """The exclusion sweep shares one neighbour query across eleven settings.

    That is a second implementation of the pooled estimate, made necessary because the
    estimator owns its own query and eleven of them per window would cost eleven times as
    much. The archived experiment asserted the two were equal; this proves it.
    """
    window = torus[:CONFIG.window]
    cells = {cell["theiler_label"]: cell
             for cell in validity._exclusion_cells(window, CONFIG, 0, (exclusion,))}
    canonical = mle.estimate(window, CONFIG.replace(theiler=exclusion,
                                                    theiler_cap=validity.UNCAPPED), 0)
    shared = cells[str(exclusion)]
    assert shared["theiler_used"] == canonical.theiler
    assert shared["MG"] == pytest.approx(canonical.MG, rel=1e-12)
    assert shared["LB"] == pytest.approx(canonical.LB, rel=1e-12)
    assert bool(shared["degenerate"]) == bool(canonical.degenerate)


def test_the_shared_query_reproduces_the_frozen_rule(torus):
    """The ``frozen`` label is the configuration's own rule, cap included."""
    window = torus[:CONFIG.window]
    cells = {cell["theiler_label"]: cell
             for cell in validity._exclusion_cells(window, CONFIG, 0, ("frozen",))}
    canonical = mle.estimate(window, CONFIG, 0)
    assert cells["frozen"]["theiler_used"] == canonical.theiler
    assert cells["frozen"]["MG"] == pytest.approx(canonical.MG, rel=1e-12)


# -- the cap ---------------------------------------------------------------------------

def test_the_cap_binds_on_a_transient_and_is_a_field_not_a_global(transient):
    """Errata item 2. The published value near 29 is the value at the cap.

    The autocorrelation of a monotone decay does not fall away, so the rule asks for an
    exclusion of order a thousand samples and the implementation's cap of 150 clips it.
    Both are settings of the configuration object here, so the sweep can report them side
    by side; in the archived tree the cap was a module global that three workers assigned
    to, one of them permanently.
    """
    window = transient[:CONFIG.window]
    cells = {cell["theiler_label"]: cell
             for cell in validity._exclusion_cells(window, CONFIG, 0,
                                                   (0, 150, "frozen", "uncapped"))}
    assert cells["frozen"]["theiler_used"] == CONFIG.theiler_cap
    assert cells["uncapped"]["theiler_used"] > 5 * CONFIG.theiler_cap, (
        "the uncapped rule should ask for far more than the cap allows on a transient")
    assert cells["0"]["MG"] == pytest.approx(1.2, abs=0.1), (
        "with no exclusion the estimator returns what maximum likelihood must return on "
        "a uniformly sampled curve")
    assert cells["150"]["MG"] > 10.0, "the exclusion is what makes the estimate diverge"
    assert cells["uncapped"]["MG"] > cells["150"]["MG"]


def test_the_recurrent_case_is_unmoved_by_the_same_variation(torus):
    """The other half of the two-by-two: what the exclusion costs a recurrent record."""
    window = torus[:CONFIG.window]
    cells = {cell["theiler_label"]: cell
             for cell in validity._exclusion_cells(window, CONFIG, 0, (0, 150, "frozen"))}
    assert cells["150"]["MG"] == pytest.approx(cells["0"]["MG"], rel=0.15)


# -- the synthetic families ------------------------------------------------------------

def test_both_embedding_dimensions_of_one_cell_see_the_same_series():
    """The identifiability ratio compares two embeddings, not two realisations.

    In the archived atlas the embedding dimension entered the seed of the generator and of
    the observer, so the two halves of the ratio were computed on different data through
    different observers. Nothing about the series here depends on ``max_E``.
    """
    state, _ = synthetic.quasiperiodic(3, 2000, seed=1)
    again, _ = synthetic.quasiperiodic(3, 2000, seed=1)
    assert np.array_equal(state, again)
    assert np.array_equal(synthetic.observe(state, 1, "generic"),
                          synthetic.observe(again, 1, "generic"))


def test_the_torus_has_the_rank_it_is_built_with():
    for rank in (1, 3, 6):
        state, meta = synthetic.quasiperiodic(rank, 20_000, seed=0)
        hard, effective = synthetic.state_rank(state)
        assert hard == rank
        assert effective == pytest.approx(rank, abs=0.05)
        assert meta["margin"] > 0.0, "an exact resonance would collapse the torus"
        # One octave at every rank above one, which is requirement 6: an added direction
        # must not also add a higher frequency, or the roughness alone orders the rank.
        # A single oscillator spans no band at all.
        assert meta["realised_band"] == pytest.approx(2.0 if rank > 1 else 1.0, rel=0.2)


def test_the_stochastic_families_are_full_rank_and_differently_smooth():
    """The two are the same rank and not the same geometry, which is why both are run."""
    ou, meta = synthetic.ornstein_uhlenbeck(4, 20_000, seed=0, tau_c=200.0)
    coloured, _ = synthetic.coloured(4, 20_000, seed=0, tau_c=200.0, order=3)
    assert synthetic.state_rank(ou)[0] == 4
    assert synthetic.state_rank(coloured)[0] == 4
    assert 0.0 < meta["innov_ratio"] < 1.0
    # The coloured cascade is smooth on the scale of tau_c and the OU process is not; the
    # roughness null is what section 6.3 tests against, so the two must differ in it.
    from actdim.estimator.companions import roughness

    assert roughness(coloured[:, 0]) < 0.2 * roughness(ou[:, 0])


def test_an_unknown_family_or_observer_is_refused():
    with pytest.raises(ValueError):
        synthetic.generate("brownian", 2, 100, 0)
    with pytest.raises(ValueError):
        synthetic.observe(np.zeros((100, 2)), 0, "whatever")


# -- the shared scoring pieces ---------------------------------------------------------

def test_the_closed_form_slope_matches_a_least_squares_fit():
    rng = np.random.default_rng(0)
    x = np.arange(20.0)
    y = 2.5 * x - 1.0 + rng.standard_normal(20)
    assert validity._slope(x, y) == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-12)


def test_the_closed_form_slope_survives_a_nan():
    """``np.polyfit`` raises here, from inside a worker, which is errata item 29."""
    x = np.arange(10.0)
    y = x.copy()
    y[3] = np.nan
    assert validity._slope(x, y) == pytest.approx(1.0, rel=1e-12)


def test_a_median_of_nothing_is_nan_rather_than_a_warning():
    assert np.isnan(validity._median([]))
    assert np.isnan(validity._median([np.nan, np.nan]))
    assert validity._median([1.0, np.nan, 3.0]) == 2.0


def test_the_crossing_point_reports_a_ceiling_above_the_grid_as_infinite():
    ranks = [2, 4, 6, 8]
    assert validity._crossing_point(ranks, ranks, 1.0) == float("inf")
    below = [2.0, 4.0, 5.0, 5.0]
    crossing = validity._crossing_point(ranks, below, 1.0)
    assert 4.0 < crossing < 8.0


# -- the window geometry ---------------------------------------------------------------

def test_the_constructed_stride_is_the_one_appendix_c_states():
    """One stride rule. The archived tree had it copied into two scripts and re-derived
    by eye in a third, and the article's table 9 gives its two values."""
    cfg = frozen.eight_direction()
    assert frozen.constructed_geometry(cfg, 26_000).stride == 3000
    assert frozen.constructed_geometry(cfg, 30_000).stride == 3666
    assert len(windows.window_starts(26_000, frozen.constructed_geometry(cfg, 26_000))) == 7


# -- the silence verdict ---------------------------------------------------------------

def _silence_rows(mg_silent, observer="loss"):
    return [{"system": "toy", "k": k, "seed": 0, "observer": observer,
             "applicable": True, "truth": float(k), "sd_trained": 1.0, "sd_silent": 0.5,
             "sd_ratio": 0.5, "moves_when_silent": True, "series_correlation": 0.99,
             "MG_trained": float(k), "MG_silent": value}
            for k, value in zip((2, 4, 6, 8), mg_silent)]


def test_an_observer_that_still_orders_the_ranks_in_silence_fails_requirement_4():
    import pandas as pd

    verdict = validity._silence_verdict(pd.DataFrame(_silence_rows([2.1, 4.0, 5.8, 7.9])))
    assert bool(verdict.iloc[0]["survives_silence"])
    assert bool(verdict.iloc[0]["fails_requirement_4"])
    assert bool(verdict.iloc[0]["system_invalidated"])


def test_an_observer_that_stops_moving_passes_it():
    import pandas as pd

    rows = _silence_rows([np.nan] * 4)
    for row in rows:
        row.update(moves_when_silent=False, sd_silent=1e-18, sd_ratio=1e-18)
    verdict = validity._silence_verdict(pd.DataFrame(rows))
    assert not bool(verdict.iloc[0]["survives_silence"])
    assert not bool(verdict.iloc[0]["fails_requirement_4"])


def test_the_one_observer_designed_to_survive_does_not_condemn_its_system():
    """``loss_step`` contains the instantaneous drive weights and is not claimed to be a
    function of the optimiser state. It is in the panel so that the contamination is
    visible; counting it as a failure would condemn the one system that passes."""
    import pandas as pd

    verdict = validity._silence_verdict(
        pd.DataFrame(_silence_rows([2.1, 4.0, 5.8, 7.9], observer="loss_step")))
    assert not bool(verdict.iloc[0]["claims_state_only"])
    assert bool(verdict.iloc[0]["survives_silence"])
    assert not bool(verdict.iloc[0]["fails_requirement_4"])
    assert not bool(verdict.iloc[0]["system_invalidated"])


def test_a_system_with_no_learning_rate_has_no_control_to_run():
    """The oscillating diagonal matrix has no optimiser, which is why section 5.1 says a
    negative result there would be decisive -- and why requirement 4 cannot be put to it."""
    assert validity._silence_config("matrix", 2, 4000, silent=True) is None
    assert validity._silence_config("subspace", 2, 4000, silent=True).eta == 0.0
    assert validity._silence_config("digits_parameter", 2, 4000, silent=True).eta_zero


# -- the experiments themselves --------------------------------------------------------

def test_every_section_six_experiment_declares_what_the_article_reads():
    from actdim.runtime import registry
    from actdim.runtime.archive import BASELINE

    registry.load()
    for name, entry in registry.REGISTRY.items():
        if not (name.startswith("valid.") or name == "sys.digits.parameter"):
            continue
        archived = set(BASELINE.get(name, {}))
        promoted = set(entry.promotes)
        assert archived <= promoted, (
            f"{name} does not promote {sorted(archived - promoted)}, which "
            f"actdim.runtime.archive maps to an archived file and actdim diff compares")
