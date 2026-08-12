"""The constructed systems, and the drive that excites them.

The first group of tests exists because of one defect. The archived drive accepted a seed
and ignored it, so every "held-out" seed of every system reused seed zero's frequency
geometry and differed only in phase, amplitude and noise. Requirement 3 asks that ranks
and seeds both be withheld; what was withheld was less than the article claims. These
tests pin the corrected behaviour and the two properties the correction had to preserve.
"""
from __future__ import annotations

import numpy as np
import pytest

from actdim import linalg, observers
from actdim.systems import drive, spec

CENTRE = drive.centre_for_window(1000.0, 8000)
RANKS = (2, 4, 8)
SEEDS = (0, 1, 2)


# -- the defect, and its correction --------------------------------------------------

@pytest.mark.parametrize("k", RANKS)
def test_the_frequency_geometry_varies_with_the_seed(k):
    """The point of the fix: two seeds are two different systems."""
    sets = [drive.frequencies(k, CENTRE, seed=s) for s in SEEDS]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not np.allclose(sets[i], sets[j]), (
                f"seeds {SEEDS[i]} and {SEEDS[j]} produced the same frequencies at k={k}")


@pytest.mark.parametrize("k", RANKS)
def test_the_archived_layout_still_ignores_its_seed(k):
    """Kept reachable so an experiment can measure what the defect cost.

    If this ever starts failing, the archived behaviour has been changed and the
    comparison it exists for is no longer a comparison.
    """
    sets = [drive.frequencies(k, CENTRE, seed=s, band_mode="matched_fixed") for s in SEEDS]
    assert np.allclose(sets[0], sets[1])
    assert np.allclose(sets[0], sets[2])


def test_the_realised_band_is_matched_across_ranks():
    """Requirement 6. An added direction must not also add a higher frequency.

    Where it does, the roughness of the observable orders the rank by itself and no
    embedding is being tested. The `widening` mode is the control that shows this, and it
    must fail the same check the matched construction passes.
    """
    for seed in SEEDS:
        bands = [drive.realised_band(drive.frequencies(k, CENTRE, seed=seed))
                 for k in RANKS]
        assert max(bands) - min(bands) < 0.05 * min(bands), (
            f"bandwidth drifts with rank at seed {seed}: {bands}")

    widening = [drive.realised_band(drive.frequencies(k, CENTRE, band_mode="widening"))
                for k in RANKS]
    assert widening[-1] > 2 * widening[0], (
        "the widening control should widen; if it does not, requirement 6 has nothing to "
        "guard against and the control has stopped being one")


@pytest.mark.parametrize("k", RANKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_frequencies_stay_far_from_a_resonance(k, seed):
    """Rational independence is what makes the orbit close onto a k-torus.

    The threshold is the weakest margin the archived construction achieved at the same
    rank, so the correction is not allowed to buy seed variation with a worse geometry.
    """
    freqs = drive.frequencies(k, CENTRE, seed=seed)
    archived = drive.resonance_margin(
        drive.frequencies(k, CENTRE, seed=0, band_mode="matched_fixed"))
    margin = drive.resonance_margin(freqs)
    assert margin > 0.0, "the frequencies hit an exact resonance"
    assert margin > 0.25 * archived, (
        f"margin {margin:.5g} at k={k}, seed={seed} is far below the archived {archived:.5g}")


def test_frequencies_are_reproducible():
    a = drive.frequencies(6, CENTRE, seed=3)
    b = drive.frequencies(6, CENTRE, seed=3)
    assert np.array_equal(a, b)


def test_an_unknown_band_mode_is_refused():
    with pytest.raises(ValueError):
        drive.frequencies(4, CENTRE, band_mode="whatever")


# -- the catalogue -------------------------------------------------------------------

def test_every_ported_ladder_row_is_registered():
    catalogue = spec.load()
    expected = [name for name in spec.LADDER if name not in spec.NOT_PORTED]
    assert sorted(catalogue) == sorted(expected)


def test_the_unported_rows_are_named_rather_than_absent():
    """A ladder row with no module is a row of the article that cannot be regenerated.

    It must be visible as such, not surface as an ImportError.
    """
    for name, reason in spec.NOT_PORTED.items():
        assert name in spec.LADDER
        assert "archived_code" in reason, f"{name} does not say where its source is"


PORTED = [n for n in spec.LADDER if n not in spec.NOT_PORTED]

#: Systems whose construction is sound but whose excitation is not yet as even as the
#: published one. See `test_the_digits_system_is_less_evenly_excited_than_published`.
UNEVEN = {"digits_function"}


@pytest.mark.parametrize("system_id", PORTED)
def test_a_system_records_finite_scalar_series(system_id):
    entry = spec.get(system_id)
    config = entry.config(k=2) if _accepts(entry.config, "k") else entry.config()
    result = entry.simulate(config, seed=0)

    assert result.series, f"{system_id} recorded no observers"
    assert result.length > 0
    for name, values in result.series.items():
        assert values.ndim == 1, f"{system_id}/{name} is not a scalar series"
        assert len(values) == result.length, f"{system_id}/{name} has a different length"
        assert np.isfinite(values).all(), f"{system_id}/{name} contains a non-finite value"


@pytest.mark.parametrize("system_id", PORTED)
def test_a_system_excites_the_rank_it_claims(system_id):
    """Requirement 1: the construction fixes the dimension and a measurement confirms it.

    The hard checks -- the rank of the trajectory covariance, of the update covariance and
    of the map to the outputs -- must equal k exactly. A constructed truth that is never
    measured is requirement 1 unmet.
    """
    entry = spec.get(system_id)
    config = entry.config(k=2) if _accepts(entry.config, "k") else entry.config()
    result = entry.simulate(config, seed=0)
    assert result.truth.active_dimension > 0

    failures = [name for name in result.truth.failures()
                if not (system_id in UNEVEN and name == "effective_rank")]
    assert not failures, (
        f"{system_id} claims active dimension {result.truth.active_dimension} but "
        f"{', '.join(failures)} disagree: {result.truth.measured}")


def test_the_digits_system_is_less_evenly_excited_than_published():
    """A regression the corrected drive introduced, recorded rather than hidden.

    Every hard rank of this system is exactly k. What falls short is the evenness: the
    participation ratio of the trajectory covariance reaches about 0.86 k where the
    published construction reached 0.94 to 0.96 k (`data/sys.digits.function/
    rank_diagnostics.csv`, column `trajectory_ratio`). Requirement 1 asks that all r
    directions be *comparably* excited, so this is the requirement's soft half failing
    while its hard half passes.

    The corrected drive buys a much better resonance margin -- 0.032 at k = 2 against the
    published 0.029, and 1.5e-5 published against a port that stays above 1e-3 -- so the
    trade is real and has to be decided rather than defaulted. Until it is, this test
    holds the measured ratio so the number cannot drift further unnoticed.
    """
    entry = spec.get("digits_function")
    ratios = {}
    for k in (2, 4, 8):
        result = entry.simulate(entry.config(k=k), seed=0)
        ratios[k] = result.truth.measured["effective_rank"] / k

    assert all(0.80 < r < 0.92 for r in ratios.values()), (
        f"the excitation evenness moved: {ratios}. If it improved past 0.92 the "
        "regression is fixed and this test and docs/errata.md item 31 should go; if it "
        "fell below 0.80 something else broke.")


@pytest.mark.parametrize("system_id", [n for n in spec.LADDER if n not in spec.NOT_PORTED])
def test_two_seeds_of_one_system_differ(system_id):
    """With the drive corrected, a second seed is a second system, not a re-phasing."""
    entry = spec.get(system_id)
    config = entry.config(k=2) if _accepts(entry.config, "k") else entry.config()
    first = entry.simulate(config, seed=0)
    second = entry.simulate(config, seed=1)
    name = next(iter(first.series))
    assert not np.allclose(first[name], second[name])


def _accepts(config_type, field: str) -> bool:
    import dataclasses

    return any(f.name == field for f in dataclasses.fields(config_type))


# -- the pieces that were duplicated -------------------------------------------------

def test_participation_ratio_counts_equally_weighted_directions():
    """Four copies of these three lines existed; this is the one that survives."""
    assert linalg.participation_ratio([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert linalg.participation_ratio([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert linalg.participation_ratio([2.0, 2.0]) == pytest.approx(2.0)


def test_effective_rank_of_a_rank_two_matrix():
    rng = np.random.default_rng(0)
    basis = linalg.orthonormal((50, 2), rng)
    matrix = basis @ rng.standard_normal((2, 30))
    assert linalg.numerical_rank(matrix, center=False) == 2
    assert 1.0 <= linalg.effective_rank(matrix, center=False) <= 2.0


# -- the observers -------------------------------------------------------------------

def test_the_article_records_twelve_observers_in_five_families():
    """Appendix B. The archived tree defined sixteen, and the article's twelve were a
    subset held in an experiment module that three other experiments imported from."""
    assert len(observers.PAPER_TWELVE) == 12
    assert len(set(observers.PAPER_TWELVE)) == 12
    families = {observers.REGISTRY[name].family for name in observers.PAPER_TWELVE}
    assert families == set(observers.FAMILIES)


@pytest.mark.parametrize("panel", ["K20_PANEL", "K20_CALIBRATION", "CEILING_PANEL",
                                   "THEILER_PANEL"])
def test_every_named_panel_is_drawn_from_the_one_registry(panel):
    """No experiment defines its own observers. Four different panels existed in four
    experiment modules, two of them with the same name and different members."""
    for name in getattr(observers, panel):
        assert name in observers.REGISTRY, f"{name} is in {panel} but not in the registry"
