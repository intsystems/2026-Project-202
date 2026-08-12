"""Appendix M: the closed-form representation, and what lies outside it.

Nothing here trains. The decomposition tests run at small moduli, where the complete
search finishes in milliseconds and still decides every case; the reference values are
checked at the article's own ``p = 97`` and width 500, which costs one SVD.
"""
from __future__ import annotations

import numpy as np
import pytest

from actdim.analysis import representation as R
from actdim.models.perceptron import fourier_ipr
from actdim.tasks import polynomials

# -- is there a decomposition at all? -------------------------------------------

@pytest.mark.parametrize("p", [11, 13])
@pytest.mark.parametrize("name", ["p1", "p2", "p3"])
def test_the_learnable_polynomials_decompose(name, p):
    """Exhibiting one ``(g1, g2, h)`` settles existence, and the search agrees."""
    result = R.decompose(name, p, budget=200_000)
    assert result.exists is True
    assert result.verdict == "exists"
    assert result.verified, "the known decomposition must check out on all p^2 entries"
    assert result.searched is True, "the complete search must agree with it"
    assert result.certificate is None


@pytest.mark.parametrize("p", [11, 13])
@pytest.mark.parametrize("name", ["p1x", "p2x", "p3x"])
def test_the_perturbed_polynomials_do_not_decompose(name, p):
    """The claim the article's ``provably outside the class'' rests on.

    Two independent arguments, and both must agree: the multiset certificate, which is
    one-sided and cheap, and the complete backtracking search, whose ``False`` is a
    proof of non-existence rather than a failure to find one.
    """
    result = R.decompose(name, p, budget=200_000)
    assert result.exists is False
    assert result.verdict == "none"
    assert not result.verified
    assert result.certificate is not None
    assert result.searched is False
    assert result.nodes > 0


def test_the_certificate_alone_decides_at_the_articles_modulus():
    """At p = 23 the cheap argument settles all three, so the search is optional."""
    for name in polynomials.PERTURBED:
        result = R.decompose(name, 23, budget=0)
        assert result.exists is False
        assert "multiset certificate" in result.reasons[0]
        assert result.searched is None and result.nodes == 0


def test_a_verdict_is_about_the_table_at_that_modulus():
    """At p = 5 one perturbed polynomial really does decompose, and is reported so.

    The routine decides the table in front of it. A modulus that small collapses the
    perturbation into the base polynomial, which is a fact about p = 5 and not a bug;
    appendix O runs at 97 and the archived campaign checked 23.
    """
    assert R.decompose("p3x", 5, budget=10_000).exists is True
    assert R.decompose("p3x", 7, budget=10_000).exists is False


def test_decompose_all_covers_the_six_rows_and_summarises_flat():
    results = R.decompose_all(11, budget=100_000)
    assert list(results) == list(polynomials.POLYNOMIALS)
    assert [r.exists for r in results.values()] == [True, False, True, False,
                                                    True, False]
    summary = results["p1x"].summary()
    assert summary["verdict"] == "none"
    assert summary["rows"] == summary["cols"] == 11
    assert "certificate" in summary and "complete search" in summary["reasons"]


def test_the_reduced_table_keeps_the_question_it_was_asked():
    """Deduplicating rows and columns is what the search prunes with."""
    table = polynomials.table("p1", 11)
    reduced = R._reduce(table)
    assert reduced.shape[0] <= table.shape[0]
    assert len({tuple(r) for r in reduced}) == reduced.shape[0]


# -- the closed-form weights -----------------------------------------------------

CLOSED_FORM_TASKS = ["add", "sub", "sq_sum", "sum_sq", "p1", "p2", "p3"]


@pytest.mark.parametrize("task", CLOSED_FORM_TASKS)
def test_the_closed_form_solves_its_task(task):
    p, width = 11, 128
    w1, w2 = R.closed_form(task, p, width)
    scored = R.score_weights(w1, w2, p, task)
    assert scored["acc"] == pytest.approx(1.0, abs=0.02)
    # The amplitude is fixed by the construction rather than free, so the mean output at
    # the correct class is the reading that catches a wrong constant in the forward
    # pass. A training run cannot catch that: it would still produce a curve.
    assert scored["mean_peak"] == pytest.approx(1.0, abs=0.25)


def test_the_amplitude_and_the_norm_are_the_ones_appendix_m_quotes():
    assert R.amplitude(97) == pytest.approx(7.29, abs=0.01)
    w1, w2 = R.closed_form("add", 97, 500)
    scored = R.score_weights(w1, w2, 97, "add")
    assert scored["acc"] == pytest.approx(1.0)
    assert scored["mean_peak"] == pytest.approx(1.0, abs=0.05)
    assert scored["weight_norm"] == pytest.approx(5.2, abs=0.1)


def test_a_non_invertible_outer_map_still_reaches_the_whole_table():
    """``(n + m)^2 mod p`` is representable, at 100 per cent and not 51.

    Writing the readout as a forward map, so that every preimage of an output index
    contributes to it, is what makes the difference; inverting ``h`` instead picks one
    branch of the square root.
    """
    w1, w2 = R.closed_form("sum_sq", 23, 256)
    assert R.score_weights(w1, w2, 23, "sum_sq")["acc"] == pytest.approx(1.0, abs=0.01)


def test_a_task_without_a_closed_form_gets_no_reference():
    for task in ("mul", "mix_quad", "no_grok", "p1x"):
        assert R.reference(task, 97) is None
        assert not R.has_closed_form(task)
    with pytest.raises(KeyError):
        R.closed_form("mul", 97, 100)


# -- the three numbers of appendix M ---------------------------------------------

def test_the_mode_count_is_the_dimension_the_closed_form_fixes():
    assert R.mode_count(97) == 49
    assert R.mode_count(23) == 12


@pytest.mark.parametrize("task,expected", [
    ("add", 1.000), ("sub", 1.000), ("sq_sum", 0.052),
    ("p1", 1.000), ("p2", 1.000), ("p3", 0.062),
])
def test_the_order_parameter_reference_matches_appendix_m(task, expected):
    """Each task's own reference, which is the only thing its measured value means.

    Two of the tasks that generalise have a reference near the floor, because their
    representation is periodic in ``g1(n)`` while the spectrum is taken over the raw
    index and ``g1`` is not linear there. Against such a reference a measured 0.044
    indicates convergence; against the 1.000 modular addition sets it would say nothing
    had been learned.
    """
    reference = R.reference(task, 97, 500)
    assert reference["order_parameter"] == pytest.approx(expected, abs=0.002)
    assert reference["mode_count"] == 49


def test_the_effective_rank_carries_almost_no_signal():
    """Appendix M's reason for declining to read anything into it.

    The closed-form first layer reads 148.8 and random initialisation reads about 139,
    within seven per cent of it before any training has happened.
    """
    analytic = R.reference("add", 97, 500)["effective_rank"]
    floor = R.initialisation_reference(97, 500)
    assert analytic == pytest.approx(148.8, abs=0.5)
    assert floor["effective_rank"] == pytest.approx(139, abs=2)
    assert abs(analytic - floor["effective_rank"]) / analytic < 0.07
    # The order parameter, by contrast, separates the two by a factor of twenty-five.
    assert floor["order_parameter"] == pytest.approx(0.041, abs=0.003)


def test_the_first_layer_of_the_closed_form_is_full_rank():
    """500 x 194, because each non-zero frequency contributes a sine and a cosine
    direction in each of the two input blocks."""
    w1, _ = R.closed_form("add", 97, 500)
    assert w1.shape == (500, 194)
    assert np.linalg.matrix_rank(w1) == 194


def test_one_neuron_carries_one_frequency():
    w1, _ = R.closed_form("add", 23, 64)
    assert fourier_ipr(w1[:, :23]) == pytest.approx(1.0, abs=1e-9)
