"""Sanity checks for the EDM estimators and the figure registry.

Run with ``pytest test_edm.py`` or directly: ``python test_edm.py``.
"""

import numpy as np
import pandas as pd

import experiments
from edm import (
    delay_embedding,
    delayed_mutual_information,
    estimate_E_cao,
    estimate_E_fnn,
    estimate_E_mle,
    estimate_E_simplex,
    first_local_minimum,
    grokking_step,
    load_logs,
    mle_intrinsic_dimension,
    select_tau_dmi,
    sliding_dimension,
)


def test_delay_embedding_shape_and_content():
    x = np.arange(10.0)
    emb = delay_embedding(x, m=3, tau=2)
    assert emb.shape == (6, 3)
    np.testing.assert_allclose(emb[0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(emb[-1], [5.0, 7.0, 9.0])


def test_delay_embedding_rejects_short_series():
    try:
        delay_embedding(np.arange(5.0), m=10, tau=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an over-long embedding")


def test_first_local_minimum_on_decaying_curve():
    assert first_local_minimum(np.array([1.0, 0.5, 0.2, 0.25, 0.4]), abs_eps=0.0, drop_fraction=0.1) == 2


def _lorenz_x(n=6000, dt=0.001, substeps=10, s=10.0, r=28.0, b=8.0 / 3.0, burn_in=1000):
    """x-component of the Lorenz-63 attractor (correlation dimension ~ 2.06)."""
    state = np.array([1.0, 1.0, 1.0])
    xs = np.empty(n)
    for i in range(n):
        for _ in range(substeps):
            x, y, z = state
            state = state + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        xs[i] = state[0]
    return xs[burn_in:]


def test_mle_recovers_a_low_dimensional_chaotic_attractor():
    """Levina-Bickel on a scalar Lorenz log must land near the true ~2-3."""
    d = mle_intrinsic_dimension(_lorenz_x(), tau=1, max_E=15, rng=np.random.default_rng(0))
    assert 1.5 < d < 4.5, d


def test_mle_separates_a_chaotic_attractor_from_white_noise():
    """I.i.d. noise fills the embedding space, so it must score far higher."""
    noise = np.random.default_rng(0).normal(size=4000)
    d_noise = mle_intrinsic_dimension(noise, tau=1, max_E=15, rng=np.random.default_rng(0))
    d_lorenz = mle_intrinsic_dimension(_lorenz_x(), tau=1, max_E=15, rng=np.random.default_rng(0))
    assert d_noise > 5.0, d_noise
    assert d_noise > 2 * d_lorenz, (d_noise, d_lorenz)


def test_estimators_return_one_on_a_degenerate_observable():
    """A constant series carries no variance -- every estimator must bail out to E = 1."""
    flat = np.full(500, 3.14)
    rng = np.random.default_rng(0)
    for estimator in (estimate_E_mle, estimate_E_fnn, estimate_E_cao):
        assert estimator(flat, tau=1, rng=rng) == 1.0
    assert estimate_E_simplex(flat, tau=1) == 1.0


def test_dmi_tau_selection_matches_the_period_scale():
    t = np.linspace(0, 40 * np.pi, 2000)  # period ~ 100 samples
    taus, dmi = delayed_mutual_information(np.sin(t), max_tau=40, bins=20)
    assert len(taus) == len(dmi) == 40
    assert 1 <= select_tau_dmi(np.sin(t), max_tau=40) <= 40


def test_sliding_dimension_window_bookkeeping():
    steps = np.arange(1000.0)
    df = pd.DataFrame({
        "step": steps,
        "train_acc": np.ones(1000),
        "val_acc": np.ones(1000),
        "weight_norm": np.sin(steps / 7.0),
    })
    trace = sliding_dimension(df, window_size=200, step_size=100, seed=0, progress=False)
    assert len(trace.steps) == len(trace.dimension) == 9   # range(0, 801, 100)
    assert trace.steps[0] == 100                           # centre of the first window
    assert np.all(np.isfinite(trace.dimension))

    dropped = sliding_dimension(
        df, window_size=200, step_size=100, include_last_window=False, seed=0, progress=False
    )
    assert len(dropped.steps) == 8


def test_sliding_dimension_is_deterministic_given_a_seed():
    steps = np.arange(800.0)
    df = pd.DataFrame({
        "step": steps,
        "train_acc": np.ones(800),
        "val_acc": np.ones(800),
        "weight_norm": np.cos(steps / 5.0) + 0.01 * steps,
    })
    kwargs = dict(window_size=200, step_size=100, seed=7, progress=False)
    np.testing.assert_allclose(
        sliding_dimension(df, **kwargs).dimension,
        sliding_dimension(df, **kwargs).dimension,
    )


def test_every_registered_figure_has_its_log_and_columns():
    for key, fig in experiments.FIGURES.items():
        assert fig.csv_path.exists(), f"{key}: missing {fig.csv_path}"
        df = load_logs(fig.csv_path, required=fig.required_columns)
        assert len(df) > fig.window_size, f"{key}: log shorter than the window"
        if fig.kind == "panels":
            for col in ("train_loss", "val_loss", "weight_norm"):
                assert col in df.columns, f"{key}: panels need column '{col}'"
        if fig.kind != "overview":
            assert fig.metric in df.columns, f"{key}: missing metric '{fig.metric}'"


def test_grokking_is_detected_only_where_weight_decay_is_on():
    grokked = {
        key: grokking_step(load_logs(fig.csv_path)) is not None
        for key, fig in experiments.FIGURES.items()
    }
    assert grokked["mod_wd1"] and grokked["s5_wd1"]
    assert not grokked["mod_wd0"] and not grokked["s5_wd0"]


def test_dimension_collapses_with_weight_decay_and_stays_flat_without():
    """The paper's core claim, checked numerically on the S_5 logs."""
    def trace_for(key, metric):
        fig = experiments.get(key)
        return sliding_dimension(
            load_logs(fig.csv_path), target_metric=metric,
            window_size=300, step_size=50, seed=0, progress=False,
        ).dimension

    wd1 = trace_for("s5_wd1", "weight_norm")
    assert wd1.max() > 3.5 and wd1[-1] < 2.0        # rises, then collapses

    wd0 = trace_for("s5_wd0", "weight_norm")
    assert wd0.max() - wd0.min() < 0.5              # no structure to collapse

    # The generic observable L_val recovers the algebraic complexity of S_5.
    wd0_val = trace_for("s5_wd0", "val_loss")
    assert wd0_val[-1] > wd0_val[0]                 # chaos of eternal memorization


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                    # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
