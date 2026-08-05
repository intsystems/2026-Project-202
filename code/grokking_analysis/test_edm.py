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
    embedding_dimension_scan,
    grokking_step,
    identifiability_ratio,
    load_logs,
    local_roughness,
    local_svd_dimension,
    mle_intrinsic_dimension,
    mle_log_ratio_sums,
    resolve_theiler_window,
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


def test_mackay_ghahramani_is_the_inverse_average_of_the_local_estimates():
    """MG pools the likelihood before inverting; that is a harmonic mean of d_i."""
    k, series = 5, _lorenz_x()
    sums = mle_log_ratio_sums(series, tau=1, max_E=15, k_neighbors=k, rng=np.random.default_rng(0))
    harmonic = 1.0 / np.mean(1.0 / ((k - 1) / sums))
    mg = mle_intrinsic_dimension(
        series, tau=1, max_E=15, k_neighbors=k,
        correction="mackay_ghahramani", rng=np.random.default_rng(0),
    )
    assert abs(mg - harmonic) / harmonic < 1e-3, (mg, harmonic)


def test_mackay_ghahramani_never_exceeds_levina_bickel():
    """AM-HM: averaging local estimates can only inflate the pooled value."""
    rng = np.random.default_rng(0)
    series_set = {
        "lorenz": _lorenz_x(),
        "noise": rng.normal(size=3000),
        "random_walk": np.cumsum(rng.normal(size=3000)),
    }
    for name, fig_key in (("s5_wd1", "weight_norm"), ("s5_wd0", "val_loss")):
        df = load_logs(experiments.get(name).csv_path)
        series_set[f"{name}:{fig_key}"] = df[fig_key].to_numpy()[:1000]

    for name, series in series_set.items():
        lb = mle_intrinsic_dimension(series, tau=1, correction="levina_bickel",
                                     rng=np.random.default_rng(1))
        mg = mle_intrinsic_dimension(series, tau=1, correction="mackay_ghahramani",
                                     rng=np.random.default_rng(1))
        assert mg <= lb + 1e-9, f"{name}: MG={mg} > LB={lb}"


def test_mackay_ghahramani_rejects_an_unknown_correction():
    try:
        mle_intrinsic_dimension(np.linspace(0, 1, 500), correction="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown correction")


def test_mackay_ghahramani_keeps_the_collapse_signal():
    """The correction shifts the level, but the paper's qualitative claim must hold."""
    df = load_logs(experiments.get("s5_wd1").csv_path)
    kwargs = dict(target_metric="weight_norm", window_size=300, step_size=50,
                  seed=0, progress=False)
    lb = sliding_dimension(df, method="mle", **kwargs).dimension
    mg = sliding_dimension(df, method="mle_mg", **kwargs).dimension

    assert mg.max() > 2 * mg[-1]                       # still rises, then collapses
    assert np.corrcoef(lb, mg)[0, 1] > 0.95            # same shape, lower level
    assert np.all(mg <= lb + 1e-9)


def test_theiler_window_excludes_temporal_neighbours():
    """Distances must grow once the neighbours next to i in time are barred."""
    series = np.sin(np.linspace(0, 8 * np.pi, 600)) + np.linspace(0, 1, 600)
    plain = mle_log_ratio_sums(series, tau=1, max_E=15, k_neighbors=5,
                               theiler_window=0, rng=np.random.default_rng(0))
    theiler = mle_log_ratio_sums(series, tau=1, max_E=15, k_neighbors=5,
                                 theiler_window=30, rng=np.random.default_rng(0))
    assert plain is not None and theiler is not None
    assert len(plain) == len(theiler)
    assert not np.allclose(plain, theiler)


def test_theiler_window_resolution():
    series = np.linspace(0, 1, 500)
    assert resolve_theiler_window(0, series, tau=1, max_E=15) == 0
    assert resolve_theiler_window("embedding", series, tau=1, max_E=15) == 14
    assert resolve_theiler_window("embedding", series, tau=2, max_E=8) == 14
    assert resolve_theiler_window(7, series, tau=1, max_E=15) == 7


def test_theiler_window_bails_out_when_the_window_is_too_short():
    """W too wide for the window leaves fewer than k candidates -> no estimate."""
    series = np.cumsum(np.random.default_rng(0).normal(size=60))
    assert mle_log_ratio_sums(series, tau=1, max_E=15, theiler_window=25) is None


def test_smooth_oversampled_series_returns_the_tangent_line_value():
    """Without a Theiler window, k temporal neighbours give d = (k-1)/sum log(k/j).

    For k = 5 that constant is 1.227 -- the number a straight line produces. This
    is the failure mode the Theiler window exists to prevent.
    """
    k = 5
    tangent = (k - 1) / sum(np.log(k / j) for j in range(1, k))
    assert abs(tangent - 1.227) < 0.001

    smooth = np.linspace(0.0, 1.0, 800) ** 2          # monotone, heavily oversampled
    d = mle_intrinsic_dimension(smooth, tau=1, max_E=15, k_neighbors=k,
                                theiler_window=0, rng=np.random.default_rng(0))
    assert abs(d - tangent) < 0.15, (d, tangent)


def test_theiler_window_lifts_the_flat_weight_norm_controls():
    """The E ~ 1.35 plateau of the WD=0 runs is a tangent artifact, not a measurement."""
    for key in ("mod_wd0", "s5_wd0"):
        fig = experiments.get(key)
        df = load_logs(fig.csv_path)
        kwargs = dict(target_metric="weight_norm", window_size=fig.window_size,
                      step_size=fig.step_size, seed=0, progress=False)
        plain = sliding_dimension(df, method="mle_mg", **kwargs).dimension
        theiler = sliding_dimension(df, method="mle_mg_theiler", **kwargs).dimension

        assert plain.max() < 1.5, f"{key}: expected the tangent plateau, got {plain.max()}"
        assert theiler.min() > 5.0, f"{key}: Theiler should expose a high dimension, got {theiler.min()}"


def test_label_position_shifts_estimates_by_half_a_window():
    """Centre labels are not causal: they pin a window to a step it has not reached."""
    steps = np.arange(1000.0)
    df = pd.DataFrame({"step": steps, "train_acc": np.ones(1000), "val_acc": np.ones(1000),
                       "weight_norm": np.sin(steps / 7.0)})
    kwargs = dict(window_size=200, step_size=100, seed=0, progress=False)
    centre = sliding_dimension(df, label_position="center", **kwargs)
    right = sliding_dimension(df, label_position="right", **kwargs)

    np.testing.assert_allclose(centre.dimension, right.dimension)   # same estimates
    np.testing.assert_allclose(right.steps - centre.steps, 99)      # W/2 - 1 later
    assert right.steps[0] == 199                                    # last step actually seen


def test_label_position_rejects_nonsense():
    df = pd.DataFrame({"step": np.arange(400.0), "train_acc": np.ones(400),
                       "val_acc": np.ones(400), "weight_norm": np.arange(400.0)})
    try:
        sliding_dimension(df, label_position="middle-ish", progress=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown label_position")


def test_a_straight_line_reproduces_both_published_plateaus():
    """Neither plateau is a measurement: a line with no dynamics returns both numbers.

    Without a Theiler window the k neighbours sit at |dt| = 1..k, with one at
    |dt| = W+1..W+k. Both give a closed-form constant that depends only on k and W.
    """
    line = np.linspace(0.0, 1.0, 2000)
    k, theiler = 5, 14

    def constant(offsets):
        r = np.sort(offsets)[:k]
        return (k - 1) / np.sum(np.log(r[-1] / r[:-1]))

    plain = mle_intrinsic_dimension(line, tau=1, max_E=15, k_neighbors=k,
                                    theiler_window=0, rng=np.random.default_rng(0))
    corrected = mle_intrinsic_dimension(line, tau=1, max_E=15, k_neighbors=k,
                                        correction="mackay_ghahramani",
                                        theiler_window="embedding", rng=np.random.default_rng(0))

    assert abs(plain - constant(np.repeat(np.arange(1, k + 1), 2))) < 0.05
    assert abs(corrected - constant(np.repeat(np.arange(theiler + 1, theiler + k + 1), 2))) < 0.2

    # ... and those constants are what the WD=0 control runs report.
    for key, expected in (("mod_wd0", plain), ("s5_wd0", plain)):
        df = load_logs(experiments.get(key).csv_path)
        measured = sliding_dimension(df, target_metric="weight_norm", method="mle",
                                     window_size=300, step_size=50, seed=0, progress=False)
        assert abs(measured.dimension.mean() - expected) < 0.1, (key, measured.dimension.mean())


def test_line_constant_predicts_the_measured_straight_line():
    """The closed form in best_practice must match what the estimator actually returns."""
    import best_practice

    line = np.linspace(0.0, 1.0, 3000)
    for k, theiler in ((5, 0), (5, 14), (20, 0), (20, 45)):
        predicted = best_practice.line_constant(k, theiler)
        measured = mle_intrinsic_dimension(
            line, tau=1, max_E=10, k_neighbors=k,
            correction="mackay_ghahramani", dither=None, clamp_to_max_E=False,
            theiler_window=theiler,
        )
        assert abs(measured - predicted) / predicted < 0.05, (k, theiler, measured, predicted)


def test_best_practice_pipeline_recovers_lorenz():
    """The tuned pipeline must find the right answer where one exists."""
    import best_practice

    rows = best_practice.calibrate()
    for label, low, _high, ratio in rows:
        assert 1.5 < low < 3.0, (label, low)      # true correlation dimension ~ 2.06
    theiler_ratio = rows[-1][3]
    assert abs(theiler_ratio - 1.0) < 0.1, theiler_ratio


def test_local_svd_reference_values_are_exact():
    """PR has closed-form reference values, so it needs no calibration."""
    t = np.arange(40.0)
    assert abs(local_svd_dimension(3 + 2 * t) - 1.0) < 1e-6          # straight ramp
    assert abs(local_svd_dimension(np.sin(t / 3)) - 2.0) < 0.05      # planar arc
    noise = np.random.default_rng(0).normal(size=40)
    assert local_svd_dimension(noise) > 5.0                          # fills the embedding


def test_local_svd_survives_a_20_sample_segment():
    """The point of switching estimator: 100-200 iterations is 10-40 logged samples."""
    for n in (10, 20, 40):
        ramp = 3.0 + 2.0 * np.arange(float(n))
        assert abs(local_svd_dimension(ramp) - 1.0) < 1e-6, n
        noise = np.random.default_rng(1).normal(size=n)
        assert local_svd_dimension(noise) > 1.5, n


def test_local_svd_separates_grokking_runs_from_their_controls():
    """Over ~200 iterations the weight norm leaves PR = 1 only when the model groks."""
    for treatment, control in (("mod_wd1", "mod_wd0"), ("s5_wd1", "s5_wd0")):
        peaks = {}
        for key in (treatment, control):
            fig = experiments.get(key)
            df = load_logs(fig.csv_path)
            sps = int(np.median(np.diff(df["step"].to_numpy())))
            trace = sliding_dimension(
                df, target_metric="weight_norm", method="svd",
                window_size=200 // sps, step_size=1, label_position="right",
                clip=None, seed=0, progress=False, estimator_kwargs=dict(degenerate=np.nan),
            )
            peaks[key] = np.nanmax(trace.dimension)
        assert peaks[treatment] > 3.0, (treatment, peaks[treatment])
        assert peaks[control] < 1.1, (control, peaks[control])


def test_local_svd_is_largely_a_roughness_statistic_here():
    """The null model must be run alongside PR, not assumed away.

    On these logs a linear detrend reproduces the participation ratio closely. That
    is a limitation of the *data*, not a bug -- but it means the signal should be
    described as departure from local linearity rather than as a dimension.
    """
    def rank(v):
        return np.argsort(np.argsort(v))

    for key in ("s5_wd1", "mod_wd1"):
        series = load_logs(experiments.get(key).csv_path)["weight_norm"].to_numpy()
        window = 40 if key.startswith("s5") else 20
        segments = [series[i:i + window] for i in range(0, len(series) - window + 1, 5)]
        pr = np.array([local_svd_dimension(s) for s in segments])
        rough = np.array([local_roughness(s) for s in segments])
        ok = np.isfinite(pr) & np.isfinite(rough)
        rho = np.corrcoef(rank(pr[ok]), rank(rough[ok]))[0, 1]
        assert rho > 0.75, (key, rho)


def test_local_roughness_reference_values():
    """Exactly 0 for a straight line -- the null model has a reference value too."""
    t = np.arange(40.0)
    assert local_roughness(3 + 2 * t) < 1e-12
    assert local_roughness(np.sin(t / 3)) > 0.5
    assert np.isnan(local_roughness(np.full(40, 2.0)))       # no variation at all


def test_identifiability_ratio_separates_a_manifold_from_a_point_cloud():
    """A dimension exists only when the estimate ignores the space it is measured in."""
    lorenz = _lorenz_x(n=12000, burn_in=1000)
    assert identifiability_ratio(lorenz) < 1.15                       # flat scan -> real
    assert identifiability_ratio(np.random.default_rng(0).normal(size=3000)) > 1.6


def test_weight_norm_logs_admit_no_identifiable_dimension():
    """Documents why the Theiler-corrected weight-norm numbers are not a measurement."""
    for key in ("s5_wd1", "s5_wd0"):
        series = load_logs(experiments.get(key).csv_path)["weight_norm"].to_numpy()
        assert identifiability_ratio(series) > 1.6, key

    scan = embedding_dimension_scan(series)
    values = [scan[m] for m in sorted(scan)]
    assert values == sorted(values), "E should track max_E when no manifold is resolvable"


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
