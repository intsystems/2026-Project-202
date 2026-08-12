"""The section 7 experiments, and the shared log analysis under them.

Most of these tests exist because of a specific defect the port had to fix, and each says
which one. Three of them are about a thing that is *not* a number: a run set that was
discovered by globbing, a window label that was one logging interval too long, and a
post-transition segment sliced out of a run that had diverged after a few hundred steps.
None of the three would show up as a wrong value in a table; all three change what the
table is a table of.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from actdim.analysis import logs
from actdim.experiments import grok
from actdim.runtime import registry
from actdim.runtime.archive import BASELINE

GROK_IDS = (
    "grok.diagnostics.logs", "grok.diagnostics.perceptron", "grok.rank.dip",
    "grok.matched.window", "grok.matched.surrogate", "grok.extended.outcomes",
    "grok.prwindow", "grok.eos", "grok.repr",
)


# -- the catalogue -------------------------------------------------------------------


def test_every_section_seven_experiment_is_registered():
    registry.load()
    missing = [i for i in GROK_IDS if i not in registry.REGISTRY]
    assert not missing, f"not registered: {missing}"


@pytest.mark.parametrize("experiment_id", GROK_IDS)
def test_a_promoted_output_is_one_the_archive_mapping_knows(experiment_id):
    """A promoted name is what `actdim diff` compares against the published file.

    Renaming one silently would leave the article reading a file no experiment claims to
    produce, which is the failure `data/manifest.json` exists to prevent. `grok.repr` has
    no archived counterpart -- nothing in the archived tree wrote its tables -- so it is
    the one row with nothing to compare against.
    """
    experiment = registry.get(experiment_id)
    mapped = set(BASELINE.get(experiment_id, {}))
    if not mapped:
        assert experiment_id == "grok.repr"
        return
    assert mapped.issubset(set(experiment.promotes)), (
        f"{experiment_id} does not promote {sorted(mapped - set(experiment.promotes))}")


@pytest.mark.parametrize("experiment_id,upstream", [
    ("grok.rank.dip", "train.transformer.sketched"),
    ("grok.prwindow", "train.perceptron.sketched.long"),
])
def test_the_two_experiments_whose_input_was_lost_declare_the_campaign(experiment_id,
                                                                      upstream):
    """Errata items 15 and 16: the sketches behind these two were never kept.

    They cannot be rebuilt from the repository at any cost, only by re-training, so each
    declares the campaign that produces its input rather than exiting quietly on an empty
    directory the way the archived scripts did.
    """
    assert upstream in registry.get(experiment_id).needs


def test_every_declared_prerequisite_exists():
    registry.load()
    for experiment_id in GROK_IDS:
        for need in registry.get(experiment_id).needs:
            assert need in registry.REGISTRY, f"{experiment_id} needs unknown {need}"


# -- the run sets are declared, not discovered ---------------------------------------


def test_the_perceptron_probe_covers_the_ten_runs_the_figure_counts():
    """Errata item 8. `fig_map`'s legend reads "perceptron, full batch (10)".

    The archived command globbed every `*_train.csv` beside it, which is seven runs in the
    arithmetic directory alone, so re-running it would have turned ten into thirteen.
    """
    assert len(logs.PERCEPTRON_ARITH) + len(logs.PERCEPTRON_POLY) == 10
    assert logs.PERCEPTRON_PROBE_COLUMNS == ("train_loss", "weight_norm")


def test_the_polynomial_labels_carry_the_modulus_the_article_prints():
    """The label in the table is the article's; the file on disk may use either name."""
    assert all(name.endswith("_p97") for name in logs.PERCEPTRON_POLY)
    assert logs.log_candidates("g_p1_p97") == ("g_p1_p97_train.csv", "g_p1_train.csv")
    assert logs.log_candidates("a_add") == ("a_add_train.csv",)


def test_the_sketched_and_extended_run_sets_come_from_the_registry():
    from actdim.training import runs as trained

    assert set(logs.TRANSFORMER_SKETCHED) == set(trained.SKETCHED_RUNS)
    assert set(logs.TRANSFORMER_EXTENDED) == set(trained.EXTENDED_RUNS)
    # Sorted, because every archived table was written by a sorted glob and a regenerated
    # table has to line up with it row for row before a diff can say anything.
    assert list(logs.TRANSFORMER_SKETCHED) == sorted(logs.TRANSFORMER_SKETCHED)


def test_the_controls_are_paired_with_the_run_they_match():
    assert logs.CONTROL_OF == {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}
    assert set(logs.GENERALISING) == {"mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"}


# -- the matched-window grid ----------------------------------------------------------


def test_the_grid_has_thirty_six_cells():
    """Appendix L states the size of the grid, and section 7.3 reports every cell of it."""
    assert len(grok.MATCHED_GRID) == 36
    assert len(set(map(str, grok.MATCHED_GRID))) == 36


def test_the_headline_cell_is_what_the_rule_selects_and_not_the_best_cell():
    """Requirement 2: the configuration may not be chosen on the outcome.

    The rule, applied here to the grid rather than quoted from the article: the delay span
    must be at most a quarter of the window, and subject to that the embedding dimension is
    as large as it goes, at the frozen configuration's own neighbour count and Theiler
    rule. Recomputing it is what keeps the headline from drifting to whichever cell reads
    best.
    """
    from actdim import frozen

    base = frozen.eight_direction()
    quarter = logs.MATCHED_WINDOW / 4.0
    eligible = [c for c in grok.MATCHED_GRID
                if (c["max_E"] - 1) * c["tau"] <= quarter
                and c["k_neighbors"] == base.k_neighbors
                and c["theiler"] == base.theiler]
    assert eligible, "no cell satisfies the rule"
    best = max(eligible, key=lambda c: (c["max_E"], -c["tau"]))
    assert best == grok.MATCHED_HEADLINE


def test_the_frozen_configuration_does_not_fit_the_matched_window():
    """Why a configuration has to be chosen at all: the frozen delay span is 76 samples."""
    from actdim import frozen

    base = frozen.eight_direction()
    assert (base.max_E - 1) * base.tau > logs.MATCHED_WINDOW


# -- the surrogate control ------------------------------------------------------------


def test_the_published_control_needs_unmatched_surrogates():
    """The surrogate has to keep the record's length or it leaves the observed grid.

    `actdim.estimator.surrogates` matches endpoints by default, which trims up to fifteen
    per cent of the series; the matched-window control pairs each surrogate with the
    direct measurement's own window grid, so a shorter series would shift that grid and
    the comparison would stop being paired. Passing `match=False` reproduces the published
    control exactly, and this test is what says the default would not.
    """
    from actdim.estimator.surrogates import iaaft

    rng = np.random.default_rng(0)
    # Ends that disagree, which is the case endpoint matching exists to trim.
    x = np.linspace(0.0, 6.0, 1200) + 0.05 * rng.standard_normal(1200)

    assert len(iaaft(x, iters=5, rng=np.random.default_rng(1), match=False)) == len(x)
    assert len(iaaft(x, iters=5, rng=np.random.default_rng(1), match=True)) < len(x)


# -- the depth statistic, in one place ------------------------------------------------


def test_depth_refuses_a_grid_belonging_to_another_trace():
    """The alignment defect of `e10_surrogate`, made impossible rather than documented.

    It called `depth(t, ms, tgen)` with the observed trace's grid and a surrogate's values.
    A surrogate that produced one constant window returned a shorter array, so the call
    either raised or -- when a different window happened to be dropped and the lengths
    still matched -- compared two traces sampled at different instants.
    """
    t = np.arange(0.0, 6000.0, 100.0)
    y = np.ones_like(t)
    with pytest.raises(ValueError, match="window grid"):
        logs.depth(t, y[:-1], 3000.0)


def test_depth_is_the_pre_transition_level_over_the_post_transition_floor():
    t = np.arange(0.0, 8000.0, 100.0)
    y = np.where(t < 5000.0, 4.0, 1.0)
    assert logs.depth(t, y, 5000.0) == pytest.approx(4.0)
    # A centre with fewer than three windows on either side has no depth at all.
    assert np.isnan(logs.depth(t, y, 200.0))


def test_depth_uses_the_intervals_the_article_states():
    assert logs.PRE == (-3000, -1000)
    assert logs.POST == (-1000, 2000)


def test_the_floor_offset_is_measured_from_the_transition():
    t = np.arange(0.0, 8000.0, 100.0)
    y = np.ones_like(t)
    y[t == 5300.0] = 0.1
    assert logs.floor_offset(t, y, 5000.0) == pytest.approx(300.0)


# -- milestones ------------------------------------------------------------------------


def test_a_run_that_touches_the_threshold_and_falls_back_has_not_generalised():
    """Appendix G's rule, which is deliberately not the first-crossing rule.

    The 120,000-step reruns are asked whether a run *ended up* generalising, and two
    configurations previously counted as negatives do. A run that reaches the threshold
    and falls away has not, and `milestones` -- the right rule for a transition already
    known to have happened -- would report the first crossing regardless.
    """
    from actdim.sketch.analysis import milestones

    step = np.arange(0, 2000, 10)
    accuracy = np.where((step >= 500) & (step < 800), 1.0, 0.0)
    frame = pd.DataFrame({"step": step, "train_acc": accuracy, "val_acc": accuracy})

    assert logs.first_sustained(step, accuracy) is None
    assert milestones(frame)[1] == 500


def test_a_run_that_holds_the_threshold_reports_where_it_started_holding():
    """The rule smooths over five logged rows, centred, so a step crossing is reported two
    rows late -- 1,220 for a jump at 1,200. That lag is the price of not calling a single
    noisy row a transition, and it is the same five-row rule appendix O's stricter
    milestone uses."""
    step = np.arange(0, 2000, 10)
    accuracy = np.where(step >= 1200, 1.0, 0.0)
    assert logs.first_sustained(step, accuracy) == 1220
    assert logs.first_sustained(step, accuracy, window=1) == 1200


def test_the_milestone_map_is_read_from_the_direct_measurement(tmp_path):
    path = tmp_path / "rank_milestones.json"
    path.write_text(json.dumps([
        {"run": "mod_wd1", "t_mem": 1470, "t_gen": 13700},
        {"run": "mod_wd0", "t_mem": 630, "t_gen": None},
    ]), encoding="utf-8")
    found = logs.milestone_map(path)
    assert found == {"mod_wd1": (1470, 13700), "mod_wd0": (630, None)}
    # A control has no transition of its own and is measured in its match's window.
    assert logs.transition_of("mod_wd0", found) == 13700.0


# -- the window geometry -----------------------------------------------------------


def test_the_training_log_geometry_is_a_third_of_the_record_by_one_thousand():
    """Appendix C's override: the window and the stride move, no estimator field does."""
    from actdim import frozen

    base = frozen.eight_direction()
    cfg = logs.article_geometry(12_000)
    assert (cfg.window, cfg.stride) == (4000, 1000)
    for field in ("max_E", "tau", "k_neighbors", "theiler", "dither", "theiler_cap"):
        assert getattr(cfg, field) == getattr(base, field)


def test_the_matched_window_is_counted_in_samples_and_not_in_steps():
    """60 samples is 600 optimiser steps at a stride of 10 and 300 at a stride of 5.

    Fixing the step count instead would mismatch the S_5 runs by a factor of two; fixing
    the sample count matches every run to within two per cent of the direct measurement's
    own window.
    """
    x = np.sin(np.arange(4000) / 30.0)
    for stride in (5, 10):
        centres = np.array([1000.0, 2000.0]) * stride / 10.0
        found = list(logs.matched_windows(x, stride, centres))
        assert [len(v) for _, v in found] == [logs.MATCHED_WINDOW] * 2


def test_a_window_that_runs_off_the_record_is_dropped_with_its_centre():
    """The trace comes back on the grid it has, which is what makes the pairing honest."""
    x = np.sin(np.arange(400) / 7.0)
    centres = np.array([0.0, 1000.0, 2000.0, 100_000.0])
    found = list(logs.matched_windows(x, 10, centres))
    assert [c for c, _ in found] == [1000.0, 2000.0]


def test_a_flat_window_is_dropped_rather_than_scored():
    x = np.concatenate([np.zeros(200), np.sin(np.arange(200) / 5.0)])
    centres = np.array([1000.0, 3000.0])
    assert [c for c, _ in logs.matched_windows(x, 10, centres)] == [3000.0]


# -- the edge-of-stability guard -------------------------------------------------------


def test_a_diverged_run_is_not_a_trajectory_even_when_pandas_reads_its_gap_as_nan():
    """The guard that keeps a 567-step blow-up out of the post-transition segment.

    `diverged_at` is empty for every run that did not diverge, and pandas reads an empty
    numeric cell as NaN, which is not None. Without the conversion `analysable` would
    admit every row and the archived defect would come straight back.
    """
    from actdim.training.eos import analysable

    frame = pd.DataFrame([{"key": "a", "diverged_at": None, "n_rows": 30001},
                          {"key": "b", "diverged_at": 567, "n_rows": 567}])
    rows = frame.to_dict("records")
    assert np.isnan(rows[0]["diverged_at"])          # what pandas actually hands back
    assert [analysable(grok._clean(r)) for r in rows] == [True, False]


def test_the_logging_strides_are_the_ones_appendix_q_reports():
    assert grok.EOS_SUBSAMPLES == (1, 10, 50)
    # The frozen lag first, because it is the protocol as published; tau = 1 beside it,
    # because against a two-step cycle an even lag is close to the worst possible choice.
    assert grok.EOS_TAUS[0] == 4


# -- the window-length label ------------------------------------------------------------


def _sketch(tmp_path, name, rows, dim=8, log_every=10, seed=0):
    """A small trajectory sketch of the shape the recorder writes."""
    rng = np.random.default_rng(seed)
    step = np.arange(rows, dtype=np.int64) * log_every
    z = np.cumsum(rng.standard_normal((rows, 2, dim)), axis=0)
    path = tmp_path / name
    np.savez_compressed(
        path, step=step, z=z, zf=z * 0.5 + 1.0,
        param_step=np.abs(rng.standard_normal(rows)),
        param_norm=np.linspace(1.0, 4.0, rows),
        n_params=np.asarray(1234), dim=np.asarray(dim), n_sketch=np.asarray(2))
    return path


def test_a_window_is_labelled_by_the_steps_it_actually_covers(tmp_path):
    """Errata item 7. The article quotes these labels.

    A window of `n` samples taken every `stride` rows spans `stride*(n-1)+1` rows, so it
    covers `stride*(n-1)` logging intervals. The archived label was `stride*n*log_every`,
    one interval too many: the row printed as 600 steps covers 590.
    """
    path = _sketch(tmp_path, "long_sketch.npz", rows=400, log_every=50)
    rows = grok._prwindow_length(("a_add", str(path), "fixed_n", 2, 60, 50, None))
    assert rows, "no window fitted"
    first = rows[0]
    assert first["window_rows"] == 2 * (60 - 1) + 1
    assert first["window_steps"] == 2 * 59 * 50
    assert first["right_step"] - first["left_step"] == first["window_steps"]


# -- end to end ------------------------------------------------------------------------


def _training_log(path, rows, log_every=10, t_gen=None, seed=0):
    """A log with the columns every section 7 analysis reads."""
    rng = np.random.default_rng(seed)
    step = np.arange(rows) * log_every
    val = np.zeros(rows) if t_gen is None else (step >= t_gen).astype(float)
    frame = pd.DataFrame({
        "step": step,
        "train_loss": np.exp(-step / (rows * log_every / 4.0)) + 1e-3 * rng.standard_normal(rows),
        "val_loss": np.exp(-step / (rows * log_every / 3.0)) + 1e-3 * rng.standard_normal(rows),
        "train_acc": (step >= 200).astype(float),
        "val_acc": val,
        "weight_norm": np.linspace(1.0, 5.0, rows) + 1e-3 * rng.standard_normal(rows),
    })
    frame.to_csv(path, index=False)
    return frame


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """A `runs/` tree the context resolves inputs against, with nothing else in it."""
    from actdim.runtime import context as context_module

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(context_module, "runs_root", lambda: root)
    monkeypatch.setattr(context_module, "data_root", lambda: tmp_path / "data")
    return root


def _context(experiment_id, root, fast=True):
    from actdim.runtime.context import build

    return build(experiment_id, device="cpu", jobs=1, seed=0, fast=fast, root=root)


def test_the_collapse_runs_end_to_end_on_a_sketch(fake_tree):
    """The plumbing of `grok.rank.dip`: two geometries, six tables each, one input set."""
    upstream = fake_tree / "train.transformer.sketched"
    upstream.mkdir()
    for run, t_gen in (("mod_wd1", 2000), ("mod_wd0", None)):
        _training_log(upstream / f"{run}_train.csv", rows=600, t_gen=t_gen)
        _sketch(upstream, f"{run}_sketch.npz", rows=600, seed=1)

    # The decorator returns the function unchanged, so an experiment stays directly
    # callable in a test and the registry is not in the way.
    grok.rank_dip(_context("grok.rank.dip", fake_tree))

    out = fake_tree / "grok.rank.dip"
    for name in registry.get("grok.rank.dip").promotes:
        assert (out / name).exists(), f"{name} was not written"
    # The coarse pass is stamped, so the two passes cannot overwrite one another. That is
    # the hygiene defect the archived collapse script shipped: it pinned its figure
    # directory beside the code and the fine pass overwrote the coarse one's output.
    assert (out / "rank_windows_coarse.csv").exists()

    windows = pd.read_csv(out / "rank_windows.csv")
    assert set(windows.run) == {"mod_wd0", "mod_wd1"}
    # The eight all-NaN smoothing columns of the archived table are not emitted: a
    # sixty-row window leaves two blocks of twenty and a participation ratio needs three.
    assert not [c for c in windows.columns if "step20" in c or "step1_" in c]
    assert "centre" in windows.columns

    collapse = pd.read_csv(out / "rank_dip.csv")
    assert list(collapse.run.unique()) == ["mod_wd1"]
    assert "at" in collapse.columns, "the published column name is `at`, not `offset`"
    assert list(pd.read_csv(out / "rank_dip_controls.csv").run.unique()) == ["mod_wd0"]


def test_a_missing_sketch_names_the_campaign_that_produces_it(fake_tree):
    """Rather than writing an empty table, which is what the archived scripts did."""
    upstream = fake_tree / "train.transformer.sketched"
    upstream.mkdir()
    for run in ("mod_wd1", "mod_wd0"):
        _training_log(upstream / f"{run}_train.csv", rows=300)

    ctx = _context("grok.rank.dip", fake_tree)
    with pytest.raises(FileNotFoundError) as raised:
        grok.rank_dip(ctx)
    assert "python -m actdim run train.transformer.sketched" in str(raised.value)
    assert "errata" in str(raised.value)


def test_the_window_length_sweep_runs_once_a_sketch_exists(fake_tree):
    """`grok.prwindow` cannot be run against anything committed -- its input was never
    kept -- so the path it will take after the GPU campaign is exercised here instead."""
    upstream = fake_tree / "train.perceptron.sketched.long"
    upstream.mkdir()
    _training_log(upstream / "a_add_train.csv", rows=400, log_every=50, t_gen=8000)
    _sketch(upstream, "a_add_sketch.npz", rows=400, log_every=50, seed=3)

    grok.prwindow(_context("grok.prwindow", fake_tree))

    out = fake_tree / "grok.prwindow"
    summary = pd.read_csv(out / "pr_vs_window.csv")
    assert (out / "pr_vs_window_windows.csv").exists()
    assert set(summary.ladder) == {"fixed_n", "fixed_dt"}
    assert list(summary.run.unique()) == ["a_add"]
    # Sixty samples at every length in the ladder the article reads.
    assert set(summary[summary.ladder == "fixed_n"].n_samples) == {60}
    assert summary.window_steps.min() > 0


def test_the_extended_outcomes_run_end_to_end(fake_tree):
    """`grok.extended.outcomes` on two runs, one of which generalises and one of which
    reaches the threshold and falls back."""
    upstream = fake_tree / "train.transformer.extended"
    upstream.mkdir()
    _training_log(upstream / "grokpos_s0_train.csv", rows=3000, t_gen=5000)
    frame = _training_log(upstream / "lowdata15_s0_train.csv", rows=3000, t_gen=5000)
    frame.loc[frame.step > 20_000, "val_acc"] = 0.0
    frame.to_csv(upstream / "lowdata15_s0_train.csv", index=False)

    ctx = _context("grok.extended.outcomes", fake_tree)
    grok.extended_outcomes(ctx)

    outcomes = pd.read_csv(fake_tree / "grok.extended.outcomes" / "exp8_outcomes.csv")
    assert list(outcomes.run) == ["grokpos_s0", "lowdata15_s0"]
    assert list(outcomes.groks) == [True, False]
    early = pd.read_csv(fake_tree / "grok.extended.outcomes" / "exp8_at_20k.csv")
    assert list(early.columns) == ["run", "val_at_20k", "x_chance_at_20k",
                                   "rho_10k_20k", "t_gen"]


def test_the_perceptron_probe_runs_end_to_end(fake_tree):
    """`grok.diagnostics.perceptron`: both arms, four tables, the declared columns."""
    for upstream, runs in (("train.perceptron.arith", logs.PERCEPTRON_ARITH),
                           ("train.perceptron.poly", logs.PERCEPTRON_POLY)):
        directory = fake_tree / upstream
        directory.mkdir()
        for run in runs[:2]:
            # The port's registry keys the polynomial runs without the modulus, so the
            # file on disk is `g_p1_train.csv` where the table's label is `g_p1_p97`.
            name = run[:-4] if run.endswith("_p97") else run
            # 2,400 rows gives a 2,000-sample window and one position, which is what keeps
            # this a plumbing test rather than a second copy of the experiment.
            _training_log(directory / f"{name}_train.csv", rows=2400, t_gen=8000)

    ctx = _context("grok.diagnostics.perceptron", fake_tree)
    grok.diagnostics_perceptron(ctx)

    out = fake_tree / "grok.diagnostics.perceptron"
    windows = pd.read_csv(out / "dimension_probe_poly.csv")
    assert set(windows.run) == {"g_p1_p97", "g_p1x_p97"}
    assert set(windows.column) == set(logs.PERCEPTRON_PROBE_COLUMNS)
    assert "LB" not in windows.columns          # the atlas has it; this table does not
    summary = pd.read_csv(out / "dimension_probe_summary_poly.csv")
    assert list(summary.columns) == list(grok.SUMMARY_COLUMNS)
