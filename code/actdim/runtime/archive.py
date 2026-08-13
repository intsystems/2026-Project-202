"""The archived tree, as a baseline to measure the port against.

`../archived_code/` holds the results the article was written from. They are the reference
for two things and neither of them is production:

*Diffing.* The port fixes defects, and fixing them moves values. `actdim diff <id>`
compares a regenerated table against the archived one column by column, so that every
number that moved is named rather than discovered later in a proof.

*Bootstrapping.* Until an experiment has been re-run, the figures and the table check have
nothing to read. `actdim bootstrap` copies the archived files into `data/` and marks each
manifest entry `source: archived`, so it is obvious in the manifest which numbers are
still the old ones.

The mapping below is the only place the archived layout is written down. When an
experiment is re-run and promoted, its entry in the manifest stops being marked and the
archived copy stops being consulted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .store import repo_root


def archive_root() -> Path:
    return repo_root().parent / "archived_code"


# experiment id -> {promoted name: path relative to archived_code/}
BASELINE: Dict[str, Dict[str, str]] = {
    "calib.e8": {
        "frozen_config.json": "active_dimension/results/e1_calibration/frozen_config.json",
        "config_ranking.csv": "active_dimension/results/e1_calibration/config_ranking.csv",
        "calibration_scores.csv": "active_dimension/results/e1_calibration/calibration_scores.csv",
    },
    "calib.e20": {
        "frozen_k20.json": "active_dimension/results/k20_calibration/frozen_k20.json",
        "scores_frozen.csv": "active_dimension/results/k20_calibration/scores_frozen.csv",
        "frozen_per_r.csv": "active_dimension/results/k20_calibration/frozen_per_r.csv",
        "heldout_qp_summary.csv": "active_dimension/results/k20_calibration/heldout_qp_summary.csv",
        "invariance_controls.csv": "active_dimension/results/k20_calibration/invariance_controls.csv",
    },
    # The first two systems name their tables differently from the rest of the ladder:
    # the held-out results are `stationary_validation.csv` and the calibration grid is
    # `calibration_grid.csv`, where the later experiments write `heldout_raw.csv` and
    # `observer_ranking.csv`. The port gives every system the same names.
    "sys.matrix": {
        "stationary_validation.csv": "dimension_recovery/results/exp9_frobenius_k10/stationary_validation.csv",
        "calibration_grid.csv": "dimension_recovery/results/exp9_frobenius_k10/calibration_grid.csv",
        "zigzag_segments.csv": "dimension_recovery/results/exp9_frobenius_k10/zigzag_segments.csv",
        "best_config.json": "dimension_recovery/results/exp9_frobenius_k10/best_config.json",
    },
    "sys.matrix.k20": {
        "stationary_validation.csv": "dimension_recovery/results/exp10_frobenius_k20/stationary_validation.csv",
        "calibration_grid.csv": "dimension_recovery/results/exp10_frobenius_k20/calibration_grid.csv",
        "zigzag_segments.csv": "dimension_recovery/results/exp10_frobenius_k20/zigzag_segments.csv",
        "best_config.json": "dimension_recovery/results/exp10_frobenius_k20/best_config.json",
    },
    "sys.linear": {
        "observer_ranking.csv": "dimension_recovery/results/exp11_online_regression_k20/observer_ranking.csv",
        "heldout_raw.csv": "dimension_recovery/results/exp11_online_regression_k20/heldout_raw.csv",
        "heldout_summary.csv": "dimension_recovery/results/exp11_online_regression_k20/heldout_summary.csv",
        "calibration_raw.csv": "dimension_recovery/results/exp11_online_regression_k20/calibration_raw.csv",
    },
    "sys.logistic": {
        "observer_ranking.csv": "dimension_recovery/results/exp12_logistic_regression_k20/observer_ranking.csv",
        "heldout_raw.csv": "dimension_recovery/results/exp12_logistic_regression_k20/heldout_raw.csv",
        "heldout_summary.csv": "dimension_recovery/results/exp12_logistic_regression_k20/heldout_summary.csv",
        "calibration_raw.csv": "dimension_recovery/results/exp12_logistic_regression_k20/calibration_raw.csv",
    },
    "sys.decoder": {
        "observer_ranking.csv": "dimension_recovery/results/exp13_frozen_nonlinear_decoder_k20/observer_ranking.csv",
        "heldout_raw.csv": "dimension_recovery/results/exp13_frozen_nonlinear_decoder_k20/heldout_raw.csv",
        "heldout_summary.csv": "dimension_recovery/results/exp13_frozen_nonlinear_decoder_k20/heldout_summary.csv",
        "calibration_raw.csv": "dimension_recovery/results/exp13_frozen_nonlinear_decoder_k20/calibration_raw.csv",
    },
    "sys.subspace": {
        "observer_ranking.csv": "dimension_recovery/results/exp14_mlp_intrinsic_subspace_k20/observer_ranking.csv",
        "heldout_raw.csv": "dimension_recovery/results/exp14_mlp_intrinsic_subspace_k20/heldout_raw.csv",
        "heldout_summary.csv": "dimension_recovery/results/exp14_mlp_intrinsic_subspace_k20/heldout_summary.csv",
        "calibration_raw.csv": "dimension_recovery/results/exp14_mlp_intrinsic_subspace_k20/calibration_raw.csv",
        "functional_rank_check.csv": "dimension_recovery/results/exp14_mlp_intrinsic_subspace_k20/functional_rank_check.csv",
    },
    "sys.digits.function": {
        "observer_ranking.csv": "dimension_recovery/results/exp15_real_digits_functional_subspace_v3/observer_ranking.csv",
        "heldout_raw.csv": "dimension_recovery/results/exp15_real_digits_functional_subspace_v3/heldout_raw.csv",
        "heldout_summary.csv": "dimension_recovery/results/exp15_real_digits_functional_subspace_v3/heldout_summary.csv",
        "calibration_raw.csv": "dimension_recovery/results/exp15_real_digits_functional_subspace_v3/calibration_raw.csv",
        "rank_diagnostics.csv": "dimension_recovery/results/exp15_real_digits_functional_subspace_v3/rank_diagnostics.csv",
    },
    "sys.digits.parameter": {
        "sweep_raw.csv": "active_dimension/results/e2_rank_sweep/sweep_raw.csv",
        "observer_scores.csv": "active_dimension/results/e2_rank_sweep/observer_scores.csv",
        "calibrated_mae.csv": "active_dimension/results/e2_rank_sweep/calibrated_mae.csv",
        "ground_truth_PR.csv": "active_dimension/results/e2_rank_sweep/ground_truth_PR.csv",
    },
    "valid.regime": {
        "atlas_raw.csv": "active_dimension/results/e0_atlas/atlas_raw.csv",
        "identifiability_ratio.csv": "active_dimension/results/e0_atlas/identifiability_ratio.csv",
    },
    "valid.tau": {"tau_sensitivity.csv": "active_dimension/results/e6_tau/tau_sensitivity.csv"},
    "valid.nuisance": {
        "controls_raw.csv": "active_dimension/results/e4_controls/controls_raw.csv",
        "controls_scored.csv": "active_dimension/results/e4_controls/controls_scored.csv",
    },
    "valid.anisotropy": {
        "aniso_raw.csv": "active_dimension/results/e8_anisotropy/aniso_raw.csv",
        "aniso_summary.csv": "active_dimension/results/e8_anisotropy/aniso_summary.csv",
    },
    "valid.transitions": {
        "transitions_raw.csv": "active_dimension/results/e3_transitions/transitions_raw.csv",
    },
    # Arrived by merge from a branch predating the reorganisation, so it was written against
    # the archived `mg.py` and never had a place in this table until it was ported.
    "valid.geometry": {
        "geometry_levels.csv": "active_dimension/results/e12_mle_geometry_demo/geometry_levels.csv",
        "geometry_switch_summary.csv": "active_dimension/results/e12_mle_geometry_demo/geometry_switch_summary.csv",
        "observer_warps.csv": "active_dimension/results/e12_mle_geometry_demo/observer_warps.csv",
        "verdict.json": "active_dimension/results/e12_mle_geometry_demo/verdict.json",
    },
    "valid.theiler.cap": {
        "theiler_quick_raw.csv": "active_dimension/results/e7_theiler/theiler_quick_raw.csv",
    },
    "valid.theiler.contrast": {
        "example_traces.csv": "active_dimension/results/e11_theiler_contrast/example_traces.csv",
        "sweep_windows.csv": "active_dimension/results/e11_theiler_contrast/sweep_windows.csv",
    },
    "valid.ceiling": {
        "ceiling_summary.csv": "active_dimension/results/e10_ceiling/ceiling_summary.csv",
        "ceiling_cells.csv": "active_dimension/results/e10_ceiling/ceiling_cells.csv",
        "ceiling_fits.csv": "active_dimension/results/e10_ceiling/ceiling_fits.csv",
        "ceiling_slopes.csv": "active_dimension/results/e10_ceiling/ceiling_slopes.csv",
    },
    "grok.diagnostics.logs": {
        "real_logs_summary.csv": "active_dimension/results/e5_real_logs/real_logs_summary.csv",
        "real_logs_windows.csv": "active_dimension/results/e5_real_logs/real_logs_windows.csv",
    },
    "grok.diagnostics.perceptron": {
        "dimension_probe.csv": "gromov_arithmetic/results/arith/dimension_probe.csv",
        "dimension_probe_summary.csv": "gromov_arithmetic/results/arith/dimension_probe_summary.csv",
        "dimension_probe_poly.csv": "gromov_polynomials/results/dimension_probe.csv",
        "dimension_probe_summary_poly.csv": "gromov_polynomials/results/dimension_probe_summary.csv",
    },
    "grok.rank.dip": {
        "rank_windows.csv": "active_rank/results_fine/rank_windows.csv",
        "rank_summary.csv": "active_rank/results_fine/rank_summary.csv",
        "rank_milestones.json": "active_rank/results_fine/rank_milestones.json",
        "rank_dip.csv": "active_rank/results_fine/rank_dip.csv",
        "rank_dip_controls.csv": "active_rank/results_fine/rank_dip_controls.csv",
        "rank_dip_controls_aligned.csv": "active_rank/results_fine/rank_dip_controls_aligned.csv",
        "mod_wd1_train.csv": "active_rank/results_fine/mod_wd1_train.csv",
    },
    "grok.matched.window": {
        "headline_trace.csv": "active_dimension/results/e9_matched_window/headline_trace.csv",
    },
    "grok.matched.surrogate": {
        "surrogates.csv": "active_dimension/results/e9_matched_window/surrogates.csv",
        "surrogate_summary.csv": "active_dimension/results/e9_matched_window/surrogate_summary.csv",
        "surrogate_seed_spread.csv": "active_dimension/results/e9_matched_window/surrogate_seed_spread.csv",
    },
    "grok.extended.outcomes": {
        "exp8_outcomes.csv": "dimension_recovery/results/exp8_outcomes.csv",
        "exp8_at_20k.csv": "dimension_recovery/results/exp8_at_20k.csv",
    },
    "grok.prwindow": {
        "pr_vs_window.csv": "gromov_arithmetic/results/rank_fb_long/pr_vs_window.csv",
    },
    "grok.eos": {
        "eos_diagnostics.csv": "gromov_arithmetic/results/eos/eos_diagnostics.csv",
        "eos_diagnostics_summary.csv": "gromov_arithmetic/results/eos/eos_diagnostics_summary.csv",
        "eos_recurrence.csv": "gromov_arithmetic/results/eos/eos_recurrence.csv",
    },
    # The eight sharpness logs and one dense training log that fig_eos draws, and the two
    # polynomial logs behind fig_pairs, are inputs to a figure rather than intermediates,
    # so they belong in the tracked half alongside the summaries.
    "train.perceptron.eos": {
        "eos_runs.csv": "gromov_arithmetic/results/eos/eos_runs.csv",
        "eos_lr100000_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr100000_s1_sharp.csv",
        "eos_lr300000_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr300000_s1_sharp.csv",
        "eos_lr1e+06_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr1e+06_s1_sharp.csv",
        "eos_lr1.5e+06_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr1.5e+06_s1_sharp.csv",
        "eos_lr2e+06_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr2e+06_s1_sharp.csv",
        "eos_lr2.5e+06_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr2.5e+06_s1_sharp.csv",
        "eos_lr2.8e+06_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr2.8e+06_s1_sharp.csv",
        "eos_lr3e+06_s1_sharp.csv": "gromov_arithmetic/results/eos/eos_lr3e+06_s1_sharp.csv",
        "eos_lr2e+06_s1_train.csv": "gromov_arithmetic/results/eos/eos_lr2e+06_s1_train.csv",
    },
    "train.perceptron.poly": {
        "g_p2_p97_train.csv": "gromov_polynomials/results/g_p2_p97_train.csv",
        "g_p2x_p97_train.csv": "gromov_polynomials/results/g_p2x_p97_train.csv",
    },
    "check.sketch.cost": {
        "sketch_cost.json": "gromov_arithmetic/results/sketch_cost.json",
    },
}

# Whole directories whose per-run files the figures resolve by run name.
BASELINE_DIRS: Dict[str, str] = {
    "train.perceptron.eos": "gromov_arithmetic/results/eos",
    "train.perceptron.poly": "gromov_polynomials/results",
    "train.perceptron.arith": "gromov_arithmetic/results/arith",
    "train.transformer.sketched": "active_rank/results_fine",
    "train.transformer.extended": "dimension_recovery/results/extended",
}


def baseline_path(experiment: str, name: str) -> Optional[Path]:
    """The archived file an experiment's output should be compared against."""
    rel = BASELINE.get(experiment, {}).get(name)
    if rel is not None:
        path = archive_root() / rel
        return path if path.exists() else None
    directory = BASELINE_DIRS.get(experiment)
    if directory:
        path = archive_root() / directory / name
        return path if path.exists() else None
    return None


def baseline_names(experiment: str) -> List[str]:
    return sorted(BASELINE.get(experiment, {}))


def missing() -> List[Tuple[str, str]]:
    """Archived files the mapping names that are not on disk.

    Two are known absent and cannot be recovered without a GPU: the sketches behind
    appendix J and behind the fine windows of appendix H were never kept.
    """
    gone = []
    for experiment, files in BASELINE.items():
        for name, rel in files.items():
            if not (archive_root() / rel).exists():
                gone.append((experiment, name))
    return gone


def bootstrap(experiments: Iterable[str], mark: str = "archived") -> Dict[str, dict]:
    """Copy archived files into `data/`, marked as not yet regenerated.

    This is a bridge, not a mode to stay in: it puts the article's existing numbers where
    the figures and the table check expect to find them, so both work before anything has
    been re-run. Each manifest entry says `source: archived`, and promoting a real run
    over it clears the mark.
    """
    import shutil

    from .store import data_root, load_manifest, write_manifest
    from .provenance import sha256

    manifest = load_manifest()
    seeded: Dict[str, dict] = {}
    for experiment in experiments:
        for name, rel in BASELINE.get(experiment, {}).items():
            source = archive_root() / rel
            if not source.exists():
                continue
            target_dir = data_root() / experiment
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / name
            shutil.copy2(source, target)
            key = target.relative_to(data_root()).as_posix()
            entry = {
                "experiment": experiment,
                "source": mark,
                "archived_path": rel,
                "command": f"python -m actdim run {experiment}",
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
            }
            manifest.setdefault("files", {})[key] = entry
            seeded[key] = entry
    write_manifest(manifest)
    return seeded
