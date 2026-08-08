"""Empirical dynamic modeling toolkit for neural-network training logs.

Reconstructs the phase space of the optimizer from 1D scalar logs (Takens /
Stark delay embedding) and tracks the effective dimensionality of the resulting
attractor, which collapses at the onset of grokking.
"""

from .dimension import (
    CORRECTIONS,
    ESTIMATORS,
    Estimator,
    cao_method,
    embedding_dimension_scan,
    estimate_E_cao,
    estimate_E_fnn,
    estimate_E_mle,
    estimate_E_mle_mg,
    estimate_E_mle_mg_theiler,
    estimate_E_mle_mg_theiler_acf,
    estimate_E_roughness,
    estimate_E_simplex,
    estimate_E_svd,
    estimate_E_svd_entropy,
    false_nearest_neighbors,
    identifiability_ratio,
    local_roughness,
    local_svd_dimension,
    mle_intrinsic_dimension,
    mle_log_ratio_sums,
    resolve_theiler_window,
    simplex_projection,
)
from .embedding import (
    TAU_SELECTORS,
    autocorrelation_time,
    delay_embedding,
    delayed_mutual_information,
    first_local_minimum,
    select_tau_dmi,
    select_tau_fixed,
)
from .plots import (
    plot_dimension_vs_accuracy,
    plot_presentation_panels,
    plot_smoothed_accuracy,
    plot_smoothed_dynamics,
)
from .sliding import DimensionTrace, grokking_step, load_logs, sliding_dimension

__all__ = [
    "CORRECTIONS",
    "ESTIMATORS",
    "TAU_SELECTORS",
    "DimensionTrace",
    "Estimator",
    "autocorrelation_time",
    "cao_method",
    "delay_embedding",
    "embedding_dimension_scan",
    "delayed_mutual_information",
    "estimate_E_cao",
    "estimate_E_fnn",
    "estimate_E_mle",
    "estimate_E_mle_mg",
    "estimate_E_mle_mg_theiler",
    "estimate_E_mle_mg_theiler_acf",
    "estimate_E_roughness",
    "estimate_E_simplex",
    "estimate_E_svd",
    "estimate_E_svd_entropy",
    "false_nearest_neighbors",
    "first_local_minimum",
    "grokking_step",
    "identifiability_ratio",
    "load_logs",
    "local_roughness",
    "local_svd_dimension",
    "mle_intrinsic_dimension",
    "mle_log_ratio_sums",
    "resolve_theiler_window",
    "plot_dimension_vs_accuracy",
    "plot_presentation_panels",
    "plot_smoothed_accuracy",
    "plot_smoothed_dynamics",
    "select_tau_dmi",
    "select_tau_fixed",
    "simplex_projection",
    "sliding_dimension",
]
