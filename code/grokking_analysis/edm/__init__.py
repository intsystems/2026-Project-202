"""Empirical dynamic modeling toolkit for neural-network training logs.

Reconstructs the phase space of the optimizer from 1D scalar logs (Takens /
Stark delay embedding) and tracks the effective dimensionality of the resulting
attractor, which collapses at the onset of grokking.
"""

from .dimension import (
    ESTIMATORS,
    cao_method,
    estimate_E_cao,
    estimate_E_fnn,
    estimate_E_mle,
    estimate_E_simplex,
    false_nearest_neighbors,
    mle_intrinsic_dimension,
    simplex_projection,
)
from .embedding import (
    TAU_SELECTORS,
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
    "ESTIMATORS",
    "TAU_SELECTORS",
    "DimensionTrace",
    "cao_method",
    "delay_embedding",
    "delayed_mutual_information",
    "estimate_E_cao",
    "estimate_E_fnn",
    "estimate_E_mle",
    "estimate_E_simplex",
    "false_nearest_neighbors",
    "first_local_minimum",
    "grokking_step",
    "load_logs",
    "mle_intrinsic_dimension",
    "plot_dimension_vs_accuracy",
    "plot_presentation_panels",
    "plot_smoothed_accuracy",
    "plot_smoothed_dynamics",
    "select_tau_dmi",
    "select_tau_fixed",
    "simplex_projection",
    "sliding_dimension",
]
