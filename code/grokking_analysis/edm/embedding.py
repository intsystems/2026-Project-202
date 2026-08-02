"""Delay-coordinate embedding and time-delay selection.

Implements the ``DelayEmbedding`` operator of the paper (Sec. 3.1):

    x_t = [X(t), X(t - tau), ..., X(t - (E-1) tau)]^T  in R^E

together with the classical delayed-mutual-information heuristic for picking
the delay ``tau``.
"""

import numpy as np
from sklearn.metrics import mutual_info_score


def delay_embedding(x, m, tau):
    """Embed a 1D series into R^m with delay ``tau``.

    Returns an array of shape ``(len(x) - (m - 1) * tau, m)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n_points = len(x) - (m - 1) * tau
    if n_points <= 0:
        raise ValueError(f"Time series length {len(x)} is too short for m={m}, tau={tau}")
    return np.column_stack([x[i:i + n_points] for i in range(0, m * tau, tau)])


def delayed_mutual_information(x, max_tau=50, bins=50):
    """Mutual information between ``x(t)`` and ``x(t - tau)`` for tau = 1..max_tau."""
    x = np.asarray(x)
    taus = np.arange(1, max_tau + 1)

    if np.std(x) < 1e-8:
        return taus, np.zeros(len(taus))

    _, bin_edges = np.histogram(x, bins=bins)
    x_binned = np.digitize(x, bin_edges[:-1])

    dmi_values = [mutual_info_score(x_binned[:-tau], x_binned[tau:]) for tau in taus]
    return taus, np.array(dmi_values)


def first_local_minimum(y, abs_eps=0.01, drop_fraction=0.01):
    """Index of the first (approximate) local minimum of a decaying curve."""
    y = np.asarray(y)
    if len(y) == 0 or np.all(y == 0):
        return 0

    drop_eps = y[0] * drop_fraction
    for i in range(len(y) - 1):
        if y[i] < abs_eps:
            return i
        if y[i] - y[i + 1] < drop_eps:
            return i
    return int(np.argmin(y))


def select_tau_dmi(series, max_tau=15, bins=20):
    """Pick ``tau`` at the first local minimum of the delayed mutual information."""
    if np.std(series) < 1e-6:
        return 1
    taus, dmi_values = delayed_mutual_information(series, max_tau=max_tau, bins=bins)
    return int(taus[first_local_minimum(dmi_values, abs_eps=0.01, drop_fraction=0.01)])


def select_tau_fixed(series, value=1):
    """Constant delay. The paper's experiments use ``tau = 1``."""
    return value


TAU_SELECTORS = {
    "fixed": select_tau_fixed,
    "dmi": select_tau_dmi,
}
