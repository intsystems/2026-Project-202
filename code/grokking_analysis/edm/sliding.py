"""Sliding-window estimation of the effective dimensionality d_hat(t)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .dimension import ESTIMATORS
from .embedding import TAU_SELECTORS

E_FLOOR, E_CEILING = 1.0, 30.0
"""Estimates are clipped to this range before plotting (guards against outliers)."""


@dataclass
class DimensionTrace:
    """Result of a sliding-window sweep over a single scalar log."""

    steps: np.ndarray            # optimization step at the centre of each window
    dimension: np.ndarray        # d_hat(t)
    tau: np.ndarray              # delay used in each window
    metric: str = ""
    method: str = ""
    meta: dict = field(default_factory=dict)


def load_logs(csv_path, required=("step", "train_acc", "val_acc")):
    """Read a training-log CSV, coerce every column to numeric, drop bad rows."""
    df = pd.read_csv(csv_path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} lacks required column(s): {missing}")
    return df.dropna(subset=list(required)).reset_index(drop=True)


def grokking_step(df, threshold=0.95):
    """First step at which validation accuracy reaches ``threshold`` (None if never)."""
    grokked = df[df["val_acc"] >= threshold]
    return None if grokked.empty else float(grokked["step"].iloc[0])


def sliding_dimension(
    df,
    target_metric="weight_norm",
    method="mle",
    tau_selector="fixed",
    window_size=300,
    step_size=50,
    include_last_window=True,
    seed=0,
    progress=True,
):
    """Slide a window of ``window_size`` samples over ``df[target_metric]``.

    For every window the delay ``tau`` is chosen by ``tau_selector`` and the
    effective dimension by ``method``; the estimate is attributed to the step at
    the centre of the window.

    ``include_last_window=False`` reproduces the original notebook loop
    ``range(0, n - window_size, step_size)``, which drops the final full window.
    """
    if target_metric not in df.columns:
        raise KeyError(f"log has no column '{target_metric}' (available: {list(df.columns)})")

    method_name, estimator = ESTIMATORS[method]
    pick_tau = TAU_SELECTORS[tau_selector]
    rng = np.random.default_rng(seed)

    frame = df.dropna(subset=[target_metric])
    all_steps = frame["step"].to_numpy(dtype=np.float64)
    series = frame[target_metric].to_numpy(dtype=np.float64)

    stop = len(series) - window_size + (1 if include_last_window else 0)
    starts = range(0, max(stop, 0), step_size)

    steps, dimensions, taus = [], [], []
    for i in tqdm(starts, desc=f"{method_name} / {target_metric}", disable=not progress):
        window = series[i:i + window_size]
        if len(window) < window_size or np.isnan(window).any():
            continue

        tau = pick_tau(window)
        E = estimator(window, tau, rng=rng)

        if E is None or not np.isfinite(E):
            clean_E = np.nan
        else:
            clean_E = float(np.clip(E, E_FLOOR, E_CEILING))

        steps.append(all_steps[i + window_size // 2])
        dimensions.append(clean_E)
        taus.append(tau)

    return DimensionTrace(
        steps=np.asarray(steps, dtype=np.float64),
        dimension=np.asarray(dimensions, dtype=np.float64),
        tau=np.asarray(taus, dtype=np.float64),
        metric=target_metric,
        method=method_name,
        meta={"window_size": window_size, "step_size": step_size, "tau_selector": tau_selector},
    )
