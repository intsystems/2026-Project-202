"""Sliding-window estimation of the effective dimensionality d_hat(t)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .dimension import ESTIMATORS
from .embedding import TAU_SELECTORS

E_FLOOR, E_CEILING = 1.0, 30.0
"""Estimates are clipped to this range before plotting (guards against outliers)."""

LABEL_POSITIONS = {"center": lambda w: w // 2, "right": lambda w: w - 1, "left": lambda w: 0}
"""Which step inside the window the estimate is attributed to.

``center`` reproduces the original notebooks, but a window centred on ``t`` contains
data up to ``t + W/2``: as an *early warning* signal it peeks 750-1500 optimization
steps into the future on these logs. ``right`` labels each estimate with the last
step it actually saw, which is the only causal choice for a predictor.
"""


@dataclass
class DimensionTrace:
    """Result of a sliding-window sweep over a single scalar log."""

    steps: np.ndarray            # step each estimate is pinned to (see LABEL_POSITIONS)
    dimension: np.ndarray        # d_hat(t)
    tau: np.ndarray              # delay used in each window
    metric: str = ""
    method: str = ""             # long name, for axis labels and titles
    label: str = ""              # short name, for legends
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
    tau=None,
    window_size=300,
    step_size=50,
    include_last_window=True,
    label_position="center",
    clip=(E_FLOOR, E_CEILING),
    estimator_kwargs=None,
    seed=0,
    progress=True,
):
    """Slide a window of ``window_size`` samples over ``df[target_metric]``.

    For every window the delay ``tau`` is chosen by ``tau_selector`` and the
    effective dimension by ``method``.

    ``include_last_window=False`` reproduces the original notebook loop
    ``range(0, n - window_size, step_size)``, which drops the final full window.
    ``label_position`` decides which step the estimate is pinned to -- see
    :data:`LABEL_POSITIONS`; use ``"right"`` for a causal predictor. ``clip=None``
    reports raw estimates instead of squashing them into ``[1, 30]``, and
    ``estimator_kwargs`` is forwarded to the estimator (``k_neighbors``, ``max_E``,
    ``dither``, ``degenerate``, ...). An explicit ``tau`` overrides ``tau_selector``
    and holds the embedding geometry fixed across windows.
    """
    if label_position not in LABEL_POSITIONS:
        raise ValueError(f"unknown label_position '{label_position}'. "
                         f"Expected one of {tuple(LABEL_POSITIONS)}")
    label_offset = LABEL_POSITIONS[label_position](window_size)
    if target_metric not in df.columns:
        raise KeyError(f"log has no column '{target_metric}' (available: {list(df.columns)})")

    estimator = ESTIMATORS[method]
    pick_tau = TAU_SELECTORS[tau_selector]
    rng = np.random.default_rng(seed)
    estimator_kwargs = dict(estimator_kwargs or {})

    frame = df.dropna(subset=[target_metric])
    all_steps = frame["step"].to_numpy(dtype=np.float64)
    series = frame[target_metric].to_numpy(dtype=np.float64)

    stop = len(series) - window_size + (1 if include_last_window else 0)
    starts = range(0, max(stop, 0), step_size)

    steps, dimensions, taus = [], [], []
    for i in tqdm(starts, desc=f"{estimator.label} / {target_metric}", disable=not progress):
        window = series[i:i + window_size]
        if len(window) < window_size or np.isnan(window).any():
            continue

        window_tau = pick_tau(window) if tau is None else tau
        E = estimator.fn(window, window_tau, rng=rng, **estimator_kwargs)

        if E is None or not np.isfinite(E):
            clean_E = np.nan
        elif clip is None:
            clean_E = float(E)
        else:
            clean_E = float(np.clip(E, *clip))

        steps.append(all_steps[i + label_offset])
        dimensions.append(clean_E)
        taus.append(window_tau)

    return DimensionTrace(
        steps=np.asarray(steps, dtype=np.float64),
        dimension=np.asarray(dimensions, dtype=np.float64),
        tau=np.asarray(taus, dtype=np.float64),
        metric=target_metric,
        method=estimator.name,
        label=estimator.label,
        meta={"window_size": window_size, "step_size": step_size,
              "tau_selector": "fixed-override" if tau is not None else tau_selector,
              "label_position": label_position,
              "clip": clip, **estimator_kwargs},
    )
