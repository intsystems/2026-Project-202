"""Phase 1: do the embedding theorems' preconditions hold on these logs?

Takens and Stark reconstruct a compact invariant set that the orbit *returns to*. This
measures whether the logs contain such a set, before any dimension or causality claim is
made about them. The estimator is calibrated in ``test_edm_validation.py``: a Lorenz
trajectory holds a recurrence plateau (ratio 0.93), a monotone ramp decays to exactly
zero (ratio 0.00).

Every series is examined twice:

``raw``
    As logged. A training loss falls monotonically, and a monotone series cannot recur --
    so a near-zero ratio here is expected and is not yet an interesting finding.
``detrended``
    Minus a centred moving average, i.e. the fluctuations about the slow trend. This is
    the charitable reading: the claim EDM would need is that the *fluctuations* carry
    low-dimensional dynamics even while the mean drifts. If recurrence is absent here
    too, no windowing or detrending will rescue the attractor language.

    python phase1_preconditions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast import recurrence_stats                       # noqa: E402

CODE = Path(__file__).resolve().parent.parent
POISONED = CODE / "poisoned_batch"
GROKKING = CODE / "grokking_analysis" / "grokking_logs"

E, TAU, SMOOTH = 5, 1, 51


def detrend(x, window=SMOOTH):
    """Subtract a centred moving average -- keep the fluctuations, drop the drift."""
    x = np.asarray(x, dtype=float)
    if len(x) < window * 2:
        return x - x.mean()
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="same")
    edge = window // 2
    trend[:edge], trend[-edge:] = trend[edge], trend[-edge - 1]
    return x - trend


def datasets():
    """(family, name, {column: series}) for every log we can test."""
    out = []
    for path in sorted((POISONED / "folder_for_raw_series").glob("*.csv")):
        d = pd.read_csv(path)
        name = path.stem.replace("resnet_cifar_", "").replace("_logs", "")
        out.append(("poisoned", name, d))
    for path in sorted((POISONED / "ghost_raw_series_logs").glob("*.csv")):
        d = pd.read_csv(path)
        name = path.stem.replace("resnet_cifar_", "").replace("_logs", "")
        out.append(("ghost", name, d))
    for path in sorted(GROKKING.glob("*.csv")):
        d = pd.read_csv(path)
        name = path.stem.replace("grokking_modular_addition_logs", "mod").replace("_", " ")[:34]
        out.append(("grokking", name, d))
    return out


def main():
    rows = []
    for family, name, frame in datasets():
        columns = [c for c in ("poison_fraction", "train_loss", "val_loss", "weight_norm")
                   if c in frame.columns]
        for column in columns:
            series = frame[column].to_numpy(dtype=float)
            series = series[np.isfinite(series)]
            if len(series) < 500 or series.std() == 0:
                continue
            for variant, values in (("raw", series), ("detrended", detrend(series))):
                stats = recurrence_stats(values, E=E, tau=TAU)
                rows.append({
                    "family": family, "run": name, "column": column, "variant": variant,
                    "rr_plain": stats["rr_plain"], "rr_long": stats["rr_long"],
                    "ratio": stats["ratio"],
                })

    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "phase1_recurrence.csv", index=False)

    print("Recurrence ratio = RR(longest exclusion) / RR(no exclusion).")
    print("  ~1  the orbit genuinely returns  -> an invariant set exists, Takens applies")
    print("  ~0  every close pair was adjacent in time -> transient, Takens does not\n")

    for variant in ("raw", "detrended"):
        sub = table[table.variant == variant]
        print(f"=== {variant} ===")
        pivot = sub.pivot_table(index=["family", "run"], columns="column", values="ratio")
        print(pivot.round(3).to_string())
        print()

    print("=== summary by family (detrended) ===")
    det = table[table.variant == "detrended"]
    print(det.groupby("family").ratio.describe()[["count", "mean", "min", "max"]].round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
