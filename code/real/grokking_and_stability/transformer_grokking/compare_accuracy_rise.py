"""Compare MG@weight norm before, during, and after the rise in validation accuracy."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
from actdim.estimator.mle import estimate
from synthetic.digits_validation.estimator_calibration.settings import eight_direction


LOGS = HERE / "logs"
EXAMPLE = LOGS / "mod_wd1_train.csv"
PRIMARY_BAND = (0.10, 0.90)
SENSITIVITY_BAND = (0.20, 0.80)
PERSISTENCE = 5
MIN_WINDOWS = 3
# These are the parameters of the saved grokking figure, including its temporal exclusion.
MG_CONFIG = eight_direction(max_E=10, tau=1, k_neighbors=20, theiler="autocorr",
                            window=60, stride=10)


def accuracy_rise(frame, band=PRIMARY_BAND):
    """First logged steps starting five consecutive readings at or above each cutoff."""
    validation = frame.val_acc.to_numpy()
    steps = frame.step.to_numpy()
    if len(steps) < PERSISTENCE:
        return None, None
    windows = np.lib.stride_tricks.sliding_window_view(validation, PERSISTENCE)
    bounds = []
    for cutoff in band:
        crossings = np.flatnonzero((windows >= cutoff).all(axis=1))
        bounds.append(int(steps[crossings[0]]) if crossings.size else None)
    return tuple(bounds)


def weight_norm_mg(frame):
    """Compute MG and keep the actual start and end of every valid window."""
    values = frame.weight_norm.to_numpy(dtype=float)
    steps = frame.step.to_numpy()
    width = MG_CONFIG.window
    rows = []
    for start in range(0, len(values) - width + 1, MG_CONFIG.stride):
        end = start + width
        result = estimate(values[start:end], MG_CONFIG)
        if not result.degenerate and np.isfinite(result.MG):
            rows.append((steps[start], steps[end - 1], result.MG))
    return pd.DataFrame(rows, columns=["left_step", "right_step", "MG"])


def phase_medians(windows, memorized, start, end):
    duration = end - start
    intervals = {
        "before": (max(memorized, start - duration), start),
        "during": (start, end),
        "after": (end, end + duration),
    }
    row = {}
    for phase, (left, right) in intervals.items():
        # Use complete windows to avoid mixing measurements across the phase boundaries.
        scores = windows.loc[(windows.left_step >= left) & (windows.right_step <= right), "MG"]
        row[f"{phase}_windows"] = len(scores)
        row[f"{phase}_median"] = scores.median()
    if min(row[f"{phase}_windows"] for phase in intervals) >= MIN_WINDOWS:
        row["status"] = "compared"
        row["higher_than_both"] = row["during_median"] > max(row["before_median"], row["after_median"])
    else:
        row["status"] = "insufficient windows"
    return row


def compare_run(path, family):
    frame = pd.read_csv(path)
    run = path.stem.split("__")[-1].removesuffix("_train")
    trained = frame.loc[frame.train_acc >= 0.95, "step"]
    memorized = int(trained.iloc[0]) if not trained.empty else None
    windows = None
    for band in (PRIMARY_BAND, SENSITIVITY_BAND):
        start, end = accuracy_rise(frame, band)
        row = {
            "family": family, "run": run, "input": path.relative_to(CODE).as_posix(),
            "lower_accuracy": band[0], "upper_accuracy": band[1],
            "memorized_step": memorized, "rise_start": start, "rise_end": end,
        }
        if start is None or end is None:
            row["status"] = "rise not completed"
        elif memorized is None or not memorized < start < end:
            row["status"] = "rise overlaps training fit"
        else:
            if windows is None:
                windows = weight_norm_mg(frame)
            row.update(phase_medians(windows, memorized, start, end))
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CODE / "runs/redrawn/grokking_accuracy_rise.csv")
    args = parser.parse_args()
    perceptron = HERE.parent / "perceptron_grokking/logs"
    inputs = {
        # The example file and sketched__mod_wd1 are two recordings of the same seed and
        # configuration. Use the recording that reproduces the paper figure.
        "transformer": [EXAMPLE, *[path for path in sorted(LOGS.glob("sketched__*_train.csv"))
                                  if path.name != "sketched__mod_wd1_train.csv"],
                        *sorted(LOGS.glob("extended__*_train.csv"))],
        "perceptron": [*sorted(perceptron.glob("arithmetic__*_train.csv")),
                       *sorted(perceptron.glob("polynomial__*_train.csv"))],
    }
    report = pd.DataFrame(row for family, paths in inputs.items()
                          for path in paths for row in compare_run(path, family))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False, float_format="%.9g")
    compared = report[report.status == "compared"].copy()
    compared["higher_than_both"] = compared["higher_than_both"].astype(int)
    print(compared.groupby(["family", "lower_accuracy", "upper_accuracy"])["higher_than_both"]
          .agg(higher="sum", compared="size").to_string())
    print(f"Saved run-by-run results to {args.output}")


if __name__ == "__main__":
    main()
