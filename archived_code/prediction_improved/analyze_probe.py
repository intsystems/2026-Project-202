"""Compare the function-space signal across the control suite.

Reports, per run: when it memorized and generalized, and the level and time course of
the logit velocity -- the statistic that separated the first two runs (``method.md`` §7).
The comparison that decides H1 vs. H2 is ``lowdata*`` against ``grok`` and ``wd0``:
weight decay is identical in ``grok`` and ``lowdata*``, so if velocity tracks weight
decay they will look alike, and if it tracks impending generalization ``lowdata*`` will
look like ``wd0``.

Roughness is reported too, but §7 already showed it saturating at its ceiling on this
observable; it is kept only so that remains visible rather than assumed.

    python analyze_probe.py results
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import local_roughness                             # noqa: E402

WINDOW = 20          # logged rows per roughness segment
EPS = 0.05           # the article's Definition 1 threshold
RUNS = 5             # consecutive rows required to call an event sustained


def sustained(steps, values, threshold, runs=RUNS):
    ok = np.asarray(values) >= threshold
    for i in range(len(ok) - runs + 1):
        if ok[i:i + runs].all():
            return int(steps[i])
    return None


def causal_roughness(series, window=WINDOW):
    out = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        out[i] = local_roughness(series[i - window + 1:i + 1])
    return out


def discover(root):
    """Pair ``<tag>_probe.csv`` with ``<tag>_train.csv`` in ``root``."""
    pairs = []
    for probe_csv in sorted(root.glob("*_probe.csv")):
        tag = probe_csv.name[: -len("_probe.csv")]
        train_csv = root / f"{tag}_train.csv"
        if train_csv.exists():
            pairs.append((tag, train_csv, probe_csv))
        else:
            print(f"  (skipping {tag}: no {train_csv.name})")
    return pairs


def analyse(tag, train_csv, probe_csv):
    train = pd.read_csv(train_csv)
    probe = pd.read_csv(probe_csv)
    steps = probe["step"].to_numpy()

    t_mem = sustained(train["step"].to_numpy(), train["train_acc"].to_numpy(), 1 - EPS)
    t_gen = sustained(train["step"].to_numpy(), train["val_acc"].to_numpy(), 1 - EPS)

    velocity = probe["val_velocity"].to_numpy()
    finite = velocity[np.isfinite(velocity)]

    # Level late in the run, well after memorization: this is what separates the runs.
    late = np.isfinite(velocity) & (steps > steps.max() * 0.5)
    columns = [c for c in probe.columns if c.startswith("val_p")]
    stacked = np.stack([causal_roughness(probe[c].to_numpy()) for c in columns])
    rough = np.nanmedian(stacked[:, WINDOW - 1:], axis=0)

    return {
        "tag": tag,
        "t_mem": t_mem,
        "t_gen": t_gen,
        "gap": None if (t_gen is None or t_mem is None) else t_gen - t_mem,
        "final_val_acc": float(train["val_acc"].iloc[-20:].median()),
        "vel_median": float(np.median(finite)) if len(finite) else float("nan"),
        "vel_late": float(np.median(velocity[late])) if late.any() else float("nan"),
        "vel_last": float(np.median(velocity[-50:])),
        "rough_median": float(np.nanmedian(rough)),
    }


def main(results_dir="results"):
    root = Path(results_dir)
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1

    rows = [analyse(*pair) for pair in discover(root)]
    if not rows:
        print(f"no <tag>_train.csv / <tag>_probe.csv pairs in {root}")
        return 1

    rows.sort(key=lambda r: -r["vel_late"])
    header = (f"{'run':<12} {'t_mem':>7} {'t_gen':>7} {'gap':>7} {'val_acc':>8} "
              f"{'vel(med)':>10} {'vel(late)':>10} {'vel(end)':>10} {'rough':>7}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['tag']:<12} {str(r['t_mem']):>7} {str(r['t_gen']):>7} "
              f"{str(r['gap']):>7} {r['final_val_acc']:>8.3f} "
              f"{r['vel_median']:>10.3e} {r['vel_late']:>10.3e} {r['vel_last']:>10.3e} "
              f"{r['rough_median']:>7.3f}")

    print("\nvel(late) = median velocity over the second half of training.")
    print("H1 (tracks generalization): lowdata* should sit near wd0.")
    print("H2 (tracks weight decay)  : lowdata* should sit near grok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
