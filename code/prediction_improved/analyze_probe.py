"""First look at the probe series: does the function-space signal move before grokking?

Computes the statistic of ``method.md`` -- departure from local linearity -- over short
causal windows of each projection series, and reports it against the grokking step. The
comparison that matters is treatment vs. control, not the absolute level: a locally
straight trajectory sits at roughness 0 by construction, so only a *rise* is evidence.

    python analyze_probe.py results
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import local_roughness                             # noqa: E402

WINDOW = 20          # logged rows per segment
EPS = 0.05           # the article's Definition 1 threshold


def sustained(steps, values, threshold, runs=5):
    """First step from which ``values >= threshold`` holds for ``runs`` rows."""
    ok = values >= threshold
    for i in range(len(ok) - runs + 1):
        if ok[i:i + runs].all():
            return int(steps[i])
    return None


def causal_roughness(series, window=WINDOW):
    """Right-edge-labelled roughness: index i sees only rows [i-window+1, i]."""
    out = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        out[i] = local_roughness(series[i - window + 1:i + 1])
    return out


def analyse(key, train_csv, probe_csv):
    train = pd.read_csv(train_csv)
    probe = pd.read_csv(probe_csv)
    steps = probe["step"].to_numpy()

    t_mem = sustained(train["step"].to_numpy(), train["train_acc"].to_numpy(), 1 - EPS)
    t_gen = sustained(train["step"].to_numpy(), train["val_acc"].to_numpy(), 1 - EPS)

    print(f"\n=== {key} ===")
    print(f"  t_mem = {t_mem}   t_gen = {t_gen}"
          f"{'' if t_gen is None else f'   gap = {t_gen - t_mem}'}")

    for source in ("train", "val"):
        columns = [c for c in probe.columns if c.startswith(f"{source}_p")]
        stacked = np.stack([causal_roughness(probe[c].to_numpy()) for c in columns])
        # The first WINDOW-1 rows have no full window yet and are NaN in every
        # projection; median over them is undefined, so skip rather than warn.
        rough = np.full(stacked.shape[1], np.nan)
        rough[WINDOW - 1:] = np.nanmedian(stacked[:, WINDOW - 1:], axis=0)
        velocity = probe[f"{source}_velocity"].to_numpy()

        valid = ~np.isnan(rough)
        if t_gen is not None:
            pre = valid & (steps < t_gen)
            post = valid & (steps >= t_gen)
        else:
            half = len(steps) // 2
            pre = valid & (np.arange(len(steps)) < half)
            post = valid & (np.arange(len(steps)) >= half)

        print(f"  [{source}] roughness: overall median {np.nanmedian(rough[valid]):.4f}, "
              f"peak {np.nanmax(rough[valid]):.4f} at step {steps[valid][np.nanargmax(rough[valid])]}")
        print(f"  [{source}]   pre-{'gen' if t_gen else 'half'} median {np.nanmedian(rough[pre]):.4f}"
              f"   post median {np.nanmedian(rough[post]):.4f}")
        print(f"  [{source}] velocity : median {np.nanmedian(velocity):.4e}, "
              f"max {np.nanmax(velocity):.4e}")
    return t_gen


def main(results_dir="results"):
    root = Path(results_dir)
    runs = {
        "mod_wd1 (grokks)": (
            "grokking_modular_addition_logs_to_flat_grokking_with_stochastic.csv",
            "mod_wd1_probe.csv"),
        "mod_wd0 (control)": (
            "grokking_modular_addition_logs_to_flat_grokking_with_stochastic_without_wight_decay.csv",
            "mod_wd0_probe.csv"),
    }
    for key, (train_csv, probe_csv) in runs.items():
        if not (root / probe_csv).exists():
            print(f"skipping {key}: {root / probe_csv} not found")
            continue
        analyse(key, root / train_csv, root / probe_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
