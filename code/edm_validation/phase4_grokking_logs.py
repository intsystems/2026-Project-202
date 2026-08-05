"""Phase 4: does the validated CCM machinery say anything about our own runs?

Phase 3 established that cross mapping recovers a genuine driver from a 1-D training-loss
log, with no false positives on drivers that were logged but never applied. That result
is about *externally driven* training, which is the setting Stark's theorem covers.

The grokking runs are the opposite case: nothing external drives them, and Phase 1 showed
their raw trajectories do not recur at all. So the honest question is not "does CCM find
the cause of grokking" -- there is no injected cause to find -- but the two that the
machinery can actually answer:

1. **Between logged metrics of the same run**, does cross mapping report coupling, and is
   it symmetric? `weight_norm` and `train_loss` are two observations of one system, so
   under Takens each should predict the other; strong asymmetry would instead indicate
   one is a driver of the other in the CCM sense.
2. **Is any of it distinguishable from a shared trend?** Every pair of series in a
   training run declines together. The surrogate and shift nulls are exactly the controls
   for that, and the ghost runs in Phase 3 showed what "no coupling" looks like when the
   trend is shared: rho ~ 0.00.

The expected answer for (1) is *not* that CCM discovers something profound. It is a
specificity check: a method that reports strong coupling between every pair of metrics in
every run, including where none should exist, would be reporting the trend and would be
worthless. Phase 3 earned the right to run this test; this run is what makes the Phase 3
positives credible rather than lucky.

    python phase4_grokking_logs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccm import ccm_test                                    # noqa: E402

CODE = Path(__file__).resolve().parent.parent
LOGS = [
    ("mod_wd1 (grokks)", CODE / "prediction_improved" / "results" / "grok_train.csv"),
    ("mod_wd0 (control)", CODE / "prediction_improved" / "results" / "wd0_train.csv"),
    ("lowdata15 (no gen.)", CODE / "prediction_improved" / "results" / "lowdata15_train.csv"),
    ("s5_wd1", CODE / "grokking_analysis" / "grokking_logs"
     / "grokking_modular_addition_logs_S_5_with_stochastic.csv"),
]
PAIRS = [("weight_norm", "train_loss"), ("train_loss", "weight_norm"),
         ("train_loss", "val_loss"), ("val_loss", "train_loss")]
LIBRARY_SIZES = (20, 40, 80, 160, 320, 640, 1200)
E, TAU, N_SURROGATES = 3, 1, 39


def main():
    rows = []
    for name, path in LOGS:
        if not path.exists():
            print(f"  {name:<22} missing {path.name}")
            continue
        frame = pd.read_csv(path)
        print(f"\n=== {name} ({len(frame)} rows) ===")
        for embed_col, target_col in PAIRS:
            if embed_col not in frame or target_col not in frame:
                continue
            embed = frame[embed_col].to_numpy(dtype=float)
            target = frame[target_col].to_numpy(dtype=float)
            if embed.std() == 0 or target.std() == 0:
                continue

            result = ccm_test(embed, target, E=E, tau=TAU, library_sizes=LIBRARY_SIZES,
                              surrogate_kind="iaaft", n_surrogates=N_SURROGATES, seed=0)
            shift = ccm_test(embed, target, E=E, tau=TAU, library_sizes=LIBRARY_SIZES,
                             surrogate_kind="shift", n_surrogates=N_SURROGATES, seed=0)
            rows.append({"run": name, "embed": embed_col, "target": target_col,
                         "rho": result["rho_max"], "gain": result["gain"],
                         "p_iaaft": result["p"], "p_shift": shift["p"],
                         "detected": bool(result["detected"] and shift["detected"])})
            print(f"  {embed_col:>11} xmap {target_col:<11} rho={result['rho_max']:+.3f} "
                  f"gain={result['gain']:+.3f} p_iaaft={result['p']:.3f} "
                  f"p_shift={shift['p']:.3f} "
                  f"{'COUPLED' if rows[-1]['detected'] else ''}", flush=True)

    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "phase4_grokking_ccm.csv", index=False)

    print("\n=== how often does it fire? ===")
    if not table.empty:
        print(f"  {table.detected.sum()} of {len(table)} directed pairs pass both nulls")
        print("\n  by run:")
        print(table.groupby("run").detected.agg(["sum", "count"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
