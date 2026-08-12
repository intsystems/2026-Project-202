"""Phase 7: the confound-free test of the Phase 6 coupling signal.

Phase 6 found that `train_loss` <-> `weight_norm` coupling separates runs with a delayed
transition (median z ~1.0-1.15) from runs without (2.87-5.61), with no overlap. Two
reasons not to believe it yet:

* the *within-run* time course shows no move at t_gen, in any individual run;
* the between-run split coincides exactly with configuration -- the low group is
  {fraction=0.3, wd=1} and the high group is {fraction != 0.3} union {wd=0}. That is the
  confound that already destroyed the weight-norm signal and the function-space velocity
  signal in this project.

So this holds configuration *fixed*: eight runs of one config (mod_wd1 -- same task, same
fraction, same weight decay, same schedule), varying only the split and init seeds, which
the 30-cell sweep showed yield gaps from 1290 to 5600. Within this set nothing but the
seed differs, so a correlation between coupling and gap length cannot be explained by data
quantity or regularization.

Pre-registered reading, so the result cannot be reinterpreted after the fact:

* **strong negative correlation** (coupling low when the gap is long) -> the Phase 6
  signal is about the transition, and is a validation-free grokking statistic;
* **no correlation** -> Phase 6 tracked the configuration, exactly like its predecessors,
  and should be reported as a third instance of the same confound.

    python phase7_confound_free.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccm import cross_map_skill                             # noqa: E402
from surrogates import surrogate_test                       # noqa: E402

HERE = Path(__file__).resolve().parent
CONF = HERE / "results" / "conf"
WINDOW, STRIDE, E, TAU, N_SURROGATES, EPS, RUNS_OK = 300, 100, 3, 1, 19, 0.05, 5


def sustained(steps, values, threshold, runs=RUNS_OK):
    ok = np.asarray(values) >= threshold
    for i in range(len(ok) - runs + 1):
        if ok[i:i + runs].all():
            return int(steps[i])
    return None


def main():
    rows = []
    for path in sorted(CONF.glob("s*_i*.csv")):
        match = re.match(r"s(\d+)_i(\d+)", path.stem)
        frame = pd.read_csv(path)
        steps = frame["step"].to_numpy()
        t_mem = sustained(steps, frame["train_acc"].to_numpy(), 1 - EPS)
        t_gen = sustained(steps, frame["val_acc"].to_numpy(), 1 - EPS)
        if t_mem is None or t_gen is None:
            print(f"  {path.stem:<9} did not both memorize and generalize, skipped")
            continue
        gap = t_gen - t_mem

        loss = frame["train_loss"].to_numpy(dtype=float)
        norm = frame["weight_norm"].to_numpy(dtype=float)
        zs, zs_plateau = [], []
        for lo in range(0, len(frame) - WINDOW + 1, STRIDE):
            a, b = loss[lo:lo + WINDOW], norm[lo:lo + WINDOW]
            if a.std() == 0 or b.std() == 0:
                continue
            result = surrogate_test(
                b, lambda target: cross_map_skill(a, target, E=E, tau=TAU),
                n_surrogates=N_SURROGATES, kind="iaaft", seed=0)
            zs.append(result.z_score)
            # The plateau is where the article's claim lives: after memorization,
            # before generalization.
            right = int(steps[lo + WINDOW - 1])
            if t_mem <= right < t_gen:
                zs_plateau.append(result.z_score)

        rows.append({"run": path.stem, "split": int(match.group(1)),
                     "init": int(match.group(2)), "t_mem": t_mem, "t_gen": t_gen,
                     "gap": gap, "z_median": float(np.median(zs)),
                     "z_plateau": float(np.median(zs_plateau)) if zs_plateau else np.nan,
                     "n_plateau_windows": len(zs_plateau)})
        print(f"  {path.stem:<9} gap={gap:>5}  z_median={rows[-1]['z_median']:+6.2f}  "
              f"z_plateau={rows[-1]['z_plateau']:+6.2f} "
              f"({len(zs_plateau)} windows)", flush=True)

    table = pd.DataFrame(rows).sort_values("gap")
    table.to_csv(HERE / "results" / "phase7_confound_free.csv", index=False)

    print("\n=== configuration is identical; only the seeds differ ===")
    print(table[["run", "gap", "z_median", "z_plateau"]].to_string(index=False))

    if len(table) >= 4:
        for column in ("z_median", "z_plateau"):
            sub = table.dropna(subset=[column])
            if len(sub) < 4:
                continue
            pearson = np.corrcoef(sub.gap, sub[column])[0, 1]
            ranks = (sub.gap.rank(), sub[column].rank())
            spearman = np.corrcoef(*ranks)[0, 1]
            print(f"\ncorr(gap, {column}): pearson {pearson:+.3f}  spearman {spearman:+.3f}"
                  f"  (n={len(sub)})")
        print("\nPre-registered: a strong negative correlation means the signal tracks the")
        print("transition; none means it tracked the configuration, as in Phases 2 and 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
