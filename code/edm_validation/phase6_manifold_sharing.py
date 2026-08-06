"""Phase 6: does the *coupling* between two training-side logs change at grokking?

This is the article's ambition -- read the internal transition off cheap 1-D logs --
attempted with the one tool that has survived validation here.

**Why this question and not the article's.** Everything that failed in this project failed
the same way: a statistic computed on a *single* transient series, where no invariant set
exists, returns a number that reflects smoothness rather than dynamics. Cross mapping
asks a different question -- do two observables lie on a *common* manifold? -- and that is
well-posed on a transient, needs no recurrence of either series alone, and is exactly what
Takens' theorem licenses: two smooth observations of one state should each reconstruct it,
hence predict each other.

**The observables are both training-side**: `train_loss` and `weight_norm`. No validation
data is used anywhere, which is the property the article claimed and never delivered.

**The prediction.** During memorization under weight decay the two are driven by different
processes -- the loss is pinned near zero by an interpolating solution while the norm
decays under regularization -- so they need not share a manifold. If generalization is a
reorganization of the function, both respond to it and their coupling should change. Runs
that never generalize give the control.

Nothing here is claimed without the two nulls of `ccm.py`, and the statistic is tracked in
*causal* windows: each value uses only data up to its right edge.

    python phase6_manifold_sharing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccm import cross_map_skill                             # noqa: E402
from surrogates import surrogate_test                       # noqa: E402

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE / "prediction_improved" / "results"

# 4 runs that generalize, 3 that never do -- same architecture, same task, same optimizer.
RUNS = {
    "grok": 13700, "grok_seed1": 2990, "grok_seed2": 3140, "nogap": 780,
    "wd0": None, "lowdata15": None, "lowdata20": None,
}

WINDOW, STRIDE, E, TAU, N_SURROGATES = 300, 100, 3, 1, 19


def windows(n, width=WINDOW, stride=STRIDE):
    return [(s, s + width) for s in range(0, n - width + 1, stride)]


def coupling(a, b):
    """Excess cross-map skill over an IAAFT ensemble, in surrogate sigmas.

    Reported as a z-score rather than raw rho because rho is not comparable across
    windows: a window where both series are smooth gives a high rho for any pair. The
    surrogate ensemble absorbs exactly that, so what is left is coupling.
    """
    result = surrogate_test(b, lambda target: cross_map_skill(a, target, E=E, tau=TAU),
                            n_surrogates=N_SURROGATES, kind="iaaft", seed=0)
    return result.z_score, result.statistic, result.p_value


def main():
    rows = []
    for name, t_gen in RUNS.items():
        path = RESULTS / f"{name}_train.csv"
        if not path.exists():
            print(f"  {name:<11} missing {path.name}")
            continue
        frame = pd.read_csv(path)
        loss = frame["train_loss"].to_numpy(dtype=float)
        norm = frame["weight_norm"].to_numpy(dtype=float)
        steps = frame["step"].to_numpy()

        print(f"\n=== {name}  (t_gen={t_gen}) ===", flush=True)
        for lo, hi in windows(len(frame)):
            a, b = loss[lo:hi], norm[lo:hi]
            if a.std() == 0 or b.std() == 0:
                continue
            z, rho, p = coupling(a, b)
            right = int(steps[hi - 1])            # causal label: last step seen
            rows.append({"run": name, "t_gen": t_gen, "step": right,
                         "z": z, "rho": rho, "p": p,
                         "generalizes": t_gen is not None,
                         "after_gen": (t_gen is not None and right >= t_gen)})
            mark = ""
            if t_gen is not None:
                mark = " <-- after generalization" if right >= t_gen else ""
            print(f"  step<={right:>6}  rho={rho:+.3f}  z={z:+7.2f}  p={p:.3f}{mark}",
                  flush=True)

    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "phase6_manifold_sharing.csv", index=False)

    print("\n\n=== does coupling separate the two families? ===")
    print("median z by run:")
    print(table.groupby("run").z.median().round(2).to_string())

    print("\nfraction of windows passing the surrogate null (p<=0.05):")
    print(table.assign(sig=table.p <= 0.05).groupby("run").sig.mean().round(3).to_string())

    grokking = table[table.generalizes]
    if not grokking.empty:
        print("\nwithin generalizing runs, before vs after t_gen (median z):")
        print(grokking.groupby("after_gen").z.median().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
