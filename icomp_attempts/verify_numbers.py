"""Check every number quoted in the report against the result file that produced it.

Run after editing report.tex. A heavy rewrite silently broke one bound here: the weight-norm
stationarity index was quoted as "between 57 and 1087", but 57 is only the minimum for the
grokking run and the minimum across all runs is 20.4.

    python verify_numbers.py        # exits non-zero on any mismatch
"""
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "code" / "edm_validation" / "results"
ok = True


def check(label, actual, quoted, tol=0.02):
    global ok
    good = abs(actual - quoted) <= tol * max(abs(quoted), 1e-9)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {label:<52} report {quoted:>9} actual {actual:>9.3f}")


d9 = pd.read_csv(f"{R}/phase9_paper_settings_summary.csv")
grok = d9[d9.run.str.startswith("grok")]
check("dim spread across 3 seeds (factor)", grok.dim_late.max() / grok.dim_late.min(), 7.7, 0.02)
for run, quoted in (("grok", -0.54), ("grok_s1", 0.94), ("grok_s2", 0.90)):
    check(f"trend {run}", float(d9[d9.run == run].trend_rho.iloc[0]), quoted, 0.03)
for run, quoted in (("lowdata15", 0.75), ("lowdata20", 0.22)):
    check(f"trend {run}", float(d9[d9.run == run].trend_rho.iloc[0]), quoted, 0.05)

d14 = pd.read_csv(f"{R}/phase14_drift_ratio.csv")
wn = d14[d14.observable == "weight_norm"].stationarity_index
check("weight_norm stationarity index, min", wn.min(), 20, 0.03)
check("weight_norm stationarity index, max", wn.max(), 1087, 0.03)
tl = d14[(d14.observable == "train_loss") & (d14.window_steps == 25)].stationarity_index
check("train_loss index at 25 steps, max", tl.max(), 3.2, 0.05)

rec = pd.read_csv(f"{R}/phase14_local_recurrence.csv")
lor = rec[rec.series == "Lorenz-63 (attractor)"].iloc[0]
check("Lorenz recurrence at 40 steps", float(lor["excl_40"]), 0.68, 0.03)
gtl = rec[(rec.series == "grok_dense train_loss") & (rec.sampling == "every step")].iloc[0]
lo = min(float(gtl[f"excl_{e}"]) for e in (40, 80, 160))
hi = max(float(gtl[f"excl_{e}"]) for e in (40, 80, 160))
check("grok train_loss plateau, low", lo, 0.64, 0.03)
check("grok train_loss plateau, high", hi, 0.69, 0.03)

d15 = pd.read_csv(f"{R}/phase15_local_determinism.csv")
g = d15[d15.series == "grok train_loss"].iloc[0]
check("grok train_loss simplex skill", float(g.skill_h1), 0.739, 0.01)
check("grok train_loss surrogate mean", float(g.surrogate_mean), 0.722, 0.01)
check("grok train_loss p", float(g.p), 0.52, 0.05)
for name, quoted in (("Lorenz-63, dt=0.01 (oversampled)", 0.400), ("Lorenz-63, dt=0.1", 0.010)):
    check(f"{name} p", float(d15[d15.series == name].p.iloc[0]), quoted, 0.05)

d12 = pd.read_csv(f"{R}/phase12_matched_gains.csv")
c = d12[d12.coupled]
check("coupled runs with fwd gain > rev", (c.gain_fwd > c.gain_rev).sum(), 7, 0.01)
gh = d12[~d12.coupled]
check("max |ghost gain|", max(gh.gain_fwd.abs().max(), gh.gain_rev.abs().max()), 0.039, 0.05)

d10 = pd.read_csv(f"{R}/phase10_length_dependence.csv")
check("Lorenz growth at n=2000", float(d10[d10.n == 2000].lorenz.iloc[0]), 8.86, 0.01)
check("Lorenz growth at n=12000", float(d10[d10.n == 12000].lorenz.iloc[0]), 1.66, 0.01)

sweep = pd.read_csv(HERE.parent / "code" / "prediction_improved" / "results" / "sweep" / "summary.csv")
check("sweep gap mean", sweep.gap.mean(), 3062, 0.01)
check("sweep gap sd", sweep.gap.std(), 1130, 0.01)
check("sweep gap min", sweep.gap.min(), 1290, 0.01)
check("sweep gap max", sweep.gap.max(), 5600, 0.01)

print("\nALL CONSISTENT" if ok else "\nMISMATCHES FOUND")
raise SystemExit(0 if ok else 1)
