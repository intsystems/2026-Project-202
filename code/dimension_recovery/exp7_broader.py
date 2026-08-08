"""Every run in the repo that logs a weight norm, not just the seven usually used.

The evaluation set behind every criterion in this project has been 11 generalising runs
of one task and 3 controls, two of which share a configuration. The repository actually
contains logs for four tasks (modular addition mod 113 and mod 211, S_5, S_6), two
optimisers (full batch and minibatch) and three weight decays, including three
never-generalising runs at WD = 0 that have never been used as controls.

Two questions here:

  * does the dimension-drop rule survive on runs it was not built on;
  * item 6 of the brief -- at WD = 0, has the estimate already fallen by the time
    memorisation completes, or does it fall later? If it falls *after* t_mem then WD = 0
    is not the trivial control it has been treated as.

Units. These logs are written at strides of 1, 5, 10 and 49 optimisation steps. The
estimator window is therefore defined in ROWS, not steps, and the step equivalent is
printed beside every result: two runs at the same row-window are not at the same
physical scale, and exp6 shows that scale moves the estimate.

**Read `exp8_extended.py` before quoting any false-alarm number from section 3.** Two of
the five controls used here generalise at a longer budget, so their firings are not false
alarms and the counts below are not a specificity measurement.

    python exp7_broader.py
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension as mle                     # noqa: E402

CODE = HERE.parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

WIN_ROWS = 60           # estimator window, in logged rows
H_ROWS = 60             # criterion interval, in logged rows (= WIN_ROWS, so disjoint)
MAX_E, K = 15, 5


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def first_sustained(steps, acc, thr=0.95, w=5):
    a = pd.Series(acc).rolling(w, min_periods=1, center=True).mean().to_numpy()
    ok = a >= thr
    i = None
    for j in range(len(ok) - 1, -1, -1):
        if not ok[j]:
            break
        i = j
    return int(steps[i]) if i is not None else None


def inventory():
    """Every log with a weight norm, with its task and weight decay from the filename."""
    pats = ["prediction_improved/results/*_train.csv",
            "edm_validation/results/conf/*.csv",
            "Grokking/grokking_logs/*.csv",
            "Grokking/modular_addition_grokking_colab/training_log*.csv",
            "dimension_recovery/results/extended/*_train.csv"]
    seen, runs = set(), []
    for p in pats:
        for f in sorted(glob.glob(str(CODE / p))):
            d = pd.read_csv(f)
            if not {"step", "weight_norm", "train_acc", "val_acc"} <= set(d.columns):
                continue
            s = d["step"].to_numpy()
            if len(s) < 3 * WIN_ROWS:
                continue
            key = (len(d), round(float(d.weight_norm.iloc[0]), 3),
                   round(float(d.weight_norm.iloc[-1]), 3))
            if key in seen:                       # the same run stored twice
                continue
            seen.add(key)
            name = os.path.basename(f).replace("_train.csv", "").replace(".csv", "")
            name = name.replace("grokking_modular_addition_logs_", "ma_")
            name = name.replace("_with_stochastic", "_st").replace("training_log_", "")
            tm = first_sustained(s, d["train_acc"].to_numpy())
            tg = first_sustained(s, d["val_acc"].to_numpy())
            runs.append({"name": name[:34], "path": f, "n": len(d),
                         "stride": int(s[1] - s[0]), "t_mem": tm, "t_gen": tg,
                         "gap": (tg - tm) if (tm is not None and tg is not None) else None,
                         "wn0": float(d.weight_norm.iloc[0]),
                         "wn1": float(d.weight_norm.iloc[-1])})
    return pd.DataFrame(runs)


_T = {}


def mg_trace(path):
    if path in _T:
        return _T[path]
    d = pd.read_csv(path)
    s, wn = d["step"].to_numpy(), d["weight_norm"].to_numpy()
    idx = np.arange(WIN_ROWS - 1, len(wn))
    dims = np.array([mle(wn[i - WIN_ROWS + 1:i + 1], tau=1, max_E=MAX_E, k_neighbors=K,
                         correction="mackay_ghahramani", theiler_window=0,
                         rng=np.random.default_rng(0)) for i in idx])
    _T[path] = (s[idx], dims)
    return _T[path]


def running_max_fire(r, d, x=0.30):
    pk = np.maximum.accumulate(d)
    i = np.flatnonzero(1 - d / pk >= x)
    return int(r[i[0]]) if len(i) else None


def revised_fire(r, d, delta=0.20, beta=0.10, Qrows=30, dhold=0.15, dispmax=0.08):
    n = len(r)
    for i in range(n):
        bm = slice(max(0, i - 2 * H_ROWS), max(0, i - H_ROWS))
        am = slice(max(0, i - H_ROWS), i + 1)
        b, a = d[bm], d[am]
        if len(b) < 8 or len(a) < 8:
            continue
        mb = np.median(b)
        if mb <= 0 or np.median(np.abs(b - mb)) / mb > dispmax:
            continue
        if 1 - np.median(a) / mb < delta:
            continue
        # r is in optimisation steps, so the Theil-Sen slope is per step; the interval
        # is H_ROWS rows = H_ROWS * stride steps. Normalising by the baseline makes the
        # test scale-free and comparable across logs written at different strides.
        stride = r[1] - r[0]
        if (H_ROWS * stride) * theilslopes(a, r[am])[0] / mb > -beta:
            continue
        h = d[i:min(n, i + Qrows + 1)]
        if len(h) < 3 or np.median(h) > (1 - dhold) * mb:
            continue
        return int(r[i]), int(r[min(n - 1, i + Qrows)])
    return None, None


def main():
    inv = inventory()
    inv.to_csv(OUT / "exp7_inventory.csv", index=False)

    rule("1  Every run in the repository that logs a weight norm")
    print(f"  {'run':<36}{'stride':>7}{'t_mem':>8}{'t_gen':>8}{'gap':>8}"
          f"{'||w||':>14}   class")
    for _, x in inv.iterrows():
        gen = x.t_gen is not None and not pd.isna(x.t_gen)
        has_gap = x.gap is not None and not pd.isna(x.gap) and x.gap > 100
        cls = ("generalises, gap" if (gen and has_gap) else
               "generalises, no gap" if gen else "NEVER generalises")
        print(f"  {x['name']:<36}{x.stride:>7}{str(x.t_mem):>8}{str(x.t_gen):>8}"
              f"{str(x.gap):>8}{f'{x.wn0:.0f}->{x.wn1:.0f}':>14}   {cls}")
    n_neg = int(inv.t_gen.isna().sum())
    print(f"\n  {len(inv)} runs: {len(inv) - n_neg} generalise, {n_neg} never do.")
    print("  The three controls used until now were one WD=0 run and two low-data runs")
    print("  of a single task. This set spans four tasks and two optimisers.")

    rule("2  Item 6: at WD = 0, has the estimate fallen by the time memorisation ends?")
    print("  For every never-generalising run, and for the WD=0 runs in particular:")
    print("  the estimate before t_mem, at t_mem, and later. If it only falls after")
    print("  memorisation, WD=0 is not a control in which 'nothing happens'.\n")
    print(f"  {'run':<36}{'t_mem':>8}{'d before':>10}{'d at t_mem':>12}"
          f"{'d after':>9}{'max after':>11}{'falls?':>9}")
    rows = []
    for _, x in inv.iterrows():
        r, d = mg_trace(x.path)
        tm = x.t_mem
        if tm is None or pd.isna(tm):
            continue
        # Cap at t_gen. Without this the "after" window of a generalising run includes
        # post-generalisation data, where the weight norm flattens and the estimate
        # degenerates towards max_E -- which is how an earlier version of this file
        # produced readings of 12-18 and an apparent "rise" that exp8 then refuted.
        stop = x.t_gen if (x.t_gen is not None and not pd.isna(x.t_gen)) else r.max()
        pre = d[r < tm]
        at = d[(r >= tm) & (r < tm + 20 * x.stride)]
        post = d[(r >= tm + 20 * x.stride) & (r <= stop)]
        if not len(pre) or not len(post):
            continue
        v = (float(np.median(pre)), float(np.median(at)) if len(at) else np.nan,
             float(np.median(post)), float(np.max(post)))
        gen = x.t_gen is not None and not pd.isna(x.t_gen)
        falls = "yes" if v[2] < 0.85 * v[0] else "no"
        print(f"  {('+ ' if gen else '- ') + x['name']:<36}{int(tm):>8}{v[0]:>10.2f}{v[1]:>12.2f}"
              f"{v[2]:>9.2f}{v[3]:>11.2f}{falls:>9}")
        rows.append({"run": x["name"], "generalises": gen, "t_mem": tm, "d_pre": v[0], "d_at": v[1],
                     "d_post": v[2], "d_max_post": v[3], "falls": falls})
    pd.DataFrame(rows).to_csv(OUT / "exp7_wd0.csv", index=False)

    rule("3  Both rules on every run, with the budget asymmetry made explicit")
    print("  'delay' is steps from t_mem to the trigger. For a generalising run it must")
    print("  be below the gap to be useful; for a control it is simply how long the run")
    print("  survived. A control watched for 18 600 steps has many more chances to fire")
    print("  than a positive whose plateau is 1 160 steps long.\n")
    print(f"  {'run':<36}{'gap':>8}{'budget':>9}{'maxrule delay':>15}"
          f"{'revised delay':>15}{'verdict':>22}")
    out = []
    for _, x in inv.iterrows():
        r, d = mg_trace(x.path)
        tm = x.t_mem
        if tm is None or pd.isna(tm):
            continue
        gen = x.t_gen is not None and not pd.isna(x.t_gen)
        stop = x.t_gen if gen else r.max()
        m = (r - WIN_ROWS * x.stride >= tm) & (r <= stop)
        if m.sum() < 3 * H_ROWS:
            print(f"  {x['name']:<36}{str(x.gap):>8}{'--':>9}"
                  f"{'(plateau too short)':>15}")
            continue
        rr, dd = r[m], d[m]
        t_max = running_max_fire(rr, dd)
        t_rev, dec = revised_fire(rr, dd)
        budget = int(stop - tm)
        dly_max = (t_max - tm) if t_max else None
        dly_rev = (t_rev - tm) if t_rev else None
        if gen:
            v = "hit" if (dec is not None and dec < x.t_gen) else (
                "late" if dec is not None else "miss")
        else:
            v = "FALSE ALARM" if t_rev else "silent"
        print(f"  {x['name']:<36}{str(x.gap):>8}{budget:>9}{str(dly_max):>15}"
              f"{str(dly_rev):>15}{v:>22}")
        out.append({"run": x["name"], "generalises": gen, "gap": x.gap,
                    "budget": budget, "delay_maxrule": dly_max,
                    "delay_revised": dly_rev, "verdict": v})
    f = pd.DataFrame(out)
    f.to_csv(OUT / "exp7_rules.csv", index=False)

    pos, neg = f[f.generalises], f[~f.generalises]
    print(f"\n  revised rule: recall {int((pos.verdict == 'hit').sum())}/{len(pos)}"
          f"   false alarms {int((neg.verdict == 'FALSE ALARM').sum())}/{len(neg)}")
    print(f"  max rule    : fires on {int(pos.delay_maxrule.notna().sum())}/{len(pos)}"
          f" positives and {int(neg.delay_maxrule.notna().sum())}/{len(neg)} controls")
    d_pos = pos.delay_revised.dropna()
    d_neg = neg.delay_revised.dropna()
    if len(d_pos):
        print(f"\n  delay to trigger, positives : {d_pos.min():.0f} - {d_pos.max():.0f}"
              f"  (median {d_pos.median():.0f})")
    if len(d_neg):
        print(f"  delay to trigger, controls  : {d_neg.min():.0f} - {d_neg.max():.0f}")
    print(f"  budget, positives : {pos.budget.min()} - {pos.budget.max()}")
    print(f"  budget, controls  : {neg.budget.min()} - {neg.budget.max()}")
    print("\n  False alarms per 1000 steps of control budget: "
          f"{1000 * neg.delay_revised.notna().sum() / neg.budget.sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
