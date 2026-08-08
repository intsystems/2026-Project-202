"""A candidate replacement for the MG-drop criterion, and an honest account of its margin.

Everything the audit has established points the same way: the Levina-Bickel / MacKay-
Ghahramani estimate on delay vectors of ``weight_norm`` is a monotone function of that
series' local shape (Spearman +0.934 across observables in
``../prediction_improved/report_0708_experiments.md`` section 5). If so, the estimate is
a lossy encoding of the weight norm, and the weight norm itself should do at least as
well. This script tests that directly.

The rule under test -- one line, no embedding, no neighbours, no likelihood:

    fire at the first step after t_mem at which  1 - ||w||/peak(||w||)  >=  x,

where ``peak`` is the running maximum since t_mem. It is causal (history only), needs no
validation labels, and is scale free, so unlike an absolute threshold on ||w|| it is not
tied to one architecture's units.

It also has a mechanism behind it rather than being curve-fitted: on the Omnigrok account
generalisation in this setting follows the weight norm decaying out of the region where
the memorising solution is favoured, so a sustained drawdown is the thing that ought to
precede the transition.

    python exp4_criterion.py
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension                          # noqa: E402

PRED = HERE.parent / "prediction_improved" / "results"
CONF = HERE.parent / "edm_validation" / "results" / "conf"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# generalises? -- and, from exp3, whether the negative is usable or censored
RUNS = [("grok", PRED / "grok_train.csv", True, ""),
        ("grok_s1", PRED / "grok_seed1_train.csv", True, ""),
        ("grok_s2", PRED / "grok_seed2_train.csv", True, ""),
        ("lowdata15", PRED / "lowdata15_train.csv", False, "extended_non_grokking"),
        ("lowdata20", PRED / "lowdata20_train.csv", False, "late_grokking_candidate"),
        ("wd0", PRED / "wd0_train.csv", False, "extended_non_grokking (frozen)")]
RUNS += [(os.path.basename(f)[:-4], Path(f), True, "")
         for f in sorted(glob.glob(str(CONF / "*.csv")))]


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def first_sustained(steps, acc, thr=0.95, w=5):
    a = pd.Series(acc).rolling(w, min_periods=1, center=True).mean().to_numpy()
    ok = a >= thr
    idx = None
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        idx = i
    return int(steps[idx]) if idx is not None else None


def load(path):
    d = pd.read_csv(path)
    s = d["step"].to_numpy()
    return (s, d["weight_norm"].to_numpy(),
            first_sustained(s, d["train_acc"].to_numpy()),
            first_sustained(s, d["val_acc"].to_numpy()))


def drawdown(steps, wn, t_mem):
    """Causal drawdown of the weight norm below its running post-memorisation peak."""
    m = steps >= t_mem
    s, w = steps[m], wn[m]
    return s, 1.0 - w / np.maximum.accumulate(w)


def fire_step(s, dd, x, sustain=1):
    ok = dd >= x
    streak = 0
    for i in range(len(ok)):
        streak = streak + 1 if ok[i] else 0
        if streak >= sustain:
            return int(s[i])
    return None


# ------------------------------------------------------------------ 1. the margin

def section_1():
    rule("1  The quantity the rule thresholds, run by run")
    print("  'by t_gen' is the largest drawdown reached strictly before generalisation --")
    print("  what the rule has to work with. 'ever' is the largest reached at any point")
    print("  in the run, which for a control is what the rule has to stay under.\n")
    print(f"  {'run':<12}{'gen':>4}{'t_mem':>7}{'t_gen':>7}{'by t_gen':>10}{'ever':>8}"
          f"   note")
    rows = []
    for name, path, gen, note in RUNS:
        s, wn, tm, tg = load(path)
        ss, dd = drawdown(s, wn, tm)
        ever = float(dd.max())
        pre = dd[ss < tg] if (gen and tg) else np.array([])
        by = float(pre.max()) if len(pre) else np.nan
        print(f"  {name:<12}{'Y' if gen else 'N':>4}{tm:>7}{str(tg):>7}"
              f"{by:>9.1%}{ever:>8.1%}   {note}")
        rows.append({"run": name, "generalises": gen, "t_mem": tm, "t_gen": tg,
                     "drawdown_by_tgen": by, "drawdown_ever": ever, "note": note})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp4_1_margin.csv", index=False)
    pos = frame[frame.generalises].drawdown_by_tgen
    neg = frame[~frame.generalises].drawdown_ever
    neg_use = frame[(~frame.generalises)
                    & frame.note.str.startswith("extended")].drawdown_ever
    print(f"\n  smallest drawdown a generalising run reaches before t_gen : {pos.min():.1%}")
    print(f"  largest drawdown any control reaches, ever                 : {neg.max():.1%}")
    print(f"  ... restricted to the two uncensored controls              : {neg_use.max():.1%}")
    print(f"\n  separating margin: {pos.min() - neg.max():+.1%} of the norm.")
    print("  That is the whole basis of the rule, and it is one percentage point wide on")
    print("  11 runs against 3. It is a hypothesis to be pre-registered and tested on new")
    print("  runs, not a result.")


# ---------------------------------------------------------------- 2. sweep the rule

def section_2():
    rule("2  Recall, false alarms and lead time against the threshold")
    print("  Causal throughout: the running peak uses history only, and no run's own")
    print("  future is consulted. Controls are judged over their whole post-memorisation")
    print("  budget, which is longer than any positive's plateau.\n")
    print(f"  {'x':>5}{'sustain':>9}{'recall':>9}{'FP (all 3)':>12}{'FP (uncens. 2)':>16}"
          f"{'median lead':>13}{'min lead':>10}")
    rows = []
    for x in (0.05, 0.10, 0.15, 0.18, 0.20, 0.25):
        for sustain in (1, 5):
            hit = pos = fp = fp_use = 0
            leads = []
            for name, path, gen, note in RUNS:
                s, wn, tm, tg = load(path)
                ss, dd = drawdown(s, wn, tm)
                t = fire_step(ss, dd, x, sustain)
                if gen and tg:
                    pos += 1
                    if t is not None and t < tg:
                        hit += 1
                        leads.append(tg - t)
                elif not gen:
                    fp += int(t is not None)
                    fp_use += int(t is not None and note.startswith("extended"))
            print(f"  {x:>5.0%}{sustain:>9}{f'{hit}/{pos}':>9}{f'{fp}/3':>12}"
                  f"{f'{fp_use}/2':>16}"
                  f"{(np.median(leads) if leads else np.nan):>13.0f}"
                  f"{(min(leads) if leads else np.nan):>10.0f}")
            rows.append({"x": x, "sustain": sustain, "hits": hit, "positives": pos,
                         "fp_all": fp, "fp_uncensored": fp_use,
                         "median_lead": float(np.median(leads)) if leads else np.nan,
                         "min_lead": int(min(leads)) if leads else None})
    pd.DataFrame(rows).to_csv(OUT / "exp4_2_sweep.csv", index=False)


# --------------------------------------------------- 3. against the dimension rule

def section_3():
    rule("3  Side by side with the MG-drop conjunct it would replace")
    print("  MG figures are from ../prediction_improved/results/audit/composite_2_sweep.csv")
    print("  at its best cell (window 600, drop 30%, sustain 5) and its neighbour.\n")
    ref = HERE.parent / "prediction_improved" / "results" / "audit" / "composite_2_sweep.csv"
    print(f"  {'rule':<44}{'recall':>9}{'FP':>7}{'free parameters':>18}")
    if ref.exists():
        mg = pd.read_csv(ref)
        for w, d, su in ((600, 0.30, 5), (600, 0.30, 3), (600, 0.10, 3)):
            r = mg[(mg.window == w) & (mg["drop"] == d) & (mg.sustain == su)]
            if len(r):
                r = r.iloc[0]
                print(f"  {f'MG drop {d:.0%}, sustain {su}, window {w}':<44}"
                      f"{f'{int(r.fires)}/11':>9}{f'{int(r.fp)}/3':>7}"
                      f"{'window, E, k, W, drop, sustain':>18}")
    best = pd.read_csv(OUT / "exp4_2_sweep.csv") if (OUT / "exp4_2_sweep.csv").exists() else None
    if best is not None:
        for _, r in best[best.sustain == 5].iterrows():
            print(f"  {f'norm drawdown {r.x:.0%}, sustain 5':<44}"
                  f"{f'{int(r.hits)}/11':>9}{f'{int(r.fp_all)}/3':>7}"
                  f"{'drawdown, sustain':>18}")
    print("\n  The drawdown rule has two free parameters against six, needs no embedding,")
    print("  no neighbour search and no likelihood, and its best cell is not adjacent to")
    print("  a cell that fails. Whether it generalises to other tasks is untested.")


# ------------------------------------------------------- 4. is the dimension redundant?

def section_4():
    rule("4  Does the MG estimate add anything to the drawdown?")
    print("  Within each run, Spearman between the sliding MG estimate on weight_norm and")
    print("  the causal drawdown at the same right edge. A high correlation means the")
    print("  embedding machinery is re-encoding the drawdown rather than adding to it.\n")
    from scipy.stats import spearmanr
    win, stride = 60, 10
    print(f"  {'run':<12}{'n windows':>11}{'rho(MG, drawdown)':>20}")
    rows = []
    for name, path, gen, note in RUNS:
        s, wn, tm, tg = load(path)
        right, dims = [], []
        for a in range(0, len(wn) - win + 1, stride):
            if s[a] < tm:
                continue
            dims.append(mle_intrinsic_dimension(
                wn[a:a + win], tau=1, max_E=15, k_neighbors=5,
                correction="mackay_ghahramani", theiler_window=0,
                rng=np.random.default_rng(0)))
            right.append(s[a + win - 1])
        ss, dd = drawdown(s, wn, tm)
        at = np.interp(right, ss, dd)
        d = np.asarray(dims, dtype=float)
        m = np.isfinite(d)
        rho = spearmanr(d[m], at[m]).statistic if m.sum() > 5 else np.nan
        print(f"  {name:<12}{int(m.sum()):>11}{rho:>+20.2f}")
        rows.append({"run": name, "n": int(m.sum()), "rho": rho})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp4_4_redundancy.csv", index=False)
    print(f"\n  median |rho| = {frame.rho.abs().median():.2f} over {len(frame)} runs")


SECTIONS = {"1": section_1, "2": section_2, "3": section_3, "4": section_4}


def main(argv):
    for name in (argv[1:] or list(SECTIONS)):
        if name not in SECTIONS:
            print(f"unknown section {name!r}")
            return 2
        SECTIONS[name]()
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
