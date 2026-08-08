"""Are the never-generalising runs real counterexamples, or right-censored observations?

The objection is correct in principle and has to be answered with evidence rather than
with the word "control". A run that has not generalised by step 20000 is not a run that
will not generalise; it is a run whose event time is known only to exceed 20000. Treating
it as a negative is the classic error that survival analysis exists to prevent, and it
directly inflates any false-alarm rate computed against it.

What can be decided from the logs already in the repo, without a GPU:

  A  chance level     -- is validation accuracy above what guessing gives? A run below
                         chance is not "about to generalise"; a run above it has learned
                         something about unseen pairs and the clock is still running.
  B  trend            -- is the excess over chance rising, flat, or falling at the end?
  C  milestones       -- the runs that did generalise passed through a sequence of
                         partial-accuracy levels first. How far along is each control?
  D  classification   -- late-grokking candidate / extended non-grokking / insufficient
                         budget, by a stated rule.
  E  consequence      -- what the answer does to the false-alarm counts reported in
                         ../prediction_improved/report_0708_experiments.md section 9.

What cannot: whether a censored run eventually groks. That needs the longer runs in
``extend_runs.py``, which is written for a GPU and is not run here.

    python exp3_censoring.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "prediction_improved" / "results"
CONF = HERE.parent / "edm_validation" / "results" / "conf"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

P_MOD = 113                     # modular addition modulus: chance accuracy is 1/113
VAL_BATCH = 512                 # runs.py sets val_batch_size=512; val_acc has 1/512 granularity
CHANCE = 1.0 / P_MOD

RUNS = {"grok": "grok_train.csv", "grok_s1": "grok_seed1_train.csv",
        "grok_s2": "grok_seed2_train.csv", "lowdata15": "lowdata15_train.csv",
        "lowdata20": "lowdata20_train.csv", "wd0": "wd0_train.csv"}
GENERALISES = {"grok", "grok_s1", "grok_s2"}


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def load(run):
    df = pd.read_csv(RES / RUNS[run]) if run in RUNS else pd.read_csv(CONF / f"{run}.csv")
    return df["step"].to_numpy(), df["val_acc"].to_numpy(), df["train_acc"].to_numpy()


def pooled_binomial(vals):
    """Two-sided exact test of pooled validation accuracy against 1/p."""
    n = len(vals) * VAL_BATCH
    hits = int(round(float(np.sum(vals)) * VAL_BATCH))
    hits = max(0, min(n, hits))
    return hits / n, binomtest(hits, n, CHANCE).pvalue, n


# ------------------------------------------------------------------- A. chance level

def section_A():
    rule("A  Is validation accuracy above chance at the end of the budget?")
    print(f"  Chance for modular addition mod {P_MOD} is 1/{P_MOD} = {CHANCE:.5f}.")
    print("  Pooled over the last 2000 optimisation steps, exact binomial test.\n")
    print(f"  {'run':<11}{'last-2k val_acc':>17}{'x chance':>10}{'p':>12}{'n trials':>10}"
          f"   reading")
    rows = []
    for run in RUNS:
        s, v, _ = load(run)
        tail = v[s >= s.max() - 2000]
        acc, p, n = pooled_binomial(tail)
        ratio = acc / CHANCE
        if run in GENERALISES:
            reading = "generalised"
        elif p < 0.01 and ratio > 1:
            reading = "ABOVE chance -- has learned something"
        elif p < 0.01 and ratio < 1:
            reading = "BELOW chance -- anti-generalising"
        else:
            reading = "indistinguishable from chance"
        print(f"  {run:<11}{acc:>17.5f}{ratio:>10.2f}{p:>12.2e}{n:>10}   {reading}")
        rows.append({"run": run, "tail_acc": acc, "x_chance": ratio, "p": p,
                     "n": n, "reading": reading})
    pd.DataFrame(rows).to_csv(OUT / "exp3_A_chance.csv", index=False)


# -------------------------------------------------------------------------- B. trend

def section_B():
    rule("B  Is the excess over chance still moving at the end?")
    print("  Spearman of val_acc against step over the last half of training, and the")
    print("  ratio of the final 2000 steps to the preceding 2000.\n")
    print(f"  {'run':<11}{'rho(last half)':>16}{'p':>10}{'last2k/prev2k':>15}"
          f"{'peak step':>11}{'peak/end':>10}")
    rows = []
    for run in RUNS:
        s, v, _ = load(run)
        half = s >= s.max() / 2
        rho = spearmanr(s[half], v[half])
        end = v[s >= s.max() - 2000].mean()
        prev = v[(s >= s.max() - 4000) & (s < s.max() - 2000)].mean()
        peak_step = int(s[np.argmax(v)])
        print(f"  {run:<11}{rho.statistic:>+16.3f}{rho.pvalue:>10.1e}"
              f"{(end / prev if prev > 0 else np.nan):>15.2f}{peak_step:>11}"
              f"{peak_step / s.max():>10.2f}")
        rows.append({"run": run, "rho_last_half": rho.statistic, "p": rho.pvalue,
                     "end_over_prev": end / prev if prev > 0 else np.nan,
                     "peak_step": peak_step, "peak_frac": peak_step / s.max()})
    pd.DataFrame(rows).to_csv(OUT / "exp3_B_trend.csv", index=False)
    print("\n  A run whose validation accuracy peaks in its final block and is still")
    print("  rising has not finished; a run whose accuracy peaked early and decayed has.")


# --------------------------------------------------------------------- C. milestones

def section_C():
    rule("C  How far along the road to generalisation is each run?")
    print("  The runs that generalised passed through partial accuracy first. Below,")
    print("  the first step at which a 2000-step trailing mean exceeds each multiple of")
    print("  chance. '--' means the run never reached that level.\n")
    levels = (1.0, 2.0, 4.0, 10.0, 50.0)
    print(f"  {'run':<11}{'t_gen':>7}" + "".join(f"{f'{L:g}x':>9}" for L in levels)
          + "   furthest level reached")
    rows = []
    for run in RUNS:
        s, v, ta = load(run)
        roll = pd.Series(v).rolling(200, min_periods=50).mean().to_numpy()
        cells, reached = [], 0.0
        for L in levels:
            hit = np.flatnonzero(roll >= L * CHANCE)
            if len(hit):
                cells.append(f"{int(s[hit[0]]):>9}")
                reached = L
            else:
                cells.append(f"{'--':>9}")
        tg = np.flatnonzero(pd.Series(v).rolling(5, min_periods=1,
                                                 center=True).mean().to_numpy() >= 0.95)
        tgs = int(s[tg[0]]) if len(tg) else None
        print(f"  {run:<11}{str(tgs):>7}" + "".join(cells) + f"   {reached:g}x chance")
        rows.append({"run": run, "t_gen": tgs, "furthest": reached,
                     **{f"t_{L:g}x": (int(s[np.flatnonzero(roll >= L * CHANCE)[0]])
                                      if (roll >= L * CHANCE).any() else None)
                        for L in levels}})
    pd.DataFrame(rows).to_csv(OUT / "exp3_C_milestones.csv", index=False)


# ----------------------------------------------------------------- D. classification

def classify(run):
    """Stated rule, applied to the last 2000 steps and the last half's trend.

    **REFUTED, 2026-08-08. Do not use this to label a run.** `exp8_extended.py` reran
    `lowdata15` at 120 000 steps: seed 0 sits at 0.55x chance with a falling trend at
    step 20 000 -- the exact profile this function calls `extended_non_grokking` -- and
    generalises at step 110 940. Being far below chance late in a short budget carries no
    information about whether a run will generalise later. The function is kept because
    §1 of the report quotes its output and the refutation has to stay auditable, but the
    only defensible label from a truncated run is `censored`.

    late_grokking_candidate -- above chance (p < 0.01) AND not falling. The event time is
        censored and the run must not be counted as a negative.
    extended_non_grokking   -- at or below chance, or falling. This was read as evidence
        that the run is not on a trajectory to generalise. That reading is wrong.
    insufficient_budget     -- above chance and rising, and the run also has less
        post-memorisation budget than the longest observed gap. Nothing can be concluded.
    """
    s, v, ta = load(run)
    tail = v[s >= s.max() - 2000]
    acc, p, _ = pooled_binomial(tail)
    half = s >= s.max() / 2
    rho = spearmanr(s[half], v[half]).statistic
    above = p < 0.01 and acc > CHANCE
    rising = np.isfinite(rho) and rho > 0.1
    falling = np.isfinite(rho) and rho < -0.1
    tm = np.flatnonzero(pd.Series(ta).rolling(5, min_periods=1, center=True)
                        .mean().to_numpy() >= 0.95)
    budget = s.max() - (int(s[tm[0]]) if len(tm) else 0)

    # A run whose validation accuracy has not changed by a single example in ten
    # thousand steps is not slowly improving, it is stationary. Spearman is undefined
    # there (zero variance), and "not falling" would otherwise pass it through as a
    # late-grokking candidate, which is exactly backwards.
    if np.std(v[half]) == 0:
        return "extended_non_grokking (frozen)", acc / CHANCE, rho, budget
    if above and rising and budget < 12070:
        return "insufficient_budget", acc / CHANCE, rho, budget
    if above and not falling:
        return "late_grokking_candidate", acc / CHANCE, rho, budget
    return "extended_non_grokking", acc / CHANCE, rho, budget


def section_D():
    rule("D  Classification of the three controls")
    print("  12070 steps is the longest post-memorisation interval actually observed to")
    print("  end in generalisation, in this repo, in the canonical run. A control with")
    print("  less budget than that has not been given the chance the positives had.\n")
    print(f"  {'run':<11}{'x chance':>10}{'rho':>8}{'budget':>9}   class")
    rows = []
    for run in RUNS:
        if run in GENERALISES:
            continue
        cls, ratio, rho, budget = classify(run)
        print(f"  {run:<11}{ratio:>10.2f}{rho:>+8.2f}{budget:>9}   {cls}")
        rows.append({"run": run, "x_chance": ratio, "rho": rho, "budget": budget,
                     "class": cls})
    pd.DataFrame(rows).to_csv(OUT / "exp3_D_class.csv", index=False)


# ------------------------------------------------------------------- E. consequence

def section_E():
    rule("E  What this does to the false-alarm counts already reported")
    print("  report_0708_experiments.md section 9 reports the MG-drop conjunct firing on")
    print("  2 of 3 controls at most settings, and 0 of 3 at one. Those counts assume")
    print("  every control is a true negative.\n")
    frame = pd.read_csv(OUT / "exp3_D_class.csv") if (OUT / "exp3_D_class.csv").exists() \
        else None
    if frame is None:
        print("  run section D first"); return
    n_neg = int(frame["class"].str.startswith("extended_non_grokking").sum())
    n_cen = len(frame) - n_neg
    print(f"  usable negatives            {n_neg}")
    print(f"  censored, not usable        {n_cen}")
    print("\n  Which run fired matters more than how many. From composite_7_bestcell.csv:")
    p = HERE.parent / "prediction_improved" / "results" / "audit" / "composite_7_bestcell.csv"
    if p.exists():
        fired = pd.read_csv(p)
        sub = fired[(~fired.generalises) & (fired.window == 600) & (fired["drop"] == 0.30)]
        for _, r in sub.iterrows():
            cls = frame.loc[frame.run == r.run, "class"]
            cls = cls.iloc[0] if len(cls) else "?"
            state = "fires" if pd.notna(r.t_fire) else "silent"
            print(f"    {r.run:<11} sustain={int(r.sustain)}  {state:<7}  {cls}")
    print("\n  The arithmetic that follows from this is in the README; the short form is")
    print("  that a false-alarm rate estimated on censored runs is not a false-alarm")
    print("  rate, and the composite criterion's specificity is currently unmeasured")
    print("  rather than measured and poor.")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D, "E": section_E}


def main(argv):
    for name in ([a.upper() for a in argv[1:]] or list(SECTIONS)):
        if name not in SECTIONS:
            print(f"unknown section {name!r}")
            return 2
        SECTIONS[name]()
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
