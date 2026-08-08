"""What the 120 000-step reruns settle, and what they overturn.

`launch_extended.sh` reran the three controls at six times the original budget, three
seeds for `lowdata15`, two for `wd0`, plus a positive control. The result changes the
status of every control in this project and refutes two claims made before it landed --
one of them mine, from `report_0808.md` section 6.3.

Read this file's output before quoting any false-alarm number anywhere in the repository.

    python exp8_extended.py
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension as mle                     # noqa: E402

EXT = HERE / "results" / "extended"
PRED = HERE.parent / "prediction_improved" / "results"
CONF = HERE.parent / "edm_validation" / "results" / "conf"
OUT = HERE / "results"
P_MOD, CHANCE = 113, 1.0 / 113
WIN_ROWS, CUT = 60, 20000            # the budget every earlier conclusion was drawn at


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


def load(path):
    d = pd.read_csv(path)
    s = d["step"].to_numpy()
    return (s, d["val_acc"].to_numpy(), d["weight_norm"].to_numpy(),
            first_sustained(s, d["train_acc"].to_numpy()),
            first_sustained(s, d["val_acc"].to_numpy()))


def mg_trace(wn):
    idx = np.arange(WIN_ROWS - 1, len(wn))
    return idx, np.array([mle(wn[i - WIN_ROWS + 1:i + 1], tau=1, max_E=15, k_neighbors=5,
                              correction="mackay_ghahramani", theiler_window=0,
                              rng=np.random.default_rng(0)) for i in idx])


def section_1():
    rule("1  Outcome at 120 000 steps")
    print(f"  Chance accuracy is 1/{P_MOD} = {CHANCE:.5f}. 'x chance' pools the last")
    print("  10 000 steps.\n")
    print(f"  {'run':<16}{'t_mem':>7}{'t_gen':>9}{'gap':>9}{'x chance':>10}"
          f"{'max val':>9}   outcome")
    rows = []
    for f in sorted(glob.glob(str(EXT / "*_train.csv"))):
        s, v, wn, tm, tg = load(f)
        tail = v[s >= s.max() - 10000].mean()
        name = os.path.basename(f)[:-10]
        out = f"GROKS at {tg}" if tg else "no"
        print(f"  {name:<16}{tm:>7}{str(tg):>9}{str(tg - tm) if tg else '-':>9}"
              f"{tail / CHANCE:>10.2f}{v.max():>9.4f}   {out}")
        rows.append({"run": name, "t_mem": tm, "t_gen": tg,
                     "gap": (tg - tm) if tg else None, "x_chance": tail / CHANCE,
                     "max_val": float(v.max()), "groks": tg is not None})
    pd.DataFrame(rows).to_csv(OUT / "exp8_outcomes.csv", index=False)
    print("\n  `lowdata15` groks in 1 of 3 seeds, at step 110 940 -- a plateau of 110 320")
    print("  steps, nine times the longest gap previously observed in this project. A")
    print("  second seed reaches 0.94 validation accuracy and is still climbing at the")
    print("  end of the budget. `lowdata20` groks at 39 600. The positive control groks")
    print("  at 5 110, so the harness is behaving.")


def section_2():
    rule("2  What each run looked like at step 20 000, where the project stopped")
    print("  This is the table that decides whether the controls were ever usable.\n")
    print(f"  {'run':<16}{'val@20k':>10}{'x chance':>10}{'trend 10k-20k':>15}"
          f"   eventual outcome")
    rows = []
    for f in sorted(glob.glob(str(EXT / "*_train.csv"))):
        s, v, wn, tm, tg = load(f)
        m = s <= CUT
        sv, vv = s[m], v[m]
        tail = vv[sv >= CUT - 2000].mean()
        h = sv >= CUT // 2
        rho = spearmanr(sv[h], vv[h]).statistic if np.std(vv[h]) > 0 else np.nan
        name = os.path.basename(f)[:-10]
        print(f"  {name:<16}{tail:>10.5f}{tail / CHANCE:>10.2f}{rho:>+15.3f}"
              f"   {'GROKS at ' + str(tg) if tg else 'never, in 120k'}")
        rows.append({"run": name, "val_at_20k": tail, "x_chance_at_20k": tail / CHANCE,
                     "rho_10k_20k": rho, "t_gen": tg})
    pd.DataFrame(rows).to_csv(OUT / "exp8_at_20k.csv", index=False)
    print("\n  `lowdata15_s0` sits at 0.55x chance with a falling trend at step 20 000")
    print("  and groks at 109 860. That is exactly the profile `exp3_censoring.py`")
    print("  used to classify a run as `extended_non_grokking`: below chance and not")
    print("  rising. **The diagnostic is refuted by direct counterexample.** Being far")
    print("  below chance at 20 000 steps carries no information about whether a run")
    print("  will generalise later.")


def section_3():
    rule("3  Retraction: the 'rise, not fall' separation does not survive")
    print("  report_0808.md section 6.3 reported that generalising runs excurse to")
    print("  3.2-18.0 while controls stay below 2.37. Two things were wrong with it.\n")
    print("  (a) The statistic was contaminated. exp7_broader.py measured the maximum")
    print("      over everything after t_mem WITHOUT capping at t_gen, so for a")
    print("      generalising run it included post-generalisation windows, where the")
    print("      weight norm flattens and the estimate degenerates towards max_E. The")
    print("      12-18 values were that artefact, not a rise before the transition.\n")
    print("  (b) Capped correctly at t_gen, and scored on runs it was not built on, the")
    print("      separation is gone:\n")
    print(f"  {'run':<22}{'groks':>6}{'windows':>9}{'median':>8}{'p95':>7}{'max':>7}")
    rows = []
    groups = [("previously used", [PRED / "grok_train.csv"]
               + sorted(Path(CONF).glob("*.csv"))
               + [PRED / f"{n}_train.csv" for n in ("lowdata15", "lowdata20", "wd0")]),
              ("held out (extended)", sorted(Path(EXT).glob("*_train.csv")))]
    for label, files in groups:
        print(f"  -- {label}")
        for f in files:
            s, v, wn, tm, tg = load(f)
            idx, dims = mg_trace(wn)
            r = s[idx]
            stop = tg if tg else r.max()
            post = dims[(r >= tm + 200) & (r <= stop)]
            if len(post) < 20:
                continue
            name = os.path.basename(str(f))[:20]
            print(f"  {name:<22}{('Y' if tg else 'N'):>6}{len(post):>9}"
                  f"{np.median(post):>8.2f}{np.percentile(post, 95):>7.2f}"
                  f"{post.max():>7.2f}")
            rows.append({"group": label, "run": name, "groks": tg is not None,
                         "n_windows": len(post), "median": float(np.median(post)),
                         "p95": float(np.percentile(post, 95)), "max": float(post.max())})
    f = pd.DataFrame(rows)
    f.to_csv(OUT / "exp8_rise.csv", index=False)
    held = f[f.group == "held out (extended)"]
    g, n = held[held.groks], held[~held.groks]
    print(f"\n  On the held-out runs the grokking ones reach p95 "
          f"{g.p95.min():.2f}-{g.p95.max():.2f} and the non-grokking ones "
          f"{n.p95.min():.2f}-{n.p95.max():.2f}.")
    print("  `lowdata15_s0` groks and reaches p95 1.79; `lowdata15_s1` does not grok and")
    print("  reaches 1.72. They are the same configuration at different seeds, measured")
    print("  over 11 000 windows each. The statistic does not separate them.")


def section_4():
    rule("4  Consequence for every false-alarm number in this repository")
    print("  Status of the five controls after the reruns:\n")
    print(f"  {'control':<30}{'status':<40}{'usable?'}")
    table = [("lowdata15 (fraction 0.15)",
              "config GROKS at 110 940, 1 of 3 seeds", "NO"),
             ("lowdata20 (fraction 0.20)", "GROKS at 39 600", "NO"),
             ("wd0 (weight decay 0)",
              "no grok in 120k; 8-10x chance, rising", "unresolved"),
             ("ma_S_5_without_weight_decay", "never run beyond 15 000 steps", "unresolved"),
             ("p_211_wd_0", "never rerun (198 200 steps, WD=0)", "unresolved")]
    for a, b, c in table:
        print(f"  {a:<30}{b:<40}{c}")
    print("\n  Not one of the five is established as a negative. Every false-alarm count")
    print("  and every specificity figure in report_0708.md, report_0708_experiments.md")
    print("  and report_0808.md was computed against runs whose labels are wrong or")
    print("  unknown. The correct statement is that **specificity has never been")
    print("  measured in this project**, and that the two firings previously called")
    print("  false alarms were firings on runs that do generalise.")
    print("\n  A precursor claim also needs a usable lead time, and that is now in doubt")
    print("  for a different reason: a plateau can be 110 320 steps long, so 'fires")
    print("  before t_gen' is nearly free. Lead time has to be reported relative to the")
    print("  gap, not in steps.")


def main():
    if not list(EXT.glob("*_train.csv")):
        print(f"no extended runs in {EXT}; run `sh launch_extended.sh` first")
        return 1
    for fn in (section_1, section_2, section_3, section_4):
        fn()
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
