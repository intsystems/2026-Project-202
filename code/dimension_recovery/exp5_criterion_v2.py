"""Is a "dimension drop" detector possible at all, and what must it require?

Five parts.

  1  why the running-maximum rule is not a drop detector
  2  the proposed two-interval criterion, implemented, and its four defects
  3  a revised criterion that fixes them
  4  calibration on POSITIVES ONLY, then a single evaluation on held-out runs
  5  sensitivity of the conclusion to every free parameter

The calibration protocol matters more than the criterion. There are three
never-generalising runs in this repo and one of them is right-censored (exp3), so any
threshold tuned to make the controls silent is tuned on two data points. Instead the
thresholds here are chosen using **generalising runs only** -- pick the loosest setting
that still reaches a target recall on a designated calibration set of configurations --
and the controls are looked at exactly once, afterwards. That way the false-alarm count
is an out-of-sample number rather than the thing that was optimised.

    python exp5_criterion_v2.py            # all parts, ~4 min
    python exp5_criterion_v2.py 1 3        # named parts

**Superseded 2026-08-08.** Every false-alarm figure below is scored against `lowdata15`,
`lowdata20` and `wd0`. The first two generalise at a 120 000-step budget -- see
`exp8_extended.py` -- so they are not negatives and their firings are not false alarms.
Recall figures and label-independent quantities (parameter counts, the within-run
correlation between the drawdown and the MG estimate) still stand. Nothing here measures
specificity.
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

from edm import mle_intrinsic_dimension as mle                    # noqa: E402

PRED = HERE.parent / "prediction_improved" / "results"
CONF = HERE.parent / "edm_validation" / "results" / "conf"
EXT = HERE / "results" / "extended"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

WIN_STEPS = 600          # estimator window
LOG_EVERY = 10
MAX_E, K = 15, 5

# name -> (training log, generalises, group). The group is what the calibration/test
# split is made on: whole configurations, never individual seeds.
def _runs():
    r = [("grok", PRED / "grok_train.csv", True, "headline"),
         ("grok_s1", PRED / "grok_seed1_train.csv", True, "headline"),
         ("grok_s2", PRED / "grok_seed2_train.csv", True, "headline"),
         ("lowdata15", PRED / "lowdata15_train.csv", False, "control"),
         ("lowdata20", PRED / "lowdata20_train.csv", False, "control"),
         ("wd0", PRED / "wd0_train.csv", False, "control")]
    r += [(os.path.basename(f)[:-4], Path(f), True, "conf")
          for f in sorted(glob.glob(str(CONF / "*.csv")))]
    r += [(os.path.basename(f)[:-10], Path(f), None, "extended")
          for f in sorted(glob.glob(str(EXT / "*_train.csv")))]
    return r


RUNS = _runs()
_CACHE = {}


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


def trace(name, path, win_steps=WIN_STEPS):
    """Causal MG trace on weight_norm: (steps at right edge, values, t_mem, t_gen)."""
    key = (name, win_steps)
    if key in _CACHE:
        return _CACHE[key]
    d = pd.read_csv(path)
    s, wn = d["step"].to_numpy(), d["weight_norm"].to_numpy()
    tm = first_sustained(s, d["train_acc"].to_numpy())
    tg = first_sustained(s, d["val_acc"].to_numpy())
    w = win_steps // LOG_EVERY
    right, dims = [], []
    for a in range(0, len(wn) - w + 1):
        right.append(s[a + w - 1])
        dims.append(mle(wn[a:a + w], tau=1, max_E=MAX_E, k_neighbors=K,
                        correction="mackay_ghahramani", theiler_window=0,
                        rng=np.random.default_rng(0)))
    _CACHE[key] = (np.asarray(right), np.asarray(dims, float), tm, tg)
    return _CACHE[key]


def scope(r, d, tm, tg, generalises, win_steps=WIN_STEPS):
    """Windows lying wholly inside the plateau. Controls get the same length budget."""
    left = r - win_steps + 1
    stop = tg if (generalises and tg) else r.max()
    m = (left >= tm) & (r <= stop)
    return r[m], d[m]


# ------------------------------------------- 1. the running-maximum rule is not local

def part_1():
    rule("1  Why 'below the running maximum' is not a drop detector")
    print("  The rule fires when d_t <= (1-x) * max(history). Two things follow that a")
    print("  detector should not have: the reference is a single order statistic, and")
    print("  nothing requires the fall to happen near it.\n")

    print("  (a) Synthetic: a flat trace with one injected outlier.")
    rng = np.random.default_rng(0)
    base = 1.60 + 0.05 * rng.standard_normal(400)
    for spike in (0.0, 0.4, 0.8, 1.2):
        x = base.copy()
        x[50] += spike
        peak = np.maximum.accumulate(x)
        rel = 1 - x / peak
        fired = np.flatnonzero(rel >= 0.30)
        print(f"      spike +{spike:.1f} at index 50 (trace is otherwise flat at 1.60"
              f" +- 0.05): "
              f"{'FIRES at index ' + str(int(fired[0])) if len(fired) else 'silent'}")
    print("      A single point 0.8 above a flat trace is enough. Nothing about the")
    print("      trace changed; one sample did.\n")

    print("  (b) Real traces: how much of the trigger is the single largest point?")
    print("      'max' is the published rule. 'p95' replaces the running maximum by the")
    print("      running 95th percentile -- same rule, one order statistic more robust.")
    print(f"      {'run':<12}{'gen':>4}{'peak':>7}{'2nd':>7}{'p95':>7}"
          f"{'fire@ max':>11}{'fire@ p95':>11}{'gap peak->fire':>16}")
    rows = []
    for name, path, gen, group in RUNS:
        if group == "extended" or gen is None:
            continue
        r, d, tm, tg = trace(name, path)
        r, d = scope(r, d, tm, tg, gen)
        if len(d) < 20:
            continue
        pk = np.maximum.accumulate(d)
        p95 = np.array([np.percentile(d[:i + 1], 95) for i in range(len(d))])
        f_max = np.flatnonzero(1 - d / pk >= 0.30)
        f_p95 = np.flatnonzero(1 - d / p95 >= 0.30)
        srt = np.sort(d)[::-1]
        i = int(f_max[0]) if len(f_max) else None
        j = int(np.argmax(d[:i + 1])) if i is not None else None
        rows.append({"run": name, "generalises": gen, "peak": srt[0], "second": srt[1],
                     "p95": np.percentile(d, 95),
                     "fire_max": int(r[i]) if i is not None else None,
                     "fire_p95": int(r[f_p95[0]]) if len(f_p95) else None,
                     "gap": int(r[i] - r[j]) if i is not None else None})
        print(f"      {name:<12}{'Y' if gen else 'N':>4}{srt[0]:>7.2f}{srt[1]:>7.2f}"
              f"{np.percentile(d, 95):>7.2f}"
              f"{str(rows[-1]['fire_max']):>11}{str(rows[-1]['fire_p95']):>11}"
              f"{str(rows[-1]['gap']):>16}")
    pd.DataFrame(rows).to_csv(OUT / "exp5_1_runningmax.csv", index=False)
    print("\n      The peak and the second-largest value are close in every run, so on")
    print("      THIS data the trigger is not riding on one outlier. That is luck, not")
    print("      design: part (a) shows the rule has no defence against it, and the")
    print("      traces here are smooth only because the weight norm is smooth.")
    print("      The real defect is visible in the last column -- see part 2.")


# ------------------------------------------------- 2. the proposed criterion, as given

def medians_at(r, d, t, H):
    """(median before, median after, n_before, n_after) for the two adjacent intervals."""
    b = d[(r >= t - 2 * H) & (r < t - H)]
    a = d[(r >= t - H) & (r <= t)]
    return (np.median(b) if len(b) else np.nan,
            np.median(a) if len(a) else np.nan, len(b), len(a))


def proposed(r, d, H, delta, beta_rel, Q, delta_hold, min_pts=8):
    """The criterion exactly as proposed. Returns (t0, decision_time) or (None, None)."""
    for i, t in enumerate(r):
        mb, ma, nb, na = medians_at(r, d, t, H)
        if nb < min_pts or na < min_pts or not np.isfinite(mb) or mb <= 0:
            continue
        if 1 - ma / mb < delta:
            continue
        seg = (r >= t - H) & (r <= t)
        if seg.sum() < 3:
            continue
        b = theilslopes(d[seg], r[seg])[0]
        if H * b / mb > -beta_rel:
            continue
        hold = d[(r >= t) & (r <= t + Q)]
        if len(hold) < 3 or np.median(hold) > (1 - delta_hold) * mb:
            continue
        return int(t), int(t + Q)
    return None, None


def part_2():
    rule("2  The proposed criterion, and four things wrong with it")
    print("  R_t = 1 - median(d on [t-H,t]) / median(d on [t-2H,t-H)) >= delta,")
    print("  Theil-Sen slope on [t-H,t] normalised by median(before) <= -beta_rel,")
    print("  and the low level held for Q further points.\n")
    print("  It is a large improvement on the running-maximum rule: the reference is a")
    print("  median rather than an extremum, the fall is between ADJACENT intervals so")
    print("  it is local by construction, and the hold condition rejects a single dip.")
    print("  Four defects, in decreasing order of importance:\n")
    print("  D1. **Hold reads the future.** The condition on [t0, t0+Q] cannot be")
    print("      evaluated at t0. The criterion fires at t0 but is only KNOWN at t0+Q,")
    print("      so any lead time must be measured from t0+Q. Reporting t_gen - t0 as")
    print(f"      the lead overstates it by Q, which at the values below is 300-600 steps")
    print("      out of a median lead of about 1500.")
    print("  D2. **The two intervals are not independent unless H >= W.** Each d_s is")
    print(f"      itself computed on a window of W={WIN_STEPS} steps of raw data. With")
    print("      H < W the 'before' and 'after' intervals are built from overlapping")
    print("      weight-norm samples, so R_t is damped towards zero and the slope test")
    print("      sees autocorrelated points. H >= W is required, and it is not stated.")
    print("  D3. **No stability requirement on the baseline**, though the brief asks for")
    print("      one. median(before) can be the median of a trace that is itself moving,")
    print("      in which case R_t measures the continuation of a trend rather than a")
    print("      drop. A dispersion bound on the before-interval is needed.")
    print("  D4. **R_t and the slope test are largely redundant.** Both are computed on")
    print("      the same interval and both measure the same fall; requiring both mostly")
    print("      re-weights one quantity. The slope earns its place only if it is applied")
    print("      where R_t is not -- i.e. as a *monotonicity* check, which is what")
    print("      Theil-Sen gives, so the fix is to keep it but stop treating it as")
    print("      independent evidence.")


# ------------------------------------------------------------- 3. the revised criterion

def revised(r, d, H, delta, beta_rel, Q, delta_hold, disp_max, min_pts=8):
    """As proposed, plus a stable-baseline test, and returning the honest decision time.

    Requirements, in the order the brief states them:
      stable baseline   MAD(before)/median(before) <= disp_max
      real fall         1 - median(after)/median(before) >= delta
      directed trend    H * TheilSen(after) / median(before) <= -beta_rel
      level held        median over [t0, t0+Q] <= (1-delta_hold) * median(before)
      causal            every quantity above uses data at or before its own timestamp,
                        and the criterion is reported as decided at t0+Q, not at t0
    """
    for t in r:
        mb, ma, nb, na = medians_at(r, d, t, H)
        if nb < min_pts or na < min_pts or not np.isfinite(mb) or mb <= 0:
            continue
        before = d[(r >= t - 2 * H) & (r < t - H)]
        if np.median(np.abs(before - mb)) / mb > disp_max:      # stable baseline
            continue
        if 1 - ma / mb < delta:                                  # real fall
            continue
        seg = (r >= t - H) & (r <= t)
        if seg.sum() < 3:
            continue
        if H * theilslopes(d[seg], r[seg])[0] / mb > -beta_rel:  # directed trend
            continue
        if Q > 0:                                            # level held
            hold = d[(r >= t) & (r <= t + Q)]
            if len(hold) < 3 or np.median(hold) > (1 - delta_hold) * mb:
                continue
        return int(t), int(t + Q)
    return None, None


_FEAT = {}


def features(name, path, gen, H, Qs, win_steps=WIN_STEPS):
    """Precompute every quantity the criterion thresholds, once per (run, H).

    The grid search in part 4 evaluates 540 parameter settings; recomputing the
    Theil-Sen slope inside that loop would be a million fits. Everything except the
    thresholds depends only on (run, H), so it is computed here and reused.
    """
    key = (name, H, win_steps, tuple(Qs))
    if key in _FEAT:
        return _FEAT[key]
    r, d, tm, tg = trace(name, path, win_steps)
    r, d = scope(r, d, tm, tg, bool(gen), win_steps)
    n = len(r)
    mb = np.full(n, np.nan)
    ma = np.full(n, np.nan)
    disp = np.full(n, np.inf)
    slope = np.full(n, np.inf)
    hold = {q: np.full(n, np.inf) for q in Qs}
    for i, t in enumerate(r):
        bm = (r >= t - 2 * H) & (r < t - H)
        am = (r >= t - H) & (r <= t)
        if bm.sum() < 8 or am.sum() < 8:
            continue
        b = d[bm]
        mb[i] = np.median(b)
        if mb[i] <= 0:
            continue
        ma[i] = np.median(d[am])
        disp[i] = np.median(np.abs(b - mb[i])) / mb[i]
        slope[i] = H * theilslopes(d[am], r[am])[0] / mb[i]
        for q in Qs:
            hm = (r >= t) & (r <= t + q)
            if q == 0:
                hold[q][i] = -np.inf
            elif hm.sum() >= 3:
                hold[q][i] = np.median(d[hm]) / mb[i]
    _FEAT[key] = dict(steps=r, d=d, mb=mb, ma=ma, disp=disp, slope=slope, hold=hold,
                      t_mem=tm, t_gen=tg)
    return _FEAT[key]


def fire(f, delta, beta_rel, Q, delta_hold, disp_max):
    """First index satisfying every condition; returns (t0, decision time) or (None, None)."""
    ok = (np.isfinite(f["mb"]) & (f["disp"] <= disp_max)
          & (1 - f["ma"] / f["mb"] >= delta) & (f["slope"] <= -beta_rel)
          & (f["hold"][Q] <= 1 - delta_hold))
    i = np.flatnonzero(ok)
    if not len(i):
        return None, None
    return int(f["steps"][i[0]]), int(f["steps"][i[0]] + Q)


def evaluate_fast(params, runs):
    """Same output as `evaluate`, via the precomputed feature tables."""
    rows = []
    for name, path, gen, group in runs:
        f = features(name, path, gen, params["H"], (params["Q"],))
        if len(f["steps"]) < 20:
            rows.append({"run": name, "generalises": gen, "group": group,
                         "t0": None, "decided": None, "lead": None,
                         "status": "no window"})
            continue
        t0, dec = fire(f, params["delta"], params["beta_rel"], params["Q"],
                       params["delta_hold"], params["disp_max"])
        tg = f["t_gen"]
        lead = (tg - dec) if (gen and tg and dec is not None) else None
        status = ("silent" if t0 is None else
                  ("hit" if dec < tg else "too late") if (gen and tg) else "FALSE ALARM")
        rows.append({"run": name, "generalises": gen, "group": group, "t0": t0,
                     "decided": dec, "lead": lead, "status": status})
    return pd.DataFrame(rows)


def evaluate(fn, params, runs, win_steps=WIN_STEPS):
    """Returns a frame with one row per run: fired, decision time, lead, censoring."""
    rows = []
    for name, path, gen, group in runs:
        r, d, tm, tg = trace(name, path, win_steps)
        r, d = scope(r, d, tm, tg, bool(gen), win_steps)
        if len(d) < 20:
            rows.append({"run": name, "generalises": gen, "group": group,
                         "n_windows": len(d), "t0": None, "decided": None,
                         "lead": None, "status": "no window"})
            continue
        t0, dec = fn(r, d, **params)
        lead = (tg - dec) if (gen and tg and dec is not None) else None
        if t0 is None:
            status = "silent"
        elif gen and tg:
            status = "hit" if dec < tg else "too late"
        else:
            status = "FALSE ALARM"
        rows.append({"run": name, "generalises": gen, "group": group,
                     "n_windows": len(d), "t0": t0, "decided": dec, "lead": lead,
                     "status": status})
    return pd.DataFrame(rows)


def part_3():
    rule("3  The revised criterion on the default parameters")
    p = dict(H=600, delta=0.20, beta_rel=0.10, Q=300, delta_hold=0.15, disp_max=0.08)
    print(f"  H = {p['H']} steps (= the estimator window, so the two intervals use")
    print(f"  disjoint raw data), delta = {p['delta']:.0%}, beta_rel = {p['beta_rel']:.2f},")
    print(f"  Q = {p['Q']} steps, delta_hold = {p['delta_hold']:.0%}, "
          f"MAD/median <= {p['disp_max']:.0%}.\n")
    live = [x for x in RUNS if x[3] != "extended"]
    a = evaluate(proposed, {k: v for k, v in p.items() if k != "disp_max"}, live)
    b = evaluate(revised, p, live)
    print(f"  {'run':<12}{'gen':>4}{'group':>10}"
          f"{'proposed t0':>13}{'lead(t0)':>10}{'revised t0':>12}{'decided':>9}"
          f"{'honest lead':>13}   status")
    for (_, x), (_, y) in zip(a.iterrows(), b.iterrows()):
        naive = (y.lead + p["Q"]) if y.lead is not None else None
        print(f"  {x.run:<12}{'Y' if x.generalises else 'N':>4}{x.group:>10}"
              f"{str(x.t0):>13}{str(naive):>10}{str(y.t0):>12}{str(y.decided):>9}"
              f"{str(y.lead):>13}   {y.status}")
    a.to_csv(OUT / "exp5_3_proposed.csv", index=False)
    b.to_csv(OUT / "exp5_3_revised.csv", index=False)
    for label, f in (("proposed", a), ("revised", b)):
        pos = f[f.generalises == True]                            # noqa: E712
        neg = f[f.generalises == False]                            # noqa: E712
        print(f"\n  {label:<9} recall {int((pos.status == 'hit').sum())}/{len(pos)}"
              f"   false alarms {int((neg.status == 'FALSE ALARM').sum())}/{len(neg)}")
    print("\n  'lead(t0)' is what would be reported if the hold condition were treated")
    print("  as free; 'honest lead' subtracts Q, because that is when the criterion is")
    print("  actually decidable.")


# --------------------------------------------- 4. calibration on positives, then a test

GRID = [dict(H=h, delta=de, beta_rel=be, Q=q, delta_hold=dh, disp_max=dm)
        for h in (600, 900)
        for de in (0.10, 0.15, 0.20, 0.25, 0.30)
        for be in (0.05, 0.10, 0.20)
        for q in (300, 600)
        for dh in (0.10, 0.15, 0.20)
        for dm in (0.06, 0.08, 0.12)]


def strictness(p):
    """How hard a setting is to satisfy. Used to break ties WITHOUT looking at negatives.

    Every term is oriented so that larger = stricter: a bigger required fall, a steeper
    required slope, a longer and deeper hold, a tighter baseline. The scale of each term
    is its own range over the grid, so no single parameter dominates the ordering.
    """
    return (p["delta"] / 0.30 + p["beta_rel"] / 0.20 + p["Q"] / 600
            + p["delta_hold"] / 0.20 + (0.12 - p["disp_max"]) / 0.06)


def part_4():
    rule("4  Calibration on generalising runs only, then one look at the controls")
    print("  Calibration set : the 8 `conf` runs -- one configuration, seeds only.")
    print("  Test set        : the 3 headline runs, and the 3 controls.")
    print("  Rule for choosing parameters: among settings reaching the target recall on")
    print("  the CALIBRATION POSITIVES, take the strictest. Negatives are not consulted,")
    print("  so the false-alarm count below is out of sample.\n")
    calib = [x for x in RUNS if x[3] == "conf"]
    test = [x for x in RUNS if x[3] in ("headline", "control")]
    rows = []
    for p in GRID:
        f = evaluate_fast(p, calib)
        hits = int((f.status == "hit").sum())
        leads = f.lead.dropna()
        rows.append({**p, "calib_hits": hits, "calib_n": len(f),
                     "calib_median_lead": float(leads.median()) if len(leads) else np.nan,
                     "strictness": strictness(p)})
    cal = pd.DataFrame(rows)
    cal.to_csv(OUT / "exp5_4_calibration.csv", index=False)
    print(f"  {len(cal)} settings evaluated on the calibration set.")
    print(f"  {'target recall':>14}{'settings':>10}{'chosen delta':>14}{'beta':>7}"
          f"{'Q':>6}{'hold':>7}{'disp':>7}{'calib lead':>12}")
    chosen = {}
    for target in (8, 7, 6):
        ok = cal[cal.calib_hits >= target]
        if not len(ok):
            print(f"  {f'{target}/8':>14}{0:>10}   none")
            continue
        best = ok.sort_values("strictness", ascending=False).iloc[0]
        chosen[target] = {k: best[k] for k in
                          ("H", "delta", "beta_rel", "Q", "delta_hold", "disp_max")}
        chosen[target]["H"] = int(chosen[target]["H"])
        chosen[target]["Q"] = int(chosen[target]["Q"])
        print(f"  {f'{target}/8':>14}{len(ok):>10}{best.delta:>14.2f}{best.beta_rel:>7.2f}"
              f"{int(best.Q):>6}{best.delta_hold:>7.2f}{best.disp_max:>7.2f}"
              f"{best.calib_median_lead:>12.0f}")

    print("\n  Frozen, then applied once to the test set:")
    out = []
    for target, p in chosen.items():
        f = evaluate_fast(p, test)
        pos = f[f.generalises == True]                             # noqa: E712
        neg = f[f.generalises == False]                            # noqa: E712
        leads = pos.lead.dropna()
        print(f"\n  -- calibrated at recall {target}/8: {p}")
        print(f"     {'run':<12}{'gen':>4}{'t0':>8}{'decided':>9}{'lead':>8}   status")
        for _, x in f.iterrows():
            print(f"     {x.run:<12}{'Y' if x.generalises else 'N':>4}{str(x.t0):>8}"
                  f"{str(x.decided):>9}{str(x.lead):>8}   {x.status}")
        print(f"     test recall {int((pos.status == 'hit').sum())}/{len(pos)}"
              f"   false alarms {int((neg.status == 'FALSE ALARM').sum())}/{len(neg)}"
              f"   median lead {leads.median() if len(leads) else float('nan'):.0f}")
        out.append({"target": target, **p,
                    "test_hits": int((pos.status == "hit").sum()), "test_pos": len(pos),
                    "test_fp": int((neg.status == "FALSE ALARM").sum()),
                    "test_neg": len(neg),
                    "median_lead": float(leads.median()) if len(leads) else np.nan})
    pd.DataFrame(out).to_csv(OUT / "exp5_4_test.csv", index=False)
    print("\n  The test set has three positives and three controls, one of which is")
    print("  right-censored (exp3). No false-alarm rate estimated on it can be quoted;")
    print("  what the table supports is only that the criterion is not obviously broken.")


# ----------------------------------------------------------------------- 5. sensitivity

def part_5():
    rule("5  Sensitivity: does the conclusion depend on any one parameter?")
    base = dict(H=600, delta=0.20, beta_rel=0.10, Q=300, delta_hold=0.15, disp_max=0.08)
    live = [x for x in RUNS if x[3] != "extended"]
    print(f"  base {base}\n")
    print(f"  {'parameter':<12}{'value':>8}{'recall':>9}{'false alarms':>14}"
          f"{'median lead':>13}   fires on")
    rows = []
    for key, values in (("H", (300, 600, 900, 1200)),
                        ("delta", (0.10, 0.15, 0.20, 0.25, 0.30)),
                        ("beta_rel", (0.0, 0.05, 0.10, 0.20)),
                        ("Q", (0, 300, 600, 900)),
                        ("delta_hold", (0.05, 0.15, 0.20, 0.25, 0.30, 0.40)),
                        ("disp_max", (0.06, 0.08, 0.12, 1.0))):
        for v in values:
            p = {**base, key: v}
            f = evaluate(revised, p, live)
            pos = f[f.generalises == True]                          # noqa: E712
            neg = f[f.generalises == False]                         # noqa: E712
            fp = neg[neg.status == "FALSE ALARM"].run.tolist()
            leads = pos.lead.dropna()
            print(f"  {key:<12}{v:>8}{f'{int((pos.status==chr(104)+chr(105)+chr(116)).sum())}/{len(pos)}':>9}"
                  f"{f'{len(fp)}/3':>14}"
                  f"{(leads.median() if len(leads) else float('nan')):>13.0f}"
                  f"   {', '.join(fp) if fp else '-'}")
            rows.append({"parameter": key, "value": v,
                         "recall": int((pos.status == "hit").sum()), "n_pos": len(pos),
                         "fp": len(fp), "fires_on": ";".join(fp),
                         "median_lead": float(leads.median()) if len(leads) else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "exp5_5_sensitivity.csv", index=False)
    print("\n  Read the last column. A parameter whose only effect is to switch")
    print("  `lowdata15` on and off is a threshold sitting on one run.")


SECTIONS = {"1": part_1, "2": part_2, "3": part_3, "4": part_4, "5": part_5}


def main(argv):
    for name in (argv[1:] or list(SECTIONS)):
        if name not in SECTIONS:
            print(f"unknown part {name!r}")
            return 2
        SECTIONS[name]()
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
