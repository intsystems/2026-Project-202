"""Does 'sustained MG-dimension drop AND function-space velocity above a floor' work?

The proposal is the minimal repair of the criterion: keep one geometry statistic, add one
guard against the failure mode the geometry statistic cannot see. Both conjuncts are
evaluated exactly as a real detector would have to run them.

  * strictly causal -- a window is labelled by its right edge and may never contain data
    past that step; the running peak and the running median use history only;
  * strictly scoped -- a window must lie wholly inside the plateau, so nothing that fires
    is reading the memorisation transient or the generalisation it is meant to anticipate;
  * no run-global normalisation anywhere -- 'below the running peak' is admissible,
    'below the global peak' is not, because the global peak is in the future.

Runs. The velocity conjunct needs the logit probe, which exists for seven runs. The
dimension conjunct needs only a training log, so it is evaluated on fourteen: the three
grokking seeds, the eight fixed-configuration seeds of edm_validation/results/conf that
vary only the split and the init, and the three never-generalising controls. That matters,
because the conf set is the only place in the repo where eleven real plateaus of
1600-5700 steps can be tested against one criterion.

    python exp_composite_criterion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension                     # noqa: E402

RES = HERE / "results"
CONF = HERE.parent / "edm_validation" / "results" / "conf"
OUT = RES / "audit"
OUT.mkdir(parents=True, exist_ok=True)

K, MAX_E, THEILER, STRIDE = 5, 15, 0, 10
LOG_EVERY = 10

# name -> (training log, probe file or None, generalises)
RUNS = {
    "grok":      (RES / "grok_train.csv",       RES / "grok_probe.csv",       True),
    "grok_s1":   (RES / "grok_seed1_train.csv", RES / "grok_seed1_probe.csv", True),
    "grok_s2":   (RES / "grok_seed2_train.csv", RES / "grok_seed2_probe.csv", True),
    "lowdata15": (RES / "lowdata15_train.csv",  RES / "lowdata15_probe.csv",  False),
    "lowdata20": (RES / "lowdata20_train.csv",  RES / "lowdata20_probe.csv",  False),
    "wd0":       (RES / "wd0_train.csv",        RES / "wd0_probe.csv",        False),
}
for p in sorted(CONF.glob("s*_i*.csv")):
    RUNS[f"conf_{p.stem}"] = (p, None, True)

PROBED = [r for r, (_, pr, _) in RUNS.items() if pr is not None]


# --------------------------------------------------------------------------- plumbing

def first_sustained(steps, acc, thr=0.95, smooth_w=5):
    a = pd.Series(acc).rolling(smooth_w, min_periods=1, center=True).mean().to_numpy()
    ok = a >= thr
    idx = None
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        idx = i
    return int(steps[idx]) if idx is not None else None


def events(run):
    df = pd.read_csv(RUNS[run][0])
    s = df["step"].to_numpy()
    return df, s, first_sustained(s, df["train_acc"].to_numpy()), \
        first_sustained(s, df["val_acc"].to_numpy())


def mg_trace(series, steps, window, max_E=MAX_E, k=K):
    """MacKay-Ghahramani estimate per window, labelled by the window's right edge."""
    right, dims, left = [], [], []
    for start in range(0, len(series) - window + 1, STRIDE):
        dims.append(mle_intrinsic_dimension(
            series[start:start + window], tau=1, max_E=max_E, k_neighbors=k,
            correction="mackay_ghahramani", theiler_window=THEILER,
            rng=np.random.default_rng(0)))
        left.append(steps[start])
        right.append(steps[start + window - 1])
    return np.asarray(left), np.asarray(right), np.asarray(dims, dtype=float)


_TRACE_CACHE = {}


def cached_trace(run, window, max_E=MAX_E, k=K):
    key = (run, window, max_E, k)
    if key not in _TRACE_CACHE:
        df, steps, _, _ = events(run)
        _TRACE_CACHE[key] = mg_trace(df["weight_norm"].to_numpy(), steps, window, max_E, k)
    return _TRACE_CACHE[key]


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


# --------------------------------------------------------------- the two conjuncts

def drop_detector(left, right, dims, t_mem, t_stop, drop, sustain, min_hist=3):
    """First step at which the estimate has stayed `drop` below its running peak.

    Only windows lying wholly inside [t_mem, t_stop] are considered, and the running
    peak is built from those windows alone. Returns (step, n_usable_windows).
    """
    inside = (left >= t_mem) & (right <= t_stop)
    r, d = right[inside], dims[inside]
    if len(r) == 0:
        return None, 0
    streak = 0
    for i in range(len(r)):
        if i < min_hist:
            continue
        hist = d[:i]
        if not np.isfinite(hist).any() or not np.isfinite(d[i]):
            continue
        streak = streak + 1 if d[i] <= (1 - drop) * np.nanmax(hist) else 0
        if streak >= sustain:
            return int(r[i]), len(r)
    return None, len(r)


def velocity_ok(steps_v, v, t_mem, t_stop, floor_frac, min_hist=5):
    """Steps at which V_t is at least `floor_frac` of its running post-memorisation median.

    A one-sided floor, never a score: the audit's measurement is that more velocity is
    evidence AGAINST generalisation, so any monotone use of V_t fires hardest on the
    runs it is supposed to exclude.
    """
    sel = (steps_v >= t_mem) & (steps_v <= t_stop)
    s, vv = steps_v[sel], v[sel]
    ok = np.zeros(len(s), dtype=bool)
    for i in range(len(s)):
        if i < min_hist or not np.isfinite(vv[i]):
            continue
        med = np.nanmedian(vv[:i])
        ok[i] = np.isfinite(med) and med > 0 and vv[i] >= floor_frac * med
    return s, ok


def load_velocity(run):
    _, probe, _ = RUNS[run]
    if probe is None:
        return None, None
    p = pd.read_csv(probe)
    return p["step"].to_numpy(), p["train_velocity"].to_numpy()


def horizon(t_mem, t_gen, cap=20000):
    """The interval a detector is allowed to see: the plateau, or an equal-length slice."""
    return t_gen if t_gen is not None else cap


# ------------------------------------------------- 1. does a drop exist to detect?

def section_1():
    rule("1.  Is there a drop inside the plateau at all?")
    print("  Direction of the MG estimate over the causal windows that lie wholly inside")
    print("  (t_mem, t_gen). A criterion built on a fall cannot work where the trace rises.\n")
    rows = []
    for win_steps in (3000, 1000, 600):
        win = win_steps // LOG_EVERY
        print(f"  -- window {win_steps} steps ({win} samples)")
        print(f"     {'run':<13}{'gen':>4}{'gap':>7}{'n win':>7}{'first':>8}{'last':>8}"
              f"{'peak':>7}{'min after peak':>16}{'max drop':>10}")
        for run in RUNS:
            df, steps, t_mem, t_gen = events(run)
            if t_mem is None:
                continue
            left, right, dims = cached_trace(run, win)
            inside = (left >= t_mem) & (right <= horizon(t_mem, t_gen))
            d = dims[inside]
            gen = "Y" if RUNS[run][2] else "N"
            gap = (t_gen - t_mem) if t_gen else None
            if len(d) < 2 or not np.isfinite(d).any():
                print(f"     {run:<13}{gen:>4}{str(gap):>7}{len(d):>7}"
                      f"{'--':>8}{'--':>8}{'--':>7}{'--':>16}{'--':>10}")
                rows.append({"run": run, "window": win_steps, "n_windows": len(d)})
                continue
            peak_i = int(np.nanargmax(d))
            after = d[peak_i:]
            max_drop = 1 - np.nanmin(after) / d[peak_i]
            print(f"     {run:<13}{gen:>4}{str(gap):>7}{len(d):>7}{d[0]:>8.2f}{d[-1]:>8.2f}"
                  f"{d[peak_i]:>7.2f}{np.nanmin(after):>16.2f}{max_drop:>9.0%}")
            rows.append({"run": run, "window": win_steps, "generalises": RUNS[run][2],
                         "gap": gap, "n_windows": len(d), "first": d[0], "last": d[-1],
                         "peak": d[peak_i], "min_after_peak": float(np.nanmin(after)),
                         "max_drop": max_drop})
        print()
    pd.DataFrame(rows).to_csv(OUT / "composite_1_drops.csv", index=False)
    print("  'max drop' is the largest fall below the running peak available to ANY")
    print("  threshold rule, so it is an upper bound on what the first conjunct can do.")


# --------------------------------------------------- 2. the drop conjunct, swept

def section_2():
    rule("2.  The MG-drop conjunct alone, swept over its two free parameters")
    print("  recall  = generalising runs where it fires inside the plateau")
    print("  FP      = never-generalising runs where it fires in an equally long window")
    print("  n/a     = runs where the window does not fit inside the plateau at all\n")
    rows = []
    for win_steps in (3000, 1000, 600):
        win = win_steps // LOG_EVERY
        print(f"  -- window {win_steps} steps")
        print(f"     {'drop':>6}{'sustain':>9}{'recall':>18}{'FP':>10}{'n/a':>6}"
              f"{'median lead':>14}{'lead range':>18}")
        for drop in (0.10, 0.20, 0.30, 0.50):
            for sustain in (1, 3, 5):
                fires, na, fp, leads = 0, 0, 0, []
                pos = 0
                for run in RUNS:
                    df, steps, t_mem, t_gen = events(run)
                    if t_mem is None:
                        continue
                    left, right, dims = cached_trace(run, win)
                    generalises = RUNS[run][2]
                    if generalises and t_gen is not None:
                        pos += 1
                        t_fire, n = drop_detector(left, right, dims, t_mem, t_gen,
                                                  drop, sustain)
                        if n == 0:
                            na += 1
                        elif t_fire is not None:
                            fires += 1
                            leads.append(t_gen - t_fire)
                    elif not generalises:
                        # judge negatives over a slice as long as the longest true plateau
                        t_fire, n = drop_detector(left, right, dims, t_mem,
                                                  t_mem + 12070, drop, sustain)
                        fp += int(t_fire is not None)
                lead_s = (f"{np.median(leads):.0f}" if leads else "--")
                rng_s = (f"{min(leads)}-{max(leads)}" if leads else "--")
                print(f"     {drop:>6.0%}{sustain:>9}{f'{fires}/{pos}':>18}"
                      f"{f'{fp}/3':>10}{na:>6}{lead_s:>14}{rng_s:>18}")
                rows.append({"window": win_steps, "drop": drop, "sustain": sustain,
                             "fires": fires, "positives": pos, "fp": fp, "na": na,
                             "median_lead": float(np.median(leads)) if leads else np.nan})
        print()
    pd.DataFrame(rows).to_csv(OUT / "composite_2_sweep.csv", index=False)


# ------------------------------------------------ 3. what the velocity floor removes

def section_3():
    rule("3.  The V_t floor alone")
    print("  Fraction of post-memorisation steps at which V_t stays above the given")
    print("  fraction of its own running median. The floor is a filter, not a detector:")
    print("  the question is only which runs it is capable of removing.\n")
    fracs = (0.10, 0.25, 0.50)
    print(f"  {'run':<13}{'gen':>4}" + "".join(f"{f'floor {f:.0%}':>13}" for f in fracs)
          + f"{'V median':>12}{'V last/med':>12}")
    rows = []
    for run in PROBED:
        df, steps, t_mem, t_gen = events(run)
        sv, v = load_velocity(run)
        cells = []
        for f in fracs:
            _, ok = velocity_ok(sv, v, t_mem, horizon(t_mem, t_gen), f)
            cells.append(f"{ok.mean():>12.0%}")
            rows.append({"run": run, "floor": f, "frac_above": float(ok.mean())})
        sel = (sv >= t_mem) & (sv <= horizon(t_mem, t_gen))
        med = np.nanmedian(v[sel])
        tail = np.nanmedian(v[sel][-20:]) if sel.sum() >= 20 else np.nan
        print(f"  {run:<13}{'Y' if RUNS[run][2] else 'N':>4}" + "".join(cells)
              + f"{med:>12.2e}{tail / med:>12.2f}")
    pd.DataFrame(rows).to_csv(OUT / "composite_3_velocity.csv", index=False)
    print("\n  A run-relative floor cannot remove a run whose velocity is high and steady.")
    print("  It removes only runs that decelerate relative to their own history.")


# ------------------------------------------------------------ 4. the composite

def section_4():
    rule("4.  The composite: MG drop AND V_t above floor, on the seven probed runs")
    print("  Both conjuncts must hold at the same step. Columns give the step each")
    print("  conjunct first fires and whether the composite fires before t_gen.\n")
    rows = []
    for win_steps in (3000, 1000, 600):
        win = win_steps // LOG_EVERY
        print(f"  -- window {win_steps} steps, drop 20% sustained 3, floor 25%")
        print(f"     {'run':<13}{'gen':>4}{'t_mem':>7}{'t_gen':>7}{'drop@':>8}"
              f"{'V ok@':>8}{'both@':>8}{'lead':>8}   verdict")
        for run in PROBED:
            df, steps, t_mem, t_gen = events(run)
            stop = horizon(t_mem, t_gen) if RUNS[run][2] else t_mem + 12070
            left, right, dims = cached_trace(run, win)
            t_drop, n_win = drop_detector(left, right, dims, t_mem, stop, 0.20, 3)
            sv, ok = velocity_ok(*load_velocity(run), t_mem, stop, 0.25)
            t_both = None
            if t_drop is not None:
                after = (sv >= t_drop) & ok
                t_both = int(sv[after][0]) if after.any() else None
            gen = RUNS[run][2]
            if n_win == 0:
                verdict = "no window fits the plateau"
            elif t_both is None:
                verdict = "silent" + (" (miss)" if gen and t_gen else " (correct)")
            elif gen and t_gen:
                verdict = "fires before t_gen" if t_both < t_gen else "fires late"
            else:
                verdict = "FALSE ALARM"
            lead = (t_gen - t_both) if (t_both and gen and t_gen) else None
            print(f"     {run:<13}{'Y' if gen else 'N':>4}{t_mem:>7}{str(t_gen):>7}"
                  f"{str(t_drop):>8}{str(int(sv[ok][0]) if ok.any() else None):>8}"
                  f"{str(t_both):>8}{str(lead):>8}   {verdict}")
            rows.append({"window": win_steps, "run": run, "generalises": gen,
                         "t_mem": t_mem, "t_gen": t_gen, "t_drop": t_drop,
                         "t_both": t_both, "lead": lead, "n_windows": n_win,
                         "verdict": verdict})
        print()
    pd.DataFrame(rows).to_csv(OUT / "composite_4_composite.csv", index=False)
    print("  The marginal value of the second conjunct is the number of FALSE ALARM rows")
    print("  it converts to 'silent (correct)' without converting a hit into a miss.")


# ---------------------------------------------------------------- 5. what n would settle it

def section_5():
    rule("5.  How many runs would settle this?")
    print("  The evaluation set is 3 generalising probed runs and 3 controls; the drop")
    print("  conjunct alone reaches 11 and 3. Binomial arithmetic on what that supports.\n")
    for k_pos, n_pos in ((11, 11), (10, 11), (3, 3)):
        lo, hi = _jeffreys(k_pos, n_pos)
        print(f"  {k_pos}/{n_pos} detections -> recall 95% interval [{lo:.2f}, {hi:.2f}]")
    print()
    for k_fp, n_neg in ((0, 3), (0, 10), (0, 40)):
        lo, hi = _jeffreys(k_fp, n_neg)
        print(f"  {k_fp}/{n_neg} false alarms -> FP rate 95% interval [{lo:.2f}, {hi:.2f}]")
    print("\n  Three clean controls bound the false-alarm rate only below about 0.6. A")
    print("  precursor claim needs the interval below the base rate it is competing with,")
    print("  which for 'does this run generalise' is roughly one half.")


def _jeffreys(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k + 0.5, n - k + 0.5)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 0.5, n - k + 0.5)
    return lo, hi


# ------------------------------------- 6. does the second conjunct change any decision?

def section_6():
    rule("6.  Marginal contribution of the velocity floor, over the whole grid")
    print("  For every (window, drop, sustain, floor) the composite is compared with the")
    print("  drop conjunct alone. A conjunct that never changes a decision is not a fix,")
    print("  whatever it measures.\n")
    print(f"  {'window':>7}{'drop':>6}{'sustain':>8}{'floor':>7}"
          f"{'drop alone':>13}{'composite':>12}{'FP removed':>12}{'hits lost':>11}")
    rows, best = [], []
    for win_steps in (3000, 1000, 600):
        win = win_steps // LOG_EVERY
        for drop in (0.10, 0.20, 0.30, 0.50):
            for sustain in (1, 3, 5):
                for floor in (0.10, 0.25, 0.50, 0.75, 0.90):
                    hit_a = hit_c = fp_a = fp_c = 0
                    for run in PROBED:
                        df, steps, t_mem, t_gen = events(run)
                        gen = RUNS[run][2]
                        stop = t_gen if gen else t_mem + 12070
                        left, right, dims = cached_trace(run, win)
                        t_drop, n = drop_detector(left, right, dims, t_mem, stop,
                                                  drop, sustain)
                        sv, ok = velocity_ok(*load_velocity(run), t_mem, stop, floor)
                        t_both = None
                        if t_drop is not None:
                            after = (sv >= t_drop) & ok
                            t_both = int(sv[after][0]) if after.any() else None
                        if gen:
                            hit_a += t_drop is not None
                            hit_c += t_both is not None
                        else:
                            fp_a += t_drop is not None
                            fp_c += t_both is not None
                    rows.append({"window": win_steps, "drop": drop, "sustain": sustain,
                                 "floor": floor, "hits_drop_only": hit_a,
                                 "hits_composite": hit_c, "fp_drop_only": fp_a,
                                 "fp_composite": fp_c, "fp_removed": fp_a - fp_c,
                                 "hits_lost": hit_a - hit_c})
                    if fp_a - fp_c or hit_a - hit_c:
                        best.append(rows[-1])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "composite_6_marginal.csv", index=False)
    shown = frame[(frame.floor.isin((0.25, 0.90))) & (frame.sustain == 3)]
    for _, r in shown.iterrows():
        print(f"  {r['window']:>7.0f}{r['drop']:>6.0%}{r['sustain']:>8.0f}"
              f"{r['floor']:>7.0%}"
              f"{f'{r.hits_drop_only:.0f} hit {r.fp_drop_only:.0f} fp':>13}"
              f"{f'{r.hits_composite:.0f} hit {r.fp_composite:.0f} fp':>12}"
              f"{r.fp_removed:>12.0f}{r.hits_lost:>11.0f}")
    print(f"\n  {len(frame)} parameter combinations evaluated.")
    print(f"  combinations where the velocity floor removes a false alarm: "
          f"{int((frame.fp_removed > 0).sum())}")
    print(f"  combinations where it costs a detection:                     "
          f"{int((frame.hits_lost > 0).sum())}")
    if (frame.fp_removed > 0).any():
        sub = frame[frame.fp_removed > 0]
        print("  floors at which it ever helps:", sorted(sub.floor.unique()))
    print("\n  The floor is a no-op wherever the drop conjunct is already silent, and the")
    print("  only control it can reach -- wd0, whose velocity falls by ~500x -- is the one")
    print("  control whose weight norm is a straight line, so the drop conjunct never")
    print("  fires there either. The two conjuncts fail on the same run and neither")
    print("  reaches lowdata15 or lowdata20, which are the controls that matter.")


# ---------------------------------- 7. the best-looking cell, and why it looks that way

def section_7():
    rule("7.  The best cell of the section-2 sweep, run by run")
    print("  Section 2 contains one cell with 8/11 recall and 0/3 false alarms: window")
    print("  600, drop 30%, sustain 5. Below it and its two neighbours, per run, so the")
    print("  stability of that cell can be read directly rather than inferred.\n")
    cells = [(600, 0.30, 1), (600, 0.30, 3), (600, 0.30, 5), (600, 0.40, 5)]
    print(f"  {'run':<13}{'gen':>4}{'gap':>7}" +
          "".join(f"{f'd{int(d * 100)}/s{s}':>10}" for _, d, s in cells))
    rows = []
    for run in RUNS:
        df, steps, t_mem, t_gen = events(run)
        if t_mem is None:
            continue
        gen = RUNS[run][2]
        stop = t_gen if (gen and t_gen) else t_mem + 12070
        cells_out = []
        for win_steps, drop, sustain in cells:
            left, right, dims = cached_trace(run, win_steps // LOG_EVERY)
            t_fire, n = drop_detector(left, right, dims, t_mem, stop, drop, sustain)
            if n == 0:
                cells_out.append(f"{'n/a':>10}")
            elif t_fire is None:
                cells_out.append(f"{'-':>10}")
            else:
                lead = (t_gen - t_fire) if (gen and t_gen) else None
                cells_out.append(f"{f'+{lead}' if lead else 'FIRE':>10}")
            rows.append({"run": run, "generalises": gen, "window": win_steps,
                         "drop": drop, "sustain": sustain, "t_fire": t_fire,
                         "n_windows": n})
        print(f"  {run:<13}{'Y' if gen else 'N':>4}"
              f"{str(t_gen - t_mem if (gen and t_gen) else None):>7}" + "".join(cells_out))
    pd.DataFrame(rows).to_csv(OUT / "composite_7_bestcell.csv", index=False)
    print("\n  '+N' is a detection N steps before t_gen; 'FIRE' on a control row is a")
    print("  false alarm; '-' is silence; 'n/a' means no window fits inside the plateau.")
    print("\n  Selection arithmetic. The sweep has 4 drops x 3 sustains x 3 windows = 36")
    print("  cells. If each of the three controls fired independently with probability")
    print("  one half, a cell with no false alarms would arise with probability 1/8, so")
    print("  about 4 of the 36 cells would show 0/3 by chance alone. One such cell is")
    print("  what the sweep found. Its neighbour at sustain 3 shows 2/3. That is a")
    print("  threshold sitting exactly between two controls, not a separation.")


# ------------------------------------- 8. is the lead time a prediction or a stopwatch?

def section_8():
    rule("8.  Does the detector predict t_gen, or just count steps since t_mem?")
    print("  A precursor is useful because it fires a knowable distance before the event.")
    print("  If instead it fires a fixed distance AFTER memorisation, the 'lead' is just")
    print("  the gap minus a constant -- and the gap is what nobody knows in advance.\n")
    print("  Capping the detector at t_gen -- as sections 2, 4 and 7 do, to keep it from")
    print("  reading past the event -- also throws away every run that would have fired")
    print("  late. Correlating a lead time over the survivors is then circular. Below the")
    print("  detector instead runs to the end of training, so every generalising run")
    print("  contributes a firing step whether or not it beat t_gen.\n")
    from scipy.stats import spearmanr
    rows = []
    for drop, sustain in ((0.30, 1), (0.30, 5)):
        for run in RUNS:
            df, steps, t_mem, t_gen = events(run)
            if t_mem is None or not RUNS[run][2] or t_gen is None:
                continue
            left, right, dims = cached_trace(run, 60)
            t_fire, _ = drop_detector(left, right, dims, t_mem, int(steps[-1]),
                                      drop, sustain)
            rows.append({"run": run, "drop": drop, "sustain": sustain,
                         "gap": t_gen - t_mem,
                         "delay_after_mem": (t_fire - t_mem) if t_fire else np.nan,
                         "lead_before_gen": (t_gen - t_fire) if t_fire else np.nan,
                         "in_time": bool(t_fire and t_fire < t_gen)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "composite_8_timing.csv", index=False)
    for (drop, sustain), sub in frame.groupby(["drop", "sustain"]):
        sub = sub.dropna(subset=["delay_after_mem"])
        d, g, l = sub.delay_after_mem, sub.gap, sub.lead_before_gen
        print(f"  -- drop {drop:.0%}, sustain {sustain}, window 600, uncapped, "
              f"{len(sub)}/11 runs fire ({int(sub.in_time.sum())} before t_gen)")
        print(f"     delay after t_mem  {d.min():>6.0f}-{d.max():<6.0f} "
              f"(spread {d.max() / d.min():4.1f}x)   rho with gap "
              f"{spearmanr(d, g).statistic:+.2f}")
        print(f"     lead before t_gen  {l.min():>6.0f}-{l.max():<6.0f} "
              f"(spread {abs(l.max() / l.min()):4.1f}x)   rho with gap "
              f"{spearmanr(l, g).statistic:+.2f}")
        print(f"     gap itself         {g.min():>6.0f}-{g.max():<6.0f} "
              f"(spread {g.max() / g.min():4.1f}x)")
        print()
    print("  Baseline with no dimension estimate in it: declare at t_mem + delta.")
    print(f"  {'delta':>7}{'detections':>13}{'false alarms':>15}{'median lead':>14}")
    for delta in (1000, 1500, 2000, 2500, 3000):
        hits, leads = 0, []
        pos = 0
        for run in RUNS:
            df, steps, t_mem, t_gen = events(run)
            if t_mem is None or not RUNS[run][2] or t_gen is None:
                continue
            pos += 1
            if t_mem + delta < t_gen:
                hits += 1
                leads.append(t_gen - t_mem - delta)
        print(f"  {delta:>7}{f'{hits}/{pos}':>13}{'3/3':>15}"
              f"{np.median(leads) if leads else float('nan'):>14.0f}")
    print("\n  The baseline cannot reject a control, and that is the entire remaining case")
    print("  for the dimension statistic: not when it fires, but whether it stays quiet.")


SECTIONS = {"1": section_1, "2": section_2, "3": section_3, "4": section_4,
            "5": section_5, "6": section_6, "7": section_7, "8": section_8}


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
