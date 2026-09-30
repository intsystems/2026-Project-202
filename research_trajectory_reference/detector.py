"""Online simplification detector on the parameter-norm log (E6).

PROTOCOL (fixed before any detection result was computed).

The question a practitioner asks is not "did the median fall after a step I already
know about" but "does a running monitor raise an alarm when training simplifies,
how late, and how often without a reason". This script answers that on logs
already recorded, without retraining:

  calibration  E4 runs, seeds 0-3 (cifar_events.py): no-event arms base, batch_up;
               event arms lr_step, freeze, prune. Only the parameter-norm log.
  test         E5 runs, seeds 10-13 (cifar_graded.py): all 11 arms plus the
               observer controls (x10 scale, smoothing) built from base.
Both are scored with the SAME MG configuration (article: E=20, tau=1, k=20,
Theiler = embedding span), windows of 1 000 steps, stride 500.

Rule. For window k let MG_k be its estimate. The statistic is
    D_k = median(MG_{k-M+1..k}) / median(MG_{k-M-B+1..k-M}) - 1,
a relative drop of the recent block against the block before it. An alarm is
raised at the first k with D_k < -delta, reported at the step where window k ends.
Only drops alarm: the target is simplification. Windows that end before step 1 500
are warm-up and never enter a reference block.
(M, B, delta) are chosen on the calibration set: among M in {2,3,4}, B in {3,4,6},
delta = the smallest value giving no alarm in any calibration no-event run, plus a
margin of 0.02; the (M, B) pair maximising detected calibration events, then the
shortest median delay, wins. The chosen triple is frozen and applied to the test set.

Test metrics. Event run: hit if the first alarm is raised after the event (step
4 000) and at most 5 000 steps later; an alarm before the event is a false alarm.
No-event run: any alarm is a false alarm. Reported per arm: hit rate, median delay,
false-alarm rate, and the same detector built on simple statistics of the same log
(trend crossings, lag-1 autocorrelation, detrended std), calibrated identically,
with the direction of alarm chosen on the calibration set.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "code"))
sys.path.insert(0, str(HERE))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from cifar_events import simple  # noqa: E402

CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
W, S, EVENT, WARM, HORIZON = 1000, 500, 4000, 1500, 5000
OUT = HERE / "results_detector"
CAL_EVENTS = ("lr_step", "freeze", "prune")
CAL_NULL = ("base", "batch_up")
TEST_NULL = ("base", "batch_up", "scale", "smooth")
STATS = ("MG", "crossings", "lag1", "det_std")


def score_log(x):
    rows = []
    for a in range(0, len(x) - W + 1, S):
        seg = x[a:a + W]
        rows.append({"start": a, "end": a + W, "MG": estimate(seg, CFG).MG, **simple(seg)})
    return pd.DataFrame(rows)


def calibration_windows():
    rows = []
    for f in sorted((HERE / "results_cifar").glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        if arm not in CAL_EVENTS + CAL_NULL:
            continue
        w = score_log(np.load(f)["param_norm"])
        w["arm"], w["seed"] = arm, int(seed)
        rows.append(w)
    return pd.concat(rows)


def test_windows():
    rows = []
    for f in sorted((HERE / "results_graded").glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        x = np.load(f)["param_norm"]
        variants = {arm: x}
        if arm == "base":
            sc = x.copy(); sc[EVENT:] *= 10
            sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[EVENT:] = c[EVENT:]
            variants.update({"scale": sc, "smooth": sm})
        for name, v in variants.items():
            w = score_log(v)
            w["arm"], w["seed"] = name, int(seed)
            rows.append(w)
    return pd.concat(rows)


def drop_series(g, stat, M, B, sign):
    """D_k for one run; sign=+1 alarms on drops, -1 on rises (for competitors)."""
    g = g.sort_values("start")
    v, ends = g[stat].to_numpy(), g.end.to_numpy()
    ok = ends > WARM
    v, ends = v[ok], ends[ok]
    out = []
    for k in range(M + B - 1, len(v)):
        cur = np.median(v[k - M + 1:k + 1])
        ref = np.median(v[k - M - B + 1:k - M + 1])
        out.append((ends[k], sign * (cur / ref - 1) if ref != 0 else np.nan))
    return out


def first_alarm(series, delta):
    for end, d in series:
        if np.isfinite(d) and d < -delta:
            return end
    return None


def evaluate(win, stat, M, B, delta, sign, null_arms, event_of):
    rows = []
    for (arm, seed), g in win.groupby(["arm", "seed"]):
        t = first_alarm(drop_series(g, stat, M, B, sign), delta)
        if arm in null_arms:
            rows.append({"arm": arm, "seed": seed, "event": False, "alarm": t,
                         "false_alarm": t is not None, "hit": False, "delay": np.nan})
        else:
            fa = t is not None and t <= event_of
            hit = t is not None and event_of < t <= event_of + HORIZON
            rows.append({"arm": arm, "seed": seed, "event": True, "alarm": t, "false_alarm": fa,
                         "hit": hit, "delay": (t - event_of) if hit else np.nan})
    return pd.DataFrame(rows)


def calibrate(cal, stat):
    best = None
    for sign in (+1, -1) if stat != "MG" else (+1,):
        for M, B in itertools.product((2, 3, 4), (3, 4, 6)):
            # smallest delta with no alarm on any null run
            worst = []
            for (arm, seed), g in cal[cal.arm.isin(CAL_NULL)].groupby(["arm", "seed"]):
                ds = [d for _, d in drop_series(g, stat, M, B, sign) if np.isfinite(d)]
                worst.append(-min(ds) if ds else 0.0)
            delta = max(0.0, max(worst)) + 0.02
            ev = evaluate(cal, stat, M, B, delta, sign, CAL_NULL, EVENT)
            e = ev[ev.event]
            key = (e.hit.mean(), -np.nanmedian(e.delay) if e.hit.any() else -1e9)
            if best is None or key > best[0]:
                best = (key, {"stat": stat, "M": M, "B": B, "delta": delta, "sign": sign,
                              "cal_hit": float(e.hit.mean()),
                              "cal_false_alarm_event_runs": float(e.false_alarm.mean()),
                              "cal_median_delay": float(np.nanmedian(e.delay)) if e.hit.any() else None})
    return best[1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal_path, test_path = OUT / "cal_windows.csv", OUT / "test_windows.csv"
    cal = pd.read_csv(cal_path) if cal_path.exists() else calibration_windows()
    cal.to_csv(cal_path, index=False)
    test = pd.read_csv(test_path) if test_path.exists() else test_windows()
    test.to_csv(test_path, index=False)

    chosen = {s: calibrate(cal, s) for s in STATS}
    json.dump(chosen, open(OUT / "chosen_rules.json", "w"), indent=1, default=float)

    per_run, summary = [], []
    for s, rule in chosen.items():
        ev = evaluate(test, s, rule["M"], rule["B"], rule["delta"], rule["sign"], TEST_NULL, EVENT)
        ev["stat"] = s
        per_run.append(ev)
        for arm, g in ev.groupby("arm"):
            summary.append({"stat": s, "arm": arm, "n": len(g), "hit_rate": g.hit.mean(),
                            "false_alarm_rate": g.false_alarm.mean(),
                            "median_delay": np.nanmedian(g.delay) if g.hit.any() else np.nan})
    per_run = pd.concat(per_run)
    per_run.to_csv(OUT / "test_per_run.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "test_summary.csv", index=False)

    strong = ["lr10", "lr100", "freeze_head", "freeze_bias", "prune50", "prune80", "prune95"]
    weak = ["lr3", "freeze12"]
    overall = []
    for s in STATS:
        pr = per_run[per_run.stat == s]
        overall.append({
            "stat": s, **{k: chosen[s][k] for k in ("M", "B", "delta")},
            "hit, strong events": pr[pr.arm.isin(strong)].hit.mean(),
            "hit, weak events": pr[pr.arm.isin(weak)].hit.mean(),
            "false alarm, no-event runs": pr[pr.arm.isin(TEST_NULL)].false_alarm.mean(),
            "alarm before event, event runs": pr[pr.event].false_alarm.mean(),
            "median delay, steps": np.nanmedian(pr[pr.arm.isin(strong)].delay),
        })
    ov = pd.DataFrame(overall)
    ov.to_csv(OUT / "test_overall.csv", index=False)
    pd.set_option("display.width", 220)
    print(json.dumps(chosen, indent=1, default=float))
    print(ov.round(3).to_string(index=False))
    print(summ[summ.stat == "MG"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
