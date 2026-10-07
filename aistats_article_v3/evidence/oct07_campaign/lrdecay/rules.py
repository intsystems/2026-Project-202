"""Decay rules. Each rule reads only the constant-LR trunk record up to the alarm step (causal)
and returns the alarm step or None (never decay). outcome() maps an alarm to the measured
result of decaying x10 at the first checkpoint >= alarm."""
from __future__ import annotations

import itertools

import numpy as np

EVAL = 100
TMIN_FRACS = (2 / 16, 4 / 16, 6 / 16, 8 / 16)
QS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def outcome(res, alarm):
    if alarm is None:
        return res["none"]
    for tj in res["grid"]:
        if tj >= alarm:
            return res["branches"][str(tj)]
    return res["none"]


def decay_step(res, alarm):
    if alarm is None:
        return None
    for tj in res["grid"]:
        if tj >= alarm:
            return tj
    return None


# ---------------------------------------------------------------- fixed schedule
def fixed_grid():
    return [{"f": j / 16} for j in range(2, 16)] + [{"f": None}]


def fixed(run, p):
    # floor, consistent with the checkpoint grid j*T//16 (round() skipped t_15 when T=5000 -- bug fixed
    # after the first S3 test analysis; it only affected fixed_step on condition J)
    return None if p["f"] is None else int(p["f"] * run["T"])


# ---------------------------------------------------------------- scalar window statistics
def stat_grid(quantiles):
    """quantiles: {(log, stat): {q: value}} from the calibration windows only."""
    out = []
    for tm in TMIN_FRACS:
        for sign in (1, -1):
            for q in QS:
                out.append({"type": "level", "tmin": tm, "sign": sign, "q": q})
            for M, d in itertools.product((2, 4), (0.02, 0.05, 0.1, 0.2, 0.4)):
                out.append({"type": "rel", "tmin": tm, "sign": sign, "M": M, "B": 4, "delta": d})
        for M, e in itertools.product((2, 4), (0.01, 0.02, 0.05, 0.1)):
            out.append({"type": "stable", "tmin": tm, "M": M, "B": 4, "eps": e})
    return out


def _rel(v, k, M, B):
    cur = np.nanmedian(v[k - M + 1:k + 1])
    ref = np.nanmedian(v[k - M - B + 1:k - M + 1])
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0:
        return np.nan
    return (cur - ref) / abs(ref)


def stat_rule(run, log, stat, p, quantiles):
    w = run["win"][log]
    v, ends = w[stat], w["end"]
    tmin = p["tmin"] * run["T"]
    if p["type"] == "level":
        th = quantiles[(log, stat)][p["q"]]
        for k in range(len(v)):
            if ends[k] >= tmin and np.isfinite(v[k]) and p["sign"] * (v[k] - th) > 0:
                return int(ends[k])
        return None
    M, B = p["M"], p["B"]
    for k in range(M + B - 1, len(v)):
        if ends[k] < tmin:
            continue
        d = _rel(v, k, M, B)
        if not np.isfinite(d):
            continue
        if p["type"] == "rel" and p["sign"] * d > p["delta"]:
            return int(ends[k])
        if p["type"] == "stable" and abs(d) < p["eps"]:
            return int(ends[k])
    return None


# ---------------------------------------------------------------- Pflug / Chee & Toulis 2018
def pflug_grid():
    return [{"tmin": tm, "start": s} for tm in TMIN_FRACS for s in ("zero", "tmin")]


def pflug(run, p):
    gg = np.nan_to_num(run["logs"]["gg"], nan=0.0)
    T = run["T"]
    tmin = int(p["tmin"] * T)
    s0 = 0 if p["start"] == "zero" else tmin
    S = np.cumsum(gg[s0:])
    hit = np.flatnonzero((np.arange(s0, T) >= tmin) & (np.arange(s0, T) > s0 + 10) & (S < 0))
    return int(s0 + hit[0] + 1) if len(hit) else None


# ---------------------------------------------------------------- SASA (Lang et al. 2019) / SASA+
def sasa_grid():
    out = []
    for tm in TMIN_FRACS:
        for z in (1.28, 1.96):
            out.append({"tmin": tm, "mode": "ci0", "z": z})
        for d in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75):
            out.append({"tmin": tm, "mode": "equiv", "z": 1.96, "delta": d})
    return out


def sasa_terms(run):
    lg = run["logs"]
    diss = 0.5 * run["lr"] * (1 + 0.9) * lg["dd"]
    return lg["xg"] - diss, diss


def sasa(run, p):
    delta_t, diss = run["sasa"]
    T = run["T"]
    t = int(p["tmin"] * T)
    t = max(EVAL, (t + EVAL - 1) // EVAL * EVAL)
    while t <= T:
        a = t // 2
        x = delta_t[a:t]
        N = len(x)
        nb = int(np.sqrt(N))
        if nb >= 3:
            bm = x[:nb * (N // nb)].reshape(nb, -1).mean(1)
            mu, se = x.mean(), bm.std(ddof=1) / np.sqrt(nb)
            if p["mode"] == "ci0" and abs(mu) <= p["z"] * se:
                return t
            if p["mode"] == "equiv" and abs(mu) + p["z"] * se < p["delta"] * diss[a:t].mean():
                return t
        t += EVAL
    return None


# ---------------------------------------------------------------- Pesme et al. 2020 distance diagnostic
def pesme_grid():
    return [{"tmin": tm, "q": q, "th": th} for tm in TMIN_FRACS for q in (1.5, 2.0)
            for th in (0.25, 0.5, 0.75, 1.0, 1.25)]


def pesme(run, p):
    om = np.log(np.maximum(run["logs"]["dist0"], 1e-12) ** 2)
    T = run["T"]
    t = max(EVAL, int(p["tmin"] * T) // EVAL * EVAL)
    while t < T:
        a = int(t / p["q"])
        slope = (om[t - 1] - om[a - 1]) / np.log(t / a)
        if slope < p["th"]:
            return t
        t += EVAL
    return None


# ---------------------------------------------------------------- ReduceLROnPlateau (torch semantics)
def plateau_grid():
    return [{"tmin": tm, "patience": pa, "thr": th} for tm in TMIN_FRACS for pa in (3, 5, 10, 20)
            for th in (1e-3, 1e-2, 3e-2, 1e-1)]


def _plateau(steps, metric, T, p):
    best, bad = np.inf, 0
    tmin = p["tmin"] * T
    for s, m in zip(steps, metric):
        if m < best * (1 - p["thr"]):
            best, bad = m, 0
        else:
            bad += 1
        if bad > p["patience"] and s >= tmin:
            return int(s)
    return None


def plateau_train(run, p):
    bl = run["logs"]["batch_loss"]
    T = run["T"]
    steps = np.arange(EVAL, T + 1, EVAL)
    metric = [np.mean(bl[s - EVAL:s]) for s in steps]
    return _plateau(steps, metric, T, p)


def plateau_val(run, p):
    ev = [e for e in run["ev"] if e["step"] > 0]
    return _plateau([e["step"] for e in ev], [e["val_loss"] for e in ev], run["T"], p)
