"""Window statistics and alarm rules for setting T (see tr_main.py for the protocol).

All scalar-log statistics are computed on the SAME windows (W=1000, stride 500) of the logged
parameter-norm series `log_pn` (norm, or norm^2 in T6). recurrence_rate and corr_dim are the
baselines.py definitions re-implemented with scipy pdist (same pairs, same order, same
numbers; checked in `_selftest`) to keep peak RAM per worker low.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
import sys
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
MW = HERE.parent
ROOT = MW.parent
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(MW))
sys.path.insert(0, str(ROOT / "research_trajectory_reference"))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
import baselines as BL  # noqa: E402
from cifar_events import simple  # noqa: E402

W, S, WARM, HORIZON = 1000, 500, 1500, 5000
CFG_T4 = EstimatorConfig(max_E=20, tau=4, k_neighbors=50, theiler="embedding")
CFG_T1 = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")


def _theiler_pairs_pdist(Y, idx, w):
    d = pdist(Y)
    i, j = np.triu_indices(len(Y), 1)
    keep = np.abs(idx[i] - idx[j]) > w
    return d[keep]


def recurrence_rate(x, E=20, tau=1, frac=0.1):
    Y, tau = BL.delay(x, E, tau)
    Ys, idx = BL._sub(Y, 2000)
    d = _theiler_pairs_pdist(Ys, idx, (E - 1) * tau)
    return float(np.mean(d < frac * np.median(d)))


def corr_dim(x, E=20, tau=1, q=(0.01, 0.1)):
    Y, tau = BL.delay(x, E, tau)
    Ys, idx = BL._sub(Y, 2500)
    d = _theiler_pairs_pdist(Ys, idx, (E - 1) * tau)
    r = np.quantile(d, np.linspace(q[0], q[1], 8))
    C = np.array([np.mean(d < ri) for ri in r])
    return float(np.polyfit(np.log(r), np.log(C), 1)[0])


SCALAR = {
    "MG": lambda s: estimate(s, CFG_T4).MG,                 # primary MG (E20 tau4 k50)
    "MG_t1k20": lambda s: estimate(s, CFG_T1).MG,           # training-log default
    "self_repeat": BL.self_repeat,
    "spectral_entropy": BL.spectral_entropy,
    "roughness": BL.roughness,
    "perm_entropy": partial(BL.perm_entropy, lag=1),
    "recurrence_rate": recurrence_rate,
    "corr_dim": corr_dim,
    "twonn": partial(BL.twonn_fit, tau=1),
    "linear_pr": partial(BL.linear_pr, tau=1),
}
SIMPLE = ("crossings", "lag1", "det_std")
INTERNAL = ("erank", "srank", "dormant")


def window_rows(res):
    x = np.asarray(res["log_pn"], float)
    gn = np.asarray(res["log_gn"], float)
    bl = np.asarray(res["batch_loss"], float)
    tt = np.asarray(res["truth_t"], float)
    rows = []
    for a in range(0, len(x) - W + 1, S):
        seg = x[a:a + W]
        r = {"start": a, "end": a + W}
        for k, f in SCALAR.items():
            try:
                r[k] = float(f(seg))
            except Exception:
                r[k] = np.nan
        r.update(simple(seg))
        r["gn_med"] = float(np.median(gn[a:a + W]))
        r["loss_med"] = float(np.median(bl[a:a + W]))
        sel = (tt > a) & (tt <= a + W)
        for k in INTERNAL:
            r[k] = float(np.median(np.asarray(res[f"truth_{k}"])[sel]))
        rows.append(r)
    return rows


# ---- rules ------------------------------------------------------------------------------------
# A run is a dict: {"win": DataFrame of its windows, "gn": array, "loss": array,
#                   "t_e": int|None, "event": str|None}

def ratio_series(win, stat, M, B, sign):
    g = win.sort_values("start")
    v, ends = g[stat].to_numpy(float), g.end.to_numpy()
    ok = ends > WARM
    v, ends = v[ok], ends[ok]
    out = []
    for k in range(M + B - 1, len(v)):
        cur = np.nanmedian(v[k - M + 1:k + 1])
        ref = np.nanmedian(v[k - M - B + 1:k - M + 1])
        d = sign * (cur / ref - 1) if (ref != 0 and np.isfinite(ref) and np.isfinite(cur)) else np.nan
        out.append((int(ends[k]), d))
    return out


def level_series(win, stat, sign):
    """sign=+1: alarm on low values (value below L); -1: on high values. Returned as (end, sign*v)."""
    g = win.sort_values("start")
    ends, v = g.end.to_numpy(), g[stat].to_numpy(float)
    return [(int(e), sign * vv) for e, vv in zip(ends, v) if e > WARM]


def z_series(gn, block, nref, sign):
    nb = len(gn) // block
    b = gn[:nb * block].reshape(nb, block).mean(1)
    out = []
    for j in range(nref, nb):
        end = (j + 1) * block
        if end <= WARM:
            continue
        ref = b[j - nref:j]
        sd = ref.std()
        z = (b[j] - ref.mean()) / sd if sd > 0 else 0.0
        out.append((end, sign * z))
    return out


def plateau_alarm(loss, theta, patience, alpha=0.01):
    """ReduceLROnPlateau semantics (rel mode) on the EMA of the logged mini-batch loss."""
    e = loss[0]
    best, bad = np.inf, 0
    for t in range(len(loss)):
        e = (1 - alpha) * e + alpha * loss[t]
        if t < WARM:
            continue
        if e < best * (1 - theta):
            best, bad = e, 0
        else:
            bad += 1
            if bad > patience:
                return t + 1
    return None


def first_below(series, thr):
    for end, d in series:
        if np.isfinite(d) and d < thr:
            return end
    return None


def score(alarm, t_e):
    if t_e is None:
        return {"false_alarm": alarm is not None, "hit": False, "delay": np.nan}
    fa = alarm is not None and alarm <= t_e
    hit = alarm is not None and t_e < alarm <= t_e + HORIZON
    return {"false_alarm": fa, "hit": hit, "delay": (alarm - t_e) if hit else np.nan}


# Each monitor: kind, stat, grid. The 'series' maker returns (end, d) pairs; alarm when d < thr.
def series_for(run, mon, p):
    kind, stat = mon
    if kind == "ratio":
        return ratio_series(run["win"], stat, p["M"], p["B"], p["sign"])
    if kind == "level":
        return level_series(run["win"], stat, p["sign"])
    if kind == "z":
        return z_series(run["gn"], p["block"], p["nref"], p["sign"])
    raise ValueError(kind)


def grid_for(kind):
    if kind == "ratio":
        return [dict(M=M, B=B, sign=s) for s in (1, -1) for M, B in itertools.product((2, 3, 4), (3, 4, 6))]
    if kind == "level":
        return [dict(sign=s) for s in (1, -1)]
    if kind == "z":
        return [dict(block=b, nref=n, sign=s) for s in (1, -1) for b in (20, 50, 100) for n in (20, 40)]
    raise ValueError(kind)


def threshold_from_nulls(kind, worst):
    """worst = min over null runs of the d-series (alarm iff d < thr)."""
    if kind == "ratio":
        return min(worst, 0.0) - 0.02               # E6 margin: delta = max(0, max drop) + 0.02
    if kind == "level":
        return worst - 0.02 * abs(worst)            # 2 % beyond the most extreme null level
    if kind == "z":
        return min(worst, 0.0) * 1.05               # 5 % beyond the largest null excursion
    raise ValueError(kind)


def evaluate(runs, mon, p, thr):
    out = []
    for r in runs:
        if mon[0] == "plateau":
            a = plateau_alarm(r["loss"], p["theta"], p["patience"])
        else:
            a = first_below(series_for(r, mon, p), thr)
        out.append({"alarm": a, **score(a, r["t_e"])})
    return out


def calibrate(cal_runs, mon):
    nulls = [r for r in cal_runs if r["t_e"] is None]
    evs = [r for r in cal_runs if r["t_e"] is not None]
    best = None
    if mon[0] == "plateau":
        for th, pa in itertools.product((1e-4, 1e-3, 1e-2, 0.05, 0.1), (250, 500, 1000, 2000, 3000)):
            p = dict(theta=th, patience=pa)
            nfa = sum(e["false_alarm"] for e in evaluate(nulls, mon, p, None))
            ev = evaluate(evs, mon, p, None)
            hits = sum(e["hit"] for e in ev)
            dl = np.nanmedian([e["delay"] for e in ev]) if hits else 1e9
            key = (-nfa, hits, -dl)
            if best is None or key > best[0]:
                best = (key, {**p, "thr": None, "cal_hits": hits, "cal_null_fa": nfa, "cal_delay": dl})
        return best[1]
    for p in grid_for(mon[0]):
        worst = []
        for r in nulls:
            ds = [d for _, d in series_for(r, mon, p) if np.isfinite(d)]
            if ds:
                worst.append(min(ds))
        if not worst:
            continue
        thr = threshold_from_nulls(mon[0], min(worst))
        ev = evaluate(evs, mon, p, thr)
        hits = sum(e["hit"] for e in ev)
        dl = float(np.nanmedian([e["delay"] for e in ev])) if hits else 1e9
        key = (hits, -dl)
        if best is None or key > best[0]:
            best = (key, {**p, "thr": float(thr), "cal_hits": hits, "cal_null_fa": 0, "cal_delay": dl})
    return best[1]


MONITORS = {}
for k in list(SCALAR) + list(SIMPLE):
    MONITORS[k] = ("ratio", k)
MONITORS.update({
    "gradnorm_ratio": ("ratio", "gn_med"),          # relative change of median grad norm
    "gradnorm_abs": ("level", "gn_med"),            # absolute grad-norm threshold
    "gradnorm_z": ("z", "gn"),                      # grad-norm z-score vs trailing blocks
    "loss_plateau": ("plateau", "loss"),            # ReduceLROnPlateau-style rule
    "srank_abs": ("level", "srank"),                # feature srank_0.01 absolute threshold
    "srank_ratio": ("ratio", "srank"),
    "erank_ratio": ("ratio", "erank"),
    "dormant_abs": ("level", "dormant"),            # dormant fraction absolute threshold
})


def _selftest():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(1000))
    a = (recurrence_rate(x), BL.recurrence_rate(x, tau=1))
    b = (corr_dim(x), BL.corr_dim(x, tau=1))
    print("recurrence", a, "corr_dim", b)
    assert abs(a[0] - a[1]) < 1e-12 and abs(b[0] - b[1]) < 1e-9


if __name__ == "__main__":
    _selftest()
