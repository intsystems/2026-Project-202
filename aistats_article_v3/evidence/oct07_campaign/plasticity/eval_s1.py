"""Decision protocol of run_s1.py (see its docstring): calibrate every reset rule on the
calibration pools, freeze, evaluate on the test pools by exact renewal.

usage: python eval_s1.py [F1|F2 ...]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RES = HERE / "results"
T = 40
META = ("j", "split", "family", "cond", "seed")
CAL_CONDS = ("A3W256", "A3W64", "S1W256", "S3W256")
UNSEEN = ("A1W128", "S2W128")
DELTAS = (.005, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5, 1.0)
QS = np.arange(0.05, 0.951, 0.05)
DOMAIN = ("dormant_0.0", "dormant_0.025", "dormant_0.1", "srank", "erank")
LEVEL = ("acc_mean", "loss_mean", "gnorm_mean", "pnorm_end")


def load(fam):
    df = pd.concat([pd.read_csv(f) for f in sorted(RUNS.glob(f"mon_{fam}_*.csv"))], ignore_index=True)
    return df


def stat_of(col):
    if col in DOMAIN:
        return col
    if col in LEVEL:
        return col
    return col.split("|")[0]


def runs_of(df, split, conds):
    """{cond: [ (acc[T], {monitor: v[T]}) per seed sorted ]}"""
    out = {}
    mons = [c for c in df.columns if c not in META]
    for c in conds:
        g = df[(df.split == split) & (df.cond == c)]
        pool = []
        for s, h in sorted(g.groupby("seed"), key=lambda kv: kv[0]):
            h = h.sort_values("j")
            assert len(h) == T, (c, s, len(h))
            pool.append((h["acc_mean"].to_numpy(float), {m: h[m].to_numpy(float) for m in mons}, s))
        out[c] = pool
    return out, mons


def first_index(mask):
    w = np.flatnonzero(mask)
    return int(w[0]) if len(w) else T  # T == never


def rolling_median(v, M):
    out = np.full(len(v), np.nan)
    for j in range(M - 1, len(v)):
        w = v[j - M + 1:j + 1]
        if np.all(np.isfinite(w)):
            out[j] = np.median(w)
    return out


def thresholds(cal_values):
    fin = cal_values[np.isfinite(cal_values)]
    return [float(t) for t in np.unique(np.quantile(fin, QS))] if len(fin) else []


def rule_grid(cal_values):
    """All rules grouped as {(kind, s, B, M): [thresholds]}; ABS thresholds = calibration
    quantiles of the monitor, REL thresholds = DELTAS."""
    g = {}
    for s, B, M in itertools.product((1, -1), (2, 4), (1, 2, 4)):
        g[("REL", s, B, M)] = list(DELTAS)
    ths = thresholds(cal_values)
    for s, M in itertools.product((1, -1), (1, 2, 4)):
        g[("ABS", s, 0, M)] = ths
    return g


def alarm_indices(v, head, ps):
    """First local task index j after which the rule (head, p) resets, for every p in ps
    (T = never). head = (kind, s, B, M)."""
    kind, s, B, M = head
    ps = np.asarray(ps, float)
    never = np.full(len(ps), T, int)
    ok = np.flatnonzero(np.isfinite(v))
    if not len(ok) or not len(ps):
        return never
    j0 = ok[0]
    cur = rolling_median(v, M)
    idx = np.arange(len(v))
    if kind == "REL":
        if j0 + B > len(v):
            return never
        ref = np.median(v[j0:j0 + B])
        if not np.isfinite(ref) or ref == 0:
            return never
        D = s * (cur - ref) / abs(ref)
        valid = (idx - M + 1 >= j0 + B) & np.isfinite(D)
        cm = np.minimum.accumulate(np.where(valid, D, np.inf))      # alarm when D < -p
        return np.searchsorted(-cm, ps, side="right").clip(max=T)
    valid = (idx - M + 1 >= j0) & np.isfinite(cur)
    if s == 1:                                                       # alarm when cur < th
        cm = np.minimum.accumulate(np.where(valid, cur, np.inf))
        return np.searchsorted(-cm, -ps, side="right").clip(max=T)
    cM = np.maximum.accumulate(np.where(valid, cur, -np.inf))       # alarm when cur > th
    return np.searchsorted(cM, ps, side="right").clip(max=T)


def alarm_index(v, rule):
    kind, s, B, M, p = rule
    return int(alarm_indices(v, (kind, s, B, M), [p])[0])


def stream_utils(pool, alarms):
    """Renewal evaluation: stream i starts with run i, after a reset continues with the next
    run of the pool from its task 0. alarms[r] = first alarm index of run r (T = never)."""
    R = len(pool)
    out = []
    for i in range(R):
        pos, r, tot, nres = 0, i, 0.0, 0
        while pos < T:
            f = alarms[r]
            seg = min(f + 1, T - pos) if f < T else T - pos
            tot += pool[r][0][:seg].sum()
            pos += seg
            if pos < T:
                nres += 1
            r = (r + 1) % R
        out.append((tot / T, nres))
    return out


def evaluate(pools, alarm_fn):
    """alarm_fn(cond, run_index, run) -> alarm index. Returns per-stream DataFrame."""
    rows = []
    for c, pool in pools.items():
        al = [alarm_fn(c, k, run) for k, run in enumerate(pool)]
        for i, (u, n) in enumerate(stream_utils(pool, al)):
            rows.append({"cond": c, "stream": i, "util": u, "resets": n})
    return pd.DataFrame(rows)


def score(ev):
    by = ev.groupby("cond")[["util", "resets"]].mean()
    return float(by.util.mean()), float(by.resets.mean())


def calibrate_monitor(cal_pools, m):
    vals = np.concatenate([run[1][m] for pool in cal_pools.values() for run in pool])
    best = None
    for head, ps in rule_grid(vals).items():
        al = {c: np.stack([alarm_indices(run[1][m], head, ps) for run in pool], 1)
              for c, pool in cal_pools.items()}                     # [n_p, n_runs]
        for i, p in enumerate(ps):
            us, ns = [], []
            for c, pool in cal_pools.items():
                su = stream_utils(pool, list(al[c][i]))
                us.append(np.mean([u for u, _ in su]))
                ns.append(np.mean([n for _, n in su]))
            u, n = float(np.mean(us)), float(np.mean(ns))
            key = (round(u, 9), -n)
            if best is None or key > best[0]:
                best = (key, head + (p,), u, n)
    return {"monitor": m, "rule": best[1], "cal_util": best[2], "cal_resets": best[3]}


def calibrate_fixed(cal_pools):
    best = None
    for I in range(1, T + 1):
        ev = evaluate(cal_pools, lambda c, k, run: I - 1 if I < T else T)
        u, n = score(ev)
        key = (round(u, 9), -n)
        if best is None or key > best[0]:
            best = (key, I, u, n)
    return best[1], best[2], best[3]


def boot_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(d, (n, len(d))).mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def auc(score_, y):
    from scipy.stats import rankdata
    ok = np.isfinite(score_)
    s, y = score_[ok], y[ok]
    P, N = y.sum(), (~y).sum()
    if P == 0 or N == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - P * (P + 1) / 2) / (P * N))


def detection_auc(pools, mons):
    """y_j: mean acc of tasks j+1..j+3 below mean fresh acc (tasks 0..2) of the condition."""
    S, Y = {m: [] for m in mons}, []
    for c, pool in pools.items():
        fresh = np.mean([run[0][:3].mean() for run in pool])
        for acc, vals, _ in pool:
            for j in range(T - 3):
                Y.append(acc[j + 1:j + 4].mean() < fresh)
                for m in mons:
                    S[m].append(vals[m][j])
    Y = np.array(Y)
    return {m: auc(np.array(S[m]), Y) for m in mons}, float(Y.mean())


def early_prediction(df, mons, fam):
    """Secondary (b): early score (mean v_j, j=2..4) vs plasticity loss of the run."""
    from scipy.stats import spearmanr
    res = {}
    for split in ("cal", "test"):
        g = df[df.split == split]
        rows = []
        for (c, sd), h in g.groupby(["cond", "seed"]):
            h = h.sort_values("j")
            acc = h.acc_mean.to_numpy()
            rows.append({"cond": c, "seed": sd, "loss_of_plasticity": acc[1:6].mean() - acc[35:40].mean(),
                         **{m: np.nanmean(h[m].to_numpy()[2:5]) for m in mons}})
        res[split] = pd.DataFrame(rows)
    out = []
    for m in mons:
        rc = spearmanr(res["cal"][m], res["cal"].loss_of_plasticity, nan_policy="omit")[0]
        rt = spearmanr(res["test"][m], res["test"].loss_of_plasticity, nan_policy="omit")[0]
        out.append({"monitor": m, "stat": stat_of(m), "rho_cal": rc, "rho_test": rt,
                    "rho_test_signed": rt * np.sign(rc) if np.isfinite(rc) else np.nan})
    out = pd.DataFrame(out)
    out.to_csv(RES / f"{fam}_early_prediction.csv", index=False)
    res["test"].to_csv(RES / f"{fam}_early_runs_test.csv", index=False)
    best = out.assign(a=out.rho_cal.abs()).sort_values("a", ascending=False).groupby("stat").head(1)
    print(f"\n== {fam} early prediction of plasticity loss (Spearman; best log/window per stat on cal) ==")
    print(best.sort_values("rho_test_signed", ascending=False)[["stat", "monitor", "rho_cal", "rho_test_signed"]]
          .round(3).to_string(index=False))


def main(fam):
    RES.mkdir(exist_ok=True)
    df = load(fam)
    cal_pools, mons = runs_of(df, "cal", CAL_CONDS)
    test_pools, _ = runs_of(df, "test", CAL_CONDS + UNSEEN)
    print(fam, "monitors:", len(mons), "cal runs:", sum(map(len, cal_pools.values())),
          "test runs:", sum(map(len, test_pools.values())), flush=True)

    # ---- calibration (every monitor, same grid) ------------------------------------------
    cal_path = RES / f"{fam}_calibration.json"
    if cal_path.exists():
        chosen = json.load(open(cal_path))
    else:
        chosen = {"monitors": [calibrate_monitor(cal_pools, m) for m in mons]}
        I, u, n = calibrate_fixed(cal_pools)
        chosen["fixed"] = {"I": I, "cal_util": u, "cal_resets": n}
        json.dump(chosen, open(cal_path, "w"), indent=1)
    cal = pd.DataFrame(chosen["monitors"])
    cal["stat"] = cal.monitor.map(stat_of)
    cal["window"] = cal.monitor.map(lambda m: m.split("|")[2] if m.count("|") == 2 else "level")
    cal["log"] = cal.monitor.map(lambda m: m.split("|")[1] if m.count("|") == 2 else "-")

    # ---- test: every monitor's frozen rule ------------------------------------------------
    per_stream = {}
    rows = []
    for _, r in cal.iterrows():
        rule = tuple(r.rule)
        ev = evaluate(test_pools, lambda c, k, run: alarm_index(run[1][r.monitor], rule))
        per_stream[r.monitor] = ev
        rows.append(r.to_dict() | {"test_util": score(ev)[0], "test_resets": score(ev)[1]})
    I = chosen["fixed"]["I"]
    refs = {"never": lambda c, k, run: T, "every_task": lambda c, k, run: 0,
            f"fixed_I{I}": lambda c, k, run: (I - 1) if I < T else T}
    for name, fn in refs.items():
        ev = evaluate(test_pools, fn)
        per_stream[name] = ev
        cu = score(evaluate(cal_pools, fn))
        rows.append({"monitor": name, "stat": name, "window": "-", "log": "-", "rule": None,
                     "cal_util": cu[0], "cal_resets": cu[1],
                     "test_util": score(ev)[0], "test_resets": score(ev)[1]})
    # oracle: best fixed interval per condition, chosen on test (upper reference, not a policy)
    orc = []
    for c, pool in test_pools.items():
        best = max(((np.mean([u for u, _ in stream_utils(pool, [I_ - 1 if I_ < T else T] * len(pool))]), I_)
                    for I_ in range(1, T + 1)))
        ev = evaluate({c: pool}, lambda c_, k, run: (best[1] - 1) if best[1] < T else T)
        ev["I"] = best[1]
        orc.append(ev)
    per_stream["oracle_fixed_per_cond"] = pd.concat(orc)
    rows.append({"monitor": "oracle_fixed_per_cond", "stat": "oracle_fixed_per_cond", "window": "-",
                 "log": "-", "rule": None, "cal_util": np.nan, "cal_resets": np.nan,
                 "test_util": score(per_stream["oracle_fixed_per_cond"])[0],
                 "test_resets": score(per_stream["oracle_fixed_per_cond"])[1]})
    allm = pd.DataFrame(rows)
    allm.to_csv(RES / f"{fam}_all_monitors.csv", index=False)

    # ---- per statistic: best (log, window) on calibration -> test ------------------------
    tables = {}
    for scope, filt in (("any", lambda d: d), ("span", lambda d: d[d.window.isin(["span", "-", "level"])]),
                        ("within", lambda d: d[d.window.isin(["within", "-", "level"])])):
        sub = filt(allm[allm.stat != "oracle_fixed_per_cond"])
        best = (sub.sort_values(["cal_util", "cal_resets"], ascending=[False, True])
                .drop_duplicates("stat").reset_index(drop=True))
        tables[scope] = best
    out = []
    for scope, best in tables.items():
        mg_m = best.set_index("stat").loc["MG", "monitor"]
        mg_ev = per_stream[mg_m].set_index(["cond", "stream"]).util
        for _, r in best.iterrows():
            ev = per_stream[r.monitor].set_index(["cond", "stream"]).util
            d = (mg_ev - ev.reindex(mg_ev.index)).to_numpy()
            seen = mg_ev.index.get_level_values(0).isin(CAL_CONDS)
            pc = (mg_ev - ev.reindex(mg_ev.index)).groupby(level=0).mean()
            ts = per_stream[r.monitor]
            out.append({"scope": scope, "stat": r.stat, "monitor": r.monitor, "rule": r.rule,
                        "cal_util": r.cal_util, "test_util": r.test_util,
                        "test_seen": ts[ts.cond.isin(CAL_CONDS)].groupby("cond").util.mean().mean(),
                        "test_unseen": ts[ts.cond.isin(UNSEEN)].groupby("cond").util.mean().mean(),
                        "test_resets": r.test_resets,
                        "MG_minus": float(np.mean(d)), "ci_lo": boot_ci(d)[0], "ci_hi": boot_ci(d)[1],
                        "MG_minus_seen": float(np.mean(d[seen])), "MG_minus_unseen": float(np.mean(d[~seen])),
                        "conds_MG_ge": int((pc >= -1e-12).sum())})
        out.append({"scope": scope, "stat": "oracle_fixed_per_cond", "monitor": "oracle",
                    "test_util": score(per_stream["oracle_fixed_per_cond"])[0]})
    tab = pd.DataFrame(out)
    tab.to_csv(RES / f"{fam}_by_stat.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    for scope in ("any", "span", "within"):
        t = tab[tab.scope == scope].sort_values("test_util", ascending=False)
        print(f"\n== {fam} scope={scope} (best log/window per statistic chosen on calibration) ==")
        print(t[["stat", "monitor", "cal_util", "test_util", "test_seen", "test_unseen", "test_resets",
                 "MG_minus", "ci_lo", "ci_hi", "conds_MG_ge"]].round(4).to_string(index=False))

    # per-condition table for the main entries
    keep = ["MG", "acc_mean", "loss_mean", "pnorm_end", "dormant_0.0", "dormant_0.1", "srank", "erank",
            "self_repeat", "spectral_entropy", "roughness", "recurrence_rate", "never", "every_task",
            f"fixed_I{I}"]
    pcs = []
    b = tables["any"].set_index("stat")
    for s in keep:
        if s in b.index:
            ev = per_stream[b.loc[s, "monitor"]]
            pcs.append(ev.groupby("cond").agg(util=("util", "mean"), resets=("resets", "mean")).assign(stat=s))
    oc = per_stream["oracle_fixed_per_cond"]
    pcs.append(oc.groupby("cond").agg(util=("util", "mean"), resets=("resets", "mean")).assign(stat="oracle"))
    pc = pd.concat(pcs).reset_index().pivot(index="stat", columns="cond", values="util")
    pc.to_csv(RES / f"{fam}_per_cond.csv")
    print(f"\n== {fam} per-condition test utility ==")
    print(pc.round(4).to_string())

    # ---- secondary: detection AUC (sign chosen on calibration) ---------------------------
    auc_cal, base_cal = detection_auc(cal_pools, mons)
    auc_test, base_test = detection_auc(test_pools, mons)
    det = pd.DataFrame({"monitor": mons, "auc_cal": [auc_cal[m] for m in mons],
                        "auc_test_raw": [auc_test[m] for m in mons]})
    det["sign"] = np.where(det.auc_cal >= 0.5, 1, -1)
    det["auc_test"] = np.where(det.sign == 1, det.auc_test_raw, 1 - det.auc_test_raw)
    det["auc_cal_signed"] = np.maximum(det.auc_cal, 1 - det.auc_cal)
    det["stat"] = det.monitor.map(stat_of)
    det.to_csv(RES / f"{fam}_detection_auc.csv", index=False)
    bd = det.sort_values("auc_cal", key=lambda s: (s - 0.5).abs(), ascending=False).groupby("stat").head(1)
    print(f"\n== {fam} detection AUC (positives cal {base_cal:.2f}, test {base_test:.2f}); "
          f"best monitor per statistic chosen on calibration ==")
    early_prediction(df, mons, fam)
    print(bd.sort_values("auc_test", ascending=False)[["stat", "monitor", "sign", "auc_cal_signed", "auc_test"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    for fam in sys.argv[1:] or ["F1", "F2"]:
        main(fam)
