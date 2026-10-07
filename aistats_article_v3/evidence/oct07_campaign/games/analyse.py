"""Calibrate every rule on CAL, score on all test arms (protocol in main.py docstring)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FAMILIES = {"peak_count": [0.01, 0.02, 0.05, 0.1], "harmonic_count": [0.01, 0.02, 0.05, 0.1],
            "state_count": [0.01, 0.03, 0.1], "xplay_count": [0.03, 0.1, 0.3]}
SCALAR = ["MG", "MG_tau1", "twonn", "recurrence_rate", "corr_dim", "spectral_entropy", "peak_count",
          "harmonic_count", "self_repeat", "self_repeat_long", "roughness", "perm_entropy",
          "linear_pr", "crossings", "lag1", "det_std"]
DOMAIN = ["xi2", "state_pr", "state_count", "xplay_pr", "xplay_count"]
ARMS = ["TEST_IN", "S_sim", "S_pg", "S_decay", "S_win", "S_noise", "S_K4", "S_wide"]


def load():
    d = pd.DataFrame([json.loads(l) for l in open(OUT / "runs.jsonl")])
    base = [c[3:] for c in d.columns if c.startswith("w0_")]
    for b in base:
        d[b] = d[[f"w0_{b}", f"w1_{b}"]].median(axis=1, skipna=True)
    d["cls"] = d.K.clip(upper=3)
    return d, base


def classify(v, sign, t1, t2):
    s = sign * v
    k = 1 + (s > t1).astype(int) + (s > t2).astype(int)
    return np.where(np.isfinite(v), k, 2)


def cost(khat, k):
    return np.where(khat < k, 2 * (k - khat), khat - k)


def calibrate(v, y):
    best = None
    ok = np.isfinite(v)
    for sign in (1, -1):
        s = np.sort(np.unique(sign * v[ok]))
        cand = np.concatenate([[-np.inf], (s[1:] + s[:-1]) / 2, [np.inf]])
        sv = sign * v
        # precompute class indicator for each threshold
        above = np.array([np.where(ok, sv > t, False) for t in cand])  # (n_cand, n)
        for i in range(len(cand)):
            k2 = 1 + above[i].astype(int)
            for j in range(i, len(cand)):
                k = np.where(ok, k2 + above[j].astype(int), 2)
                acc = np.mean(k == y)
                c = np.mean(cost(k, y))
                key = (acc, -c)
                if best is None or key > best[0]:
                    best = (key, sign, cand[i], cand[j])
    return dict(acc=best[0][0], cost=-best[0][1], sign=best[1], t1=best[2], t2=best[3])


def main():
    d, base = load()
    cal = d[d.arm == "CAL"]
    y = cal.cls.values
    rules = {}
    for st in SCALAR + DOMAIN:
        names = [f"{st}_{r}" for r in FAMILIES[st]] if st in FAMILIES else [st]
        best = None
        for nm in names:
            if nm not in d.columns:
                continue
            r = calibrate(cal[nm].values.astype(float), y)
            r["var"] = nm
            if best is None or (r["acc"], -r["cost"]) > (best["acc"], -best["cost"]):
                best = r
        rules[st] = best
    rows = []
    for st, r in rules.items():
        row = {"stat": st, "var": r["var"], "sign": r["sign"], "CAL": round(r["acc"], 3)}
        for arm in ARMS:
            t = d[d.arm == arm]
            if not len(t):
                continue
            v = t[r["var"]].values.astype(float)
            k = classify(v, r["sign"], r["t1"], r["t2"])
            row[arm] = round(float(np.mean(k == t.cls.values)), 3)
            row[f"{arm}_cost"] = round(float(np.mean(cost(k, t.cls.values))), 3)
            ok = np.isfinite(v)
            row[f"{arm}_rho"] = round(float(spearmanr(v[ok], t.K.values[ok])[0]), 3) if ok.sum() > 5 else np.nan
            row[f"{arm}_nan"] = int((~ok).sum())
        rows.append(row)
    for name, kconst in (("always_2", 2), ("always_3", 3)):
        row = {"stat": name, "var": name, "sign": 0, "CAL": round(float(np.mean(y == kconst)), 3)}
        for arm in ARMS:
            t = d[d.arm == arm]
            if len(t):
                row[arm] = round(float(np.mean(t.cls.values == kconst)), 3)
                row[f"{arm}_cost"] = round(float(np.mean(cost(np.full(len(t), kconst), t.cls.values))), 3)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    json.dump(rules, open(OUT / "rules.json", "w"), indent=1, default=float)
    acc_cols = ["stat", "var", "CAL"] + [a for a in ARMS if a in res.columns]
    print("ACCURACY"); print(res[acc_cols].to_string(index=False))
    print("COST"); print(res[["stat"] + [f"{a}_cost" for a in ARMS if f"{a}_cost" in res.columns]].to_string(index=False))
    print("RHO"); print(res[["stat"] + [f"{a}_rho" for a in ARMS if f"{a}_rho" in res.columns]].to_string(index=False))
    # TEST_IN by noise level
    t = d[d.arm == "TEST_IN"]
    if len(t):
        out = []
        for st, r in rules.items():
            row = {"stat": st}
            for B, g in t.groupby("B"):
                k = classify(g[r["var"]].values.astype(float), r["sign"], r["t1"], r["t2"])
                row[f"B={B:g}"] = round(float(np.mean(k == g.cls.values)), 3)
            out.append(row)
        bt = pd.DataFrame(out); bt.to_csv(OUT / "test_in_by_B.csv", index=False)
        print("TEST_IN by B"); print(bt.to_string(index=False))
        # sensitivity: runs where all delay statistics are finite
        fin = np.isfinite(t[[rules[s]["var"] for s in SCALAR]].values).all(1)
        out = []
        for st, r in rules.items():
            g = t[fin]
            k = classify(g[r["var"]].values.astype(float), r["sign"], r["t1"], r["t2"])
            out.append({"stat": st, "acc_allfinite": round(float(np.mean(k == g.cls.values)), 3), "n": int(fin.sum())})
        print("TEST_IN, runs with all scalar stats finite"); print(pd.DataFrame(out).to_string(index=False))
    k4 = d[d.arm == "S_K4"]
    if len(k4):
        out = []
        for st, r in rules.items():
            v = k4[r["var"]].values.astype(float); K = k4.K.values
            a3, a4 = v[(K == 3) & np.isfinite(v)], v[(K == 4) & np.isfinite(v)]
            auc = np.mean([(r["sign"] * (b - a) > 0) + 0.5 * (b == a) for a in a3 for b in a4]) if len(a3) and len(a4) else np.nan
            out.append({"stat": st, "AUC_K4_vs_K3": round(float(auc), 3)})
        print("S_K4: K=4 vs K=3 separation (in the calibrated direction)"); print(pd.DataFrame(out).to_string(index=False))
    tm = pd.DataFrame(list(d.timing.values)).mean() / 2
    print("mean seconds per window:", tm.round(3).to_dict())
    print("mean sim seconds per run:", round(d.t_sim.mean(), 2), "attempts mean/max", d.attempts.mean(), d.attempts.max())


if __name__ == "__main__":
    main()
