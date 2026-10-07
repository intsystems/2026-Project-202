"""S3b analysis (protocol in anneal.py): decay-and-stop under a mean compute budget C."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse as A  # noqa: E402
import rules as R  # noqa: E402

ANN = HERE / "results_anneal"


def attach(runs):
    out = []
    for r in runs:
        f = ANN / f"ann_{r['tag']}.json"
        if f.exists():
            r["ann"] = json.load(open(f))
            out.append(r)
    return out


def kk(r, key):
    K = r["ann"]["K"]
    return f"K{K}" if key == "full" else f"K{K // 2}"


def outcome(r, alarm, key):
    a = r["ann"]
    G, K, T = a["grid"], a["K"], r["T"]
    tj = G[-1]
    if alarm is not None:
        for t in G:
            if t >= alarm:
                tj = t
                break
    o = a["anneal"][str(tj)][kk(r, key)]
    return o["test_acc"], o["test_loss"], (tj + K) / T


def evaluate(runs, fn, p, key):
    z = np.array([outcome(r, fn(r, p), key) for r in runs])
    return z  # columns acc, loss, compute


def calibrate(cal, grid, fn, C, key, cache):
    best, fallback = None, None
    for i, p in enumerate(grid):
        z = cache.setdefault(i, evaluate(cal, fn, p, key))
        acc, comp = z[:, 0].mean(), z[:, 2].mean()
        if comp <= C + 1e-9 and (best is None or acc > best[0] + 1e-12):
            best = (acc, comp, p)
        if fallback is None or comp < fallback[1]:
            fallback = (acc, comp, p)
    return best or fallback


def frontier(test, key):
    """mean test acc of every fixed decision time f=j/16 on the test units -> (compute, acc)."""
    G = test[0]["ann"]["grid"]
    rows = []
    for j in range(len(G)):
        z = []
        for r in test:
            g = r["ann"]["grid"]
            o = r["ann"]["anneal"][str(g[j])][kk(r, key)]
            z.append((o["test_acc"], (g[j] + r["ann"]["K"]) / r["T"]))
        rows.append(np.array(z))
    return rows  # list over j of (n_units, 2)


def front_interp(fr, idx, comp):
    cs = np.array([z[idx, 1].mean() for z in fr])
    acc = np.array([z[idx, 0].mean() for z in fr])
    return float(np.interp(comp, cs, acc))


def oracle_alloc(test, C, key):
    """per-unit choice of decision time maximising mean acc s.t. mean compute <= C (Lagrangian)."""
    best = None
    for lam in np.linspace(0, 0.5, 501):
        accs, comps = [], []
        for r in test:
            g, K, T = r["ann"]["grid"], r["ann"]["K"], r["T"]
            opts = [(r["ann"]["anneal"][str(t)][kk(r, key)]["test_acc"], (t + K) / T) for t in g]
            a, c = max(opts, key=lambda z: z[0] - lam * z[1])
            accs.append(a); comps.append(c)
        if np.mean(comps) <= C + 1e-9 and (best is None or np.mean(accs) > best[0]):
            best = (float(np.mean(accs)), float(np.mean(comps)))
    return best


def run(C=0.5, key="full", restrict_tmin=None, verbose=True):
    runs = attach(A.load_runs())
    cal = [r for r in runs if r["split"] == "cal"]
    test = [r for r in runs if r["split"] == "test"]
    quant = {}
    for L in A.LOGS:
        for s in A.ALL_STATS:
            v = np.concatenate([r["win"][L][s] for r in cal])
            v = v[np.isfinite(v)]
            quant[(L, s)] = {q: float(np.quantile(v, q)) for q in R.QS}
    fam = A.families(quant)
    fam["fixed_step"] = ([{"f": j / 16} for j in range(2, 15)], R.fixed)
    fr = frontier(test, key)
    rows, per = [], {}
    rng = np.random.default_rng(0)
    n = len(test)
    BIDX = rng.integers(0, n, (3000, n))
    for name, (grid, fn) in fam.items():
        if restrict_tmin and name != "fixed_step":
            grid = [p for p in grid if p.get("tmin") == restrict_tmin]
        acc_c, comp_c, p = calibrate(cal, grid, fn, C, key, {})
        z = evaluate(test, fn, p, key)
        per[name] = z
        gain = z[:, 0].mean() - front_interp(fr, np.arange(n), z[:, 2].mean())
        gb = np.array([z[b, 0].mean() - front_interp(fr, b, z[b, 2].mean()) for b in BIDX[:1000]])
        rows.append({"rule": name, "cal_acc": acc_c, "cal_compute": comp_c, "test_acc": z[:, 0].mean(),
                     "test_compute": z[:, 2].mean(), "test_loss": z[:, 1].mean(),
                     "gain_vs_fixed_frontier": gain, "gain_ci_lo": np.quantile(gb, 0.025),
                     "gain_ci_hi": np.quantile(gb, 0.975), "params": json.dumps(p)})
    tab = pd.DataFrame(rows)
    mg = per["MG"]
    for name, z in per.items():
        d = mg[:, 0] - z[:, 0]
        bs = d[BIDX].mean(1)
        tab.loc[tab.rule == name, "MG_minus_rule_acc"] = d.mean()
        tab.loc[tab.rule == name, "ci_lo"] = np.quantile(bs, 0.025)
        tab.loc[tab.rule == name, "ci_hi"] = np.quantile(bs, 0.975)
        tab.loc[tab.rule == name, "MG_minus_rule_compute"] = (mg[:, 2] - z[:, 2]).mean()
    orc = oracle_alloc(test, C, key)
    tab = tab.sort_values("gain_vs_fixed_frontier", ascending=False)
    if verbose:
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
        print(f"\n=== S3b C={C} anneal={key} tmin={'all' if not restrict_tmin else restrict_tmin} "
              f"cal {len(cal)} test {n}; oracle allocation (upper bound) acc {orc[0]:.4f} compute {orc[1]:.3f}")
        print(tab.drop(columns=["params"]).round(4).to_string(index=False))
    return tab, per, orc


if __name__ == "__main__":
    allt = []
    for C in (0.5, 0.75):
        tab, per, orc = run(C)
        tab["C"] = C; tab["oracle_alloc_acc"] = orc[0]
        allt.append(tab)
    tab, _, orc = run(0.5, restrict_tmin=2 / 16)
    tab["C"] = "0.5_pure"; tab["oracle_alloc_acc"] = orc[0]
    allt.append(tab)
    tab, _, orc = run(0.5, key="half")
    tab["C"] = "0.5_halfK"; tab["oracle_alloc_acc"] = orc[0]
    allt.append(tab)
    pd.concat(allt).to_csv(ANN / "s3b_tables.csv", index=False)
    print(pd.concat(allt)[["C", "rule", "params"]].to_string(index=False))
