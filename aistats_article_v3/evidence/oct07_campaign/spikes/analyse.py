"""Calibrate every warning rule on calibration runs, score frozen rules on test runs.
Implements the rule/metric part of the protocol in spikes_main.py (nothing tuned on test)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import truth  # noqa: E402
from spikes_main import CONFIGS, CAL_SEEDS, TEST_SEEDS  # noqa: E402

RUNS = HERE / "runs"
W, S, T0, H = 500, 50, 1500, 500
BUDGETS = (2.0, 5.0)
SCALAR = ("loss", "grad_norm", "update_norm", "param_norm")
WSTAT = ("MG", "MG_t4k50", "self_repeat", "spectral_entropy", "roughness", "perm_entropy",
         "recurrence_rate", "corr_dim", "twonn", "linear_pr", "crossings", "lag1", "det_std", "var")
LEVEL_LOGS = SCALAR + ("attn_max", "attn_ent", "logz")
RULES = [("level", 0, 1), ("level", 0, -1)] + [("change", B, sg) for B in (4, 10) for sg in (1, -1)]


def load(seeds):
    runs = []
    for s in seeds:
        for c in CONFIGS:
            f = RUNS / f"{c}_s{s}_feat.csv"
            if not f.exists():
                continue
            r = np.load(RUNS / f"{c}_s{s}.npz")
            fe = pd.read_csv(f)
            ons = truth.spikes(r["loss"])
            t = fe.t.to_numpy()
            scored = t >= T0
            for o in ons:
                scored &= ~((t >= o) & (t <= o + W))
            pos = np.zeros(len(t), bool)
            for o in ons:
                pos |= (t < o) & (o <= t + H)
            runs.append({"cfg": c, "seed": s, "fe": fe, "t": t, "onsets": [o for o in ons if o > T0],
                         "all_onsets": ons, "scored": scored, "pos": pos,
                         "n_steps": int(np.isfinite(r["loss"]).sum())})
    return runs


def score(run, col, fam, B, sign):
    x = run["fe"][col].to_numpy(float) if col in run["fe"] else np.full(len(run["t"]), np.nan)
    if fam == "level":
        return sign * x
    out = np.full(len(x), np.nan)
    lag0 = W // S
    for i in range(len(x)):
        j = [i - lag0 - k for k in range(B)]
        if j[-1] >= 0:
            out[i] = sign * (x[i] - np.nanmedian(x[j]))
    return out


def evaluate(runs, scores, theta):
    hits, leads, fa, n_alarm, steps = 0, [], 0, 0, 0
    n_on = 0
    for run, sc in zip(runs, scores):
        t, ok = run["t"], run["scored"] & np.isfinite(sc)
        al = ok & (sc >= theta)
        steps += run["scored"].sum() * S
        for o in run["onsets"]:
            n_on += 1
            w = al & (t >= o - H) & (t < o)
            if w.any():
                hits += 1
                leads.append(o - t[w].min())
        last = -10 ** 9
        for ti in t[al & ~run["pos"]]:
            if ti - last >= H:
                fa += 1
                last = ti
        n_alarm += int(al.sum())
    fa_rate = fa / max(steps, 1) * 1e4
    tp = sum(int(((run["scored"] & np.isfinite(sc) & (sc >= theta)) & run["pos"]).sum()) for run, sc in zip(runs, scores))
    return {"recall": hits / max(n_on, 1), "hits": hits, "n_spikes": n_on, "fa_per10k": fa_rate, "fa": fa,
            "precision": tp / n_alarm if n_alarm else np.nan,
            "lead": float(np.median(leads)) if leads else np.nan}


def calibrate_threshold(runs, scores, budget):
    vals = np.concatenate([sc[r["scored"] & np.isfinite(sc)] for r, sc in zip(runs, scores)])
    if len(vals) == 0:
        return None, None
    cand = np.unique(vals)
    # FA count is monotone non-increasing in theta: binary search for the lowest admissible theta
    lo, hi = 0, len(cand) - 1
    if evaluate(runs, scores, cand[hi] + 1e-12)["fa_per10k"] > budget:
        return None, None
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ev = evaluate(runs, scores, cand[mid])
        if ev["fa_per10k"] <= budget:
            best = (cand[mid], ev)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        th = cand[-1] + 1e-12
        return th, evaluate(runs, scores, th)
    return best


def auc(runs, scores):
    y, s = [], []
    for r, sc in zip(runs, scores):
        ok = r["scored"] & np.isfinite(sc)
        y.append(r["pos"][ok]); s.append(sc[ok])
    y, s = np.concatenate(y), np.concatenate(s)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)


def methods(columns):
    m = {}
    for st in WSTAT:
        m[st] = [f"{st}|{k}" for k in SCALAR]
    for k, name in (("loss", "loss threshold"), ("grad_norm", "grad-norm threshold"),
                    ("update_norm", "update-norm level"), ("param_norm", "param-norm level"),
                    ("attn_max", "INT attn-logit max"), ("attn_ent", "INT attn entropy"),
                    ("logz", "INT output logZ")):
        m[name] = [f"{a}|{k}" for a in ("level", "max", "min")]
    for k in SCALAR:
        m[f"MG|{k} only"] = [f"MG|{k}"]
    m["best non-MG (any column)"] = [c for c in columns if not c.startswith("MG")]
    return m


def main():
    cal, test = load(CAL_SEEDS), load(TEST_SEEDS)
    out = {}
    print("cal runs", len(cal), "spikes", sum(len(r["onsets"]) for r in cal), "(all", sum(len(r["all_onsets"]) for r in cal), ")",
          "| test runs", len(test), "spikes", sum(len(r["onsets"]) for r in test))
    for r in cal + test:
        out.setdefault("onsets", {})[f"{r['cfg']}_s{r['seed']}"] = r["all_onsets"]
    cols = [c for c in cal[0]["fe"].columns if c != "t"]
    rows = []
    percol = {}
    for col in cols:
        best = {b: None for b in BUDGETS}
        for fam, B, sg in RULES:
            sc = [score(r, col, fam, B, sg) for r in cal]
            for budget in BUDGETS:
                th, ev = calibrate_threshold(cal, sc, budget)
                if th is None:
                    continue
                key = (ev["recall"], -ev["fa_per10k"], ev["lead"] if np.isfinite(ev["lead"]) else -1)
                if best[budget] is None or key > best[budget][0]:
                    best[budget] = (key, {"col": col, "fam": fam, "B": B, "sign": sg, "theta": float(th),
                                          **{f"cal_{k}": v for k, v in ev.items()}})
        for budget in BUDGETS:
            if best[budget]:
                percol[(budget, col)] = best[budget][1]
    for budget in BUDGETS:
        for name, allowed in methods(cols).items():
            cands = [percol[(budget, c)] for c in allowed if (budget, c) in percol]
            if not cands:
                continue
            ch = max(cands, key=lambda d: (d["cal_recall"], -d["cal_fa_per10k"],
                                           d["cal_lead"] if np.isfinite(d["cal_lead"]) else -1))
            sc = [score(r, ch["col"], ch["fam"], ch["B"], ch["sign"]) for r in test]
            ev = evaluate(test, sc, ch["theta"])
            a, ap = auc(test, sc)
            rows.append({"budget": budget, "method": name, "column": ch["col"],
                         "rule": f"{ch['fam']}{ch['B'] or ''}{'+' if ch['sign'] > 0 else '-'}",
                         "cal_recall": ch["cal_recall"], "cal_fa10k": ch["cal_fa_per10k"],
                         "test_recall": ev["recall"], "test_hits": f"{ev['hits']}/{ev['n_spikes']}",
                         "test_fa10k": ev["fa_per10k"], "test_precision": ev["precision"],
                         "test_lead": ev["lead"], "test_AUROC": a, "test_AP": ap})
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "results_test.csv", index=False)
    pc = pd.DataFrame([{"budget": b, **v} for (b, c), v in percol.items()])
    pc.to_csv(HERE / "results_cal_percolumn.csv", index=False)
    json.dump(out, open(HERE / "onsets.json", "w"), indent=0, default=int)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
