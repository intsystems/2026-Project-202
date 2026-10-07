"""E10 analysis: truth classes, selector AUC / top-1 / top-5 per (target, data seed)."""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
tag = sys.argv[1] if len(sys.argv) > 1 else "confirm"
M = pd.DataFrame(json.load(open(f"results_esn/models_{tag}.json")))
Dd = {(d["target"], d["dseed"]): d for d in json.load(open(f"results_esn/data_{tag}.json"))}
STATS = ["MG", "spectral_entropy", "self_repeat", "roughness", "peak_count", "perm_entropy",
         "recurrence_rate", "corr_dim", "twonn", "linear_pr"]
FREQS = {"T2": [1, (1 + 5 ** .5) / 2], "T3": [1, (1 + 5 ** .5) / 2, 1 + 2 ** .5]}

def faithful(r):
    if r.diverged or not (0.5 < r.std_ratio < 2) or r.lyap is None:
        return False
    L = np.array(r.lyap)
    if r.target == "lorenz":
        return bool(0.5 * 0.0181 <= L[0] <= 1.5 * 0.0181 and abs(L[1]) <= 0.003)
    q = int(r.target[1:])
    f = np.array(FREQS[r.target]) / 40.37
    pf = r.peak_freqs if isinstance(r.peak_freqs, list) else []
    fok = all(any(abs(p - fi) / fi < 0.015 for p in pf[:6]) for fi in f)
    return bool(L[0] <= 0.003 and (L >= -0.003).sum() == q and fok)

M["faithful"] = M.apply(faithful, axis=1)
M["pool"] = (~M.diverged) & (M.std_ratio > 0.5) & (M.std_ratio < 2)
score = {"MSE_1step": lambda g: -g.mse1, "VPT": lambda g: g.vpt, "D_H": lambda g: -g.D_H}
for s in STATS:
    score[f"|{s}|"] = (lambda s: lambda g: -(g[s] - Dd[(g.target.iloc[0], g.dseed.iloc[0])][s]).abs())(s)
rows = []
for (t, d), g in M[M.pool].groupby(["target", "dseed"]):
    for name, f in score.items():
        sc = f(g).astype(float).to_numpy()
        fin = np.isfinite(sc)
        sc = np.where(fin, sc, (sc[fin].min() - 1) if fin.any() else 0)
        y = g.faithful.to_numpy()
        order = np.argsort(-sc, kind="stable")
        rows.append({"target": t, "dseed": d, "selector": name, "pool": len(g), "n_faithful": int(y.sum()),
                     "auc": roc_auc_score(y, sc) if 0 < y.sum() < len(y) else np.nan,
                     "top1": bool(y[order[0]]), "top5": float(y[order[:5]].mean())})
R = pd.DataFrame(rows)
R.to_csv(f"results_esn/selection_{tag}.csv", index=False)
cls = M.groupby("target").apply(lambda g: pd.Series({"models": len(g), "pool": int(g.pool.sum()),
                                                     "faithful": int(g.faithful.sum())}))
print(cls.to_string())
summ = R.groupby("selector").agg(auc=("auc", "mean"), top1=("top1", "sum"), top5=("top5", "mean")).sort_values("auc", ascending=False)
print(summ.round(3).to_string())
print(R.pivot_table(index="selector", columns="target", values="auc").round(3).to_string())
print(R.pivot_table(index="selector", columns="target", values="top5").round(2).to_string())
