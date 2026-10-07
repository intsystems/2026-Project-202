"""Why: per base run, how strongly does each monitor track plasticity loss over tasks?
rho_acc = within-run Spearman(v_j, online acc_j); snr = |linear trend over 40 tasks| / residual std.
Median over runs of the losing conditions (all except S1W256) in both families, all seeds."""
import glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
df = pd.concat([pd.read_csv(f) for f in glob.glob("runs/mon_*.csv")])
mons = ["acc_mean", "pnorm_end", "dormant_0.0", "srank"] + [f"{s}|{L}|{w}" for s in ("MG", "self_repeat", "spectral_entropy", "roughness", "linear_pr") for L in ("pnorm", "loss") for w in ("span", "within")]
rows = []
for (fam, c, sd), h in df[df.cond != "S1W256"].groupby(["family", "cond", "seed"]):
    h = h.sort_values("j")
    acc = h.acc_mean.to_numpy()
    for m in mons:
        v = h[m].to_numpy(float); ok = np.isfinite(v); j = h.j.to_numpy()[ok]
        p = np.polyfit(j, v[ok], 1); res = v[ok] - np.polyval(p, j)
        rows.append({"fam": fam, "monitor": m, "rho_acc": spearmanr(v[ok], acc[ok])[0],
                     "snr": abs(p[0]) * 40 / (res.std() + 1e-12)})
t = pd.DataFrame(rows).groupby(["monitor", "fam"])[["rho_acc", "snr"]].median().unstack()
t.columns = [f"{a}_{b}" for a, b in t.columns]
t = t.sort_values("snr_F1", ascending=False)
t.to_csv("results/tracking_snr.csv")
print(t.round(2).to_string())
