"""Figure: test runs with spike onsets, log10 loss, and the frozen MG / competitor scores."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse as A  # noqa: E402

res = pd.read_csv(HERE / "results_test.csv")
res = res[res.budget == 2.0].set_index("method")
show = [m for m in ("MG", "self_repeat", "loss threshold", "grad-norm threshold", "INT attn-logit max")
        if m in res.index]
test = A.load(A.TEST_SEEDS)
test = [r for r in test if r["cfg"] in ("c1", "c2")][:4]
fig, ax = plt.subplots(1 + len(show), len(test), figsize=(4.2 * len(test), 1.7 * (1 + len(show))),
                       squeeze=False, sharex=True)
for j, r in enumerate(test):
    loss = np.load(A.RUNS / f"{r['cfg']}_s{r['seed']}.npz")["loss"]
    ax[0, j].plot(np.log10(loss), lw=0.3, color="0.3")
    ax[0, j].set_title(f"{r['cfg']} seed {r['seed']}: log10 train loss", fontsize=8)
    for i, m in enumerate(show, 1):
        row = res.loc[m]
        fam = row["rule"].rstrip("+-")
        B = int(fam[6:]) if fam.startswith("change") else 0
        sign = 1 if row["rule"].endswith("+") else -1
        sc = A.score(r, row["column"], fam[:6] if B else "level", B, sign)
        ok = r["scored"]
        ax[i, j].plot(r["t"], np.where(ok, sc, np.nan), lw=0.8)
        th = pd.read_csv(HERE / "results_cal_percolumn.csv")
        th = th[(th.budget == 2.0) & (th.col == row["column"])].theta.iloc[0]
        ax[i, j].axhline(th, color="k", ls="--", lw=0.6)
        ax[i, j].set_title(f"{m}: {row['column']} {row['rule']}", fontsize=7)
    for a in ax[:, j]:
        for o in r["all_onsets"]:
            a.axvline(o, color="r", lw=0.7)
            a.axvspan(o - A.H, o, color="r", alpha=0.08)
ax[-1, 0].set_xlabel("step")
fig.tight_layout()
fig.savefig(HERE / "fig_test_runs.png", dpi=90)
print("saved")
