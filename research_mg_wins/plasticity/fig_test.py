"""Figure: test utility (mean online accuracy) of every calibrated trigger, F1 and F2."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
DOM = {"dormant_0.0", "dormant_0.025", "dormant_0.1", "srank", "erank"}
LEV = {"acc_mean", "loss_mean", "gnorm_mean", "pnorm_end"}
fig, axs = plt.subplots(1, 2, figsize=(11, 6.5), constrained_layout=True)
for ax, fam in zip(axs, ("F1", "F2")):
    t = pd.read_csv(f"results/{fam}_by_stat.csv")
    t = t[(t.scope == "any") & (t.stat != "oracle_fixed_per_cond") & (t.stat != "never")]
    orc = pd.read_csv(f"results/{fam}_by_stat.csv").query("scope=='any' and stat=='oracle_fixed_per_cond'").test_util.iloc[0]
    t = t.sort_values("test_util")
    col = ["#e34948" if s in ("MG", "MG_t4") else "#2a78d6" if s in DOM else "#1baf7a" if s in LEV
           else "#8a8984" if s.startswith("fixed") or s == "every_task" else "#eda100" for s in t.stat]
    ax.hlines(range(len(t)), lo0 := t.test_util.min() - 0.004, t.test_util, color="#e6e6e3", lw=1)
    ax.scatter(t.test_util, range(len(t)), c=col, s=36, zorder=3)
    ax.set_yticks(range(len(t)), t.monitor.str.replace("|", " ", regex=False), fontsize=7)
    ax.axvline(orc, color="#0b0b0b", lw=1, ls="--")
    ax.text(orc, len(t) - 0.5, " oracle fixed\n per condition", fontsize=7, va="top")
    lo = t.test_util.min() - 0.004
    ax.set_xlim(lo, orc + 0.004)
    ax.set_title(f"{fam}: test mean online accuracy with resets", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e6e3", lw=0.6)
fig.text(0.01, 0.005, "red: MG; blue: internal domain monitors; green: log levels; orange: other scalar statistics; grey: fixed schedules", fontsize=8)
fig.savefig("fig_test_utility.png", dpi=130)
