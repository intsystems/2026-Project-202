"""Figures for REPORT_ru.md from results_main/*.csv (English labels)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results_main")
INTERNAL = {"rankme_h", "rankme_z", "alpha_h", "abs_alpha1", "lidar_z", "out_std"}
C_MG, C_SCALAR, C_INT = "#2a78d6", "#b9b8b2", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def color(s):
    return C_MG if s.startswith("MG") else (C_INT if s in INTERNAL else C_SCALAR)


panels = [("T1_config_ranking.csv", "test_rho", "T1: Spearman rho with final probe acc"),
          ("T2_bad_run_auc.csv", "test_auc", "T2: AUC, failed runs (step 3000)"),
          ("T3_early_warning_auc.csv", "test_auc", "T3: AUC, early warning (step 1500)")]
fig, axes = plt.subplots(1, 3, figsize=(13, 6.5))
for ax, (fn, col, title) in zip(axes, panels):
    d = pd.read_csv(R / fn)
    d = d[d.stat.isin(d.stat) & d[col].notna() & ~d.stat.isin(["random_choice", "default_ss_base"])]
    d = d.sort_values(col)
    ax.barh(range(len(d)), d[col], color=[color(s) for s in d.stat], height=0.7, edgecolor="#fcfcfb", linewidth=2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{s}  [{c.split('|')[1] if '|' in str(c) else 'emb'}]" for s, c in zip(d.stat, d.col)], fontsize=7.5)
    for i, v in enumerate(d[col]):
        ax.text(v + 0.01 if v >= 0 else 0.01, i, f"{v:.2f}", va="center", fontsize=7, color="#52514e")
    ax.set_title(title, fontsize=9.5, color="#0b0b0b")
    ax.axvline(0.5 if "auc" in col else 0, color="#52514e", lw=0.8, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
from matplotlib.patches import Patch  # noqa: E402
fig.legend(handles=[Patch(color=C_MG, label="MG (scalar log)"), Patch(color=C_SCALAR, label="other scalar-log statistic"),
                    Patch(color=C_INT, label="embedding-based proxy (RankMe, LiDAR, alpha-ReQ, std)")],
           loc="lower center", ncol=3, frameon=False)
fig.suptitle("Label-free selection in SSL (MNIST, 29 configs, test seeds 10-12); log/sign chosen on calibration seeds",
             fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(R / "fig_selectors.png", dpi=130)

# scatter: final probe accuracy vs the calibrated MG score and vs RankMe(h)
t = pd.read_csv(R / "test_final.csv")
t1 = pd.read_csv(R / "T1_config_ranking.csv").set_index("stat")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, st in zip(axes, ["MG", "rankme_h"]):
    col = t1.loc[st, "col"]
    for fam, mk in zip(["ss", "byol", "vic", "clr"], ["o", "s", "^", "D"]):
        g = t[t.family == fam]
        ax.scatter(g[col], g.acc, s=22, marker=mk, label=fam, alpha=0.8, edgecolor="#fcfcfb", linewidth=0.6)
    ax.set_xlabel(f"{col} (sign {int(t1.loc[st, 'sign']):+d})")
    ax.set_ylabel("final linear-probe accuracy")
    ax.set_ylim(max(0.0, t.acc.min() - 0.02), 1.0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(R / "fig_scatter.png", dpi=130)
print("ok")
