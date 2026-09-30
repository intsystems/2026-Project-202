"""Analysis of cifar_events.py (E4): ratios, discrimination, controls, cost, figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parent / "results_cifar"
FIG = RES / "figures"
HALF, W = 4000, 500
SIMPLIFY = ("lr_step", "freeze", "prune")
NOT = ("base", "batch_up")
STATS = ["traj_PR", "MG_batch_loss", "MG_probe_loss", "MG_grad_norm", "MG_param_norm",
         "MGs_batch_loss", "MGs_probe_loss", "crossings_batch_loss", "crossings_probe_loss",
         "lag1_probe_loss", "det_std_probe_loss"]


def auc(a, b):
    """P(a < b): a from simplifying arms, b from the others."""
    return float(np.mean([(x < y) + 0.5 * (x == y) for x in a for y in b]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RES / "windows.csv")
    meta = pd.DataFrame(json.load(open(RES / "meta.json")))
    for o in ("batch_loss", "probe_loss"):
        d[f"rel_{o}"] = d[f"MG_{o}"] / d[f"MGs_{o}"]
        d[f"ident_{o}"] = d[f"MG20_{o}"] / d[f"MG_{o}"]
    pre = d[(d.start >= 2000) & (d.start + W <= HALF)]
    post = d[d.start >= HALF + W]
    cols = STATS + ["rel_batch_loss", "rel_probe_loss", "ident_probe_loss"]
    r = (post.groupby(["arm", "seed"])[cols].median()
         / pre.groupby(["arm", "seed"])[cols].median()).reset_index()
    out = {"ratio_median": r.groupby("arm")[cols].median().round(3).reset_index().to_dict("records"),
           "ratio_min": r.groupby("arm")[cols].min().round(3).reset_index().to_dict("records"),
           "ratio_max": r.groupby("arm")[cols].max().round(3).reset_index().to_dict("records")}
    real = r[r.arm.isin(SIMPLIFY + NOT)]
    lab = real.arm.isin(SIMPLIFY)
    out["auc_lower_in_simplifying"] = {c: auc(real[lab][c], real[~lab][c]) for c in cols}
    out["spearman_with_trajPR_ratio"] = {c: float(spearmanr(real[c], real.traj_PR)[0]) for c in cols}
    # observer controls: same trajectory as base
    b = r[r.arm == "base"].set_index("seed")
    out["controls"] = {}
    for ctl in ("scale", "smooth"):
        c = r[r.arm == ctl].set_index("seed")
        out["controls"][ctl] = {k: float((c[k] / b[k]).median()) for k in
                                ["MG_batch_loss", "MG_probe_loss", "crossings_probe_loss",
                                 "lag1_probe_loss", "det_std_probe_loss"]}
    # per-arm pairwise separation from base (all seeds): max of arm < min of base?
    out["separated_from_base"] = {}
    for a in SIMPLIFY + ("batch_up",):
        x = r[r.arm == a]
        out["separated_from_base"][a] = {c: bool(x[c].max() < b[c].min() or x[c].min() > b[c].max())
                                         for c in ["traj_PR", "MG_batch_loss", "MG_probe_loss",
                                                   "crossings_probe_loss"]}
    # window-level tracking within runs
    dd = d[d.arm.isin(SIMPLIFY + NOT) & (d.start >= 1000)]
    out["within_run_spearman_vs_trajPR"] = {
        c: float(np.median([spearmanr(g[c], g.traj_PR)[0] for _, g in dd.groupby(["arm", "seed"])]))
        for c in ["MG_batch_loss", "MG_probe_loss", "crossings_probe_loss"]}
    out["levels"] = {
        "rel_probe_loss_median": float(d.rel_probe_loss.median()),
        "rel_batch_loss_median": float(d.rel_batch_loss.median()),
        "ident_probe_loss_median": float(d.ident_probe_loss.median()),
    }
    out["cost"] = {
        "t_MG_window_ms": float(d.t_MG_batch_loss.median() * 1e3),
        "t_PR_window_ms": float(d.t_PR.median() * 1e3),
        "traj_MB_per_run": float(meta.traj_MB.median()),
        "log_KB_per_run": 8000 * 4 / 1024,
        "t_train_s": float(meta.t_train.median()),
        "t_probe_s": float(meta.t_probe.median()),
        "P": int(meta.P.iloc[0]),
    }
    out["accuracy"] = meta.groupby("arm")[["test_acc", "train_acc"]].median().round(3).reset_index().to_dict("records")
    json.dump(out, open(RES / "summary.json", "w"), indent=1, default=float)
    figures(d, r)
    pd.set_option("display.width", 220)
    print(r.groupby("arm")[["traj_PR", "MG_batch_loss", "MG_probe_loss", "MGs_probe_loss",
                            "crossings_probe_loss", "lag1_probe_loss", "det_std_probe_loss"]]
          .median().round(2).to_string())
    print(json.dumps({k: out[k] for k in ("auc_lower_in_simplifying", "spearman_with_trajPR_ratio",
                                          "controls", "separated_from_base",
                                          "within_run_spearman_vs_trajPR", "levels", "cost",
                                          "accuracy")}, indent=1, default=float))


def figures(d, r):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    arms = ["base", "batch_up", "lr_step", "freeze", "prune"]
    col = {"base": "#666666", "batch_up": "#997700", "lr_step": "#004488",
           "freeze": "#BB5566", "prune": "#117733"}
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a in arms:
        g = d[d.arm == a].groupby("start").median(numeric_only=True)
        for i, c in enumerate(["traj_PR", "MG_batch_loss", "MG_probe_loss"]):
            ax[i].plot(g.index + W / 2, g[c], color=col[a], label=a)
    for i, t in enumerate(["trajectory PR (all weights)", "MG, mini-batch loss (free)",
                           "MG, fixed-probe loss"]):
        ax[i].axvline(HALF, color="k", ls=":", lw=1); ax[i].set_title(t, fontsize=9)
        ax[i].set_xlabel("step")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "time_courses.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for i, c in enumerate(["MG_batch_loss", "MG_probe_loss", "crossings_probe_loss"]):
        for a in arms + ["smooth", "scale"]:
            g = r[r.arm == a]
            ax[i].scatter(g.traj_PR, g[c], color=col.get(a, "k"),
                          marker="o" if a in col else ("x" if a == "smooth" else "+"), label=a, s=20)
        ax[i].set_xlabel("trajectory PR, post / pre"); ax[i].set_ylabel(c + ", post / pre")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "ratios.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
