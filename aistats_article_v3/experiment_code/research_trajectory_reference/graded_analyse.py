"""Analysis of E5 (cifar_graded.py): tests P1-P4, tables in per cent, figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parent / "results_graded"
FIG = RES / "figures"
EVENT, W = 4000, 1000
FAMILIES = {"lr": ["lr3", "lr10", "lr100"],
            "freeze": ["freeze12", "freeze_head", "freeze_bias"],
            "prune": ["prune50", "prune80", "prune95"]}
EVENTS = sum(FAMILIES.values(), [])
CONTROLS = ["base", "batch_up", "scale", "smooth"]
ORDER = CONTROLS + EVENTS
LABEL = {"base": "no event", "batch_up": "batch x4 (less noise)", "scale": "log x10",
         "smooth": "log smoothed", "lr3": "lr / 3", "lr10": "lr / 10", "lr100": "lr / 100",
         "freeze12": "freeze conv1-2 (65% move)", "freeze_head": "train head only (2.2%)",
         "freeze_bias": "train head bias only (0.07%)", "prune50": "prune 50%",
         "prune80": "prune 80%", "prune95": "prune 95%"}
COLS = ["MG", "MG_surr", "crossings", "lag1", "det_std", "update_PR", "moving_frac",
        "update_size", "MG_batch_loss"]


def per_run(d):
    pre = d[(d.start >= 2000) & (d.start + W <= EVENT)].groupby(["arm", "seed"])
    post = d[(d.start >= 5000) & (d.start <= 9000)].groupby(["arm", "seed"])
    cols = [c for c in COLS if c in d]
    a, b = pre[cols].median(), post[cols].median()
    r = (b / a).add_suffix("_r")
    return pd.concat([a.add_suffix("_pre"), b.add_suffix("_post"), r], axis=1).reset_index()


def auc(ev, ct):
    return float(np.mean([(x < y) + .5 * (x == y) for x in ev for y in ct]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RES / "windows.csv")
    meta = pd.DataFrame(json.load(open(RES / "meta.json")))
    r = per_run(d)
    r.to_csv(RES / "per_run.csv", index=False)
    g = r.groupby("arm")
    tab = pd.DataFrame({
        "MG before": g.MG_pre.median(), "MG after": g.MG_post.median(),
        "change %": 100 * (g.MG_r.median() - 1),
        "min %": 100 * (g.MG_r.min() - 1), "max %": 100 * (g.MG_r.max() - 1),
        "update PR change %": 100 * (g.update_PR_r.median() - 1),
        "moving params after": g.moving_frac_post.median(),
        "update size change %": 100 * (g.update_size_r.median() - 1),
    }).reindex(ORDER)
    tab.index = [LABEL[i] for i in tab.index]
    out = {"table": tab.round(3).reset_index().to_dict("records")}

    med = g.MG_r.median()
    ctl_runs = r[r.arm.isin(CONTROLS)].MG_r
    real_ctl = r[r.arm.isin(["base", "batch_up"])].MG_r
    out["P1_every_event_median_below_every_control_median"] = bool(med[EVENTS].max() < med[CONTROLS].min())
    out["P2_strongest_dose_below_every_control_run"] = {
        fam: bool(r[r.arm == arms[-1]].MG_r.max() < ctl_runs.min()) for fam, arms in FAMILIES.items()}
    out["P3_dose_spearman"] = {}
    for fam, arms in FAMILIES.items():
        sub = r[r.arm.isin(arms)]
        dose = sub.arm.map({a: i for i, a in enumerate(arms)})
        out["P3_dose_spearman"][fam] = float(spearmanr(dose, sub.MG_r)[0])
    base_med = med["base"]
    out["P4"] = {"scale_minus_base": float(med["scale"] - base_med),
                 "smooth_minus_base": float(med["smooth"] - base_med),
                 "batch_up_minus_base": float(med["batch_up"] - base_med)}
    out["auc_events_vs_base_batchup"] = {
        c: auc(r[r.arm.isin(EVENTS)][c + "_r"], r[r.arm.isin(["base", "batch_up"])][c + "_r"])
        for c in ["MG", "MG_surr", "crossings", "lag1", "det_std", "MG_batch_loss"]}
    out["auc_strongest_vs_all_controls"] = {
        c: auc(r[r.arm.isin([a[-1] for a in FAMILIES.values()])][c + "_r"],
               r[r.arm.isin(CONTROLS)][c + "_r"]) for c in ["MG", "crossings", "lag1", "det_std"]}
    ev = r[r.arm.isin(EVENTS)]
    out["spearman_MG_vs_updatePR_ratio_all_runs"] = float(spearmanr(r.MG_r, r.update_PR_r, nan_policy="omit")[0])
    out["rel_level_median"] = float((d.MG / d.MG_surr).median())
    out["cost"] = {"t_MG_window_ms": float(d.t_MG.median() * 1e3), "P": int(meta.P.iloc[0]),
                   "log_KB": 10000 * 4 / 1024, "traj_MB": 10000 * int(meta.P.iloc[0]) * 4 / 2 ** 20}
    out["test_acc"] = meta.groupby("arm").test_acc.median().round(3).to_dict()
    json.dump(out, open(RES / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(tab.round(2).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1, default=float))
    figures(d, r)


def figures(d, r):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"lr": "#004488", "freeze": "#BB5566", "prune": "#117733"}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, a in enumerate(ORDER):
        v = 100 * (r[r.arm == a].MG_r.values - 1)
        fam = next((f for f, arms in FAMILIES.items() if a in arms), None)
        c = col.get(fam, "#777")
        ax[0].scatter(np.full(len(v), i) + np.linspace(-.15, .15, len(v)), v, color=c, s=18, zorder=3)
        ax[0].hlines(np.median(v), i - .3, i + .3, color=c, lw=2)
    ax[0].axhline(0, color="k", lw=.6, ls=":")
    ax[0].axvspan(-.5, 3.5, color="#999", alpha=.08)
    ax[0].set_xticks(range(len(ORDER))); ax[0].set_xticklabels([LABEL[a] for a in ORDER], rotation=60, ha="right", fontsize=7.5)
    ax[0].set_ylabel("MG of parameter-norm log:\nchange after event, %")
    ax[0].set_title("grey: no simplification  |  colour: simplification events", fontsize=8.5)
    for fam, arms in FAMILIES.items():
        sub = r[r.arm.isin(arms)]
        x = 100 * (sub.update_PR_r - 1); y = 100 * (sub.MG_r - 1)
        ax[1].scatter(x, y, color=col[fam], s=18, label=fam)
    sub = r[r.arm.isin(["base", "batch_up"])]
    ax[1].scatter(100 * (sub.update_PR_r - 1), 100 * (sub.MG_r - 1), color="#777", s=18, label="controls")
    ax[1].axhline(0, color="k", lw=.6, ls=":"); ax[1].axvline(0, color="k", lw=.6, ls=":")
    ax[1].set_xlabel("update PR (all weights, expensive): change, %")
    ax[1].set_ylabel("MG change, %"); ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "mg_change.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for j, (fam, arms) in enumerate(FAMILIES.items()):
        for a, c in zip(["base"] + arms, ["#777", "#9ecae1", "#4292c6", "#08306b"]):
            gg = d[d.arm == a].groupby("start").MG.median()
            ax[j].plot(gg.index + W / 2, gg.values, color=c, label=LABEL[a])
        ax[j].axvline(EVENT, color="k", ls=":", lw=1); ax[j].set_title(fam); ax[j].set_xlabel("training step")
        ax[j].legend(frameon=False, fontsize=6.5)
    ax[0].set_ylabel("MG, parameter norm\n(median of 4 seeds)")
    fig.tight_layout(); fig.savefig(FIG / "time_courses.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
