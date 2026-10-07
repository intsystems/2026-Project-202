"""Analysis of E8 (collapse.py): P1-P4, tables, figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detector import drop_series  # noqa: E402

RES = HERE / "results_collapse"
FIG = RES / "figures"
EVENT, W, S, HORIZON = 4000, 1000, 500, 5000
OBS_CTRL = ("scale", "smooth")


def loss_rise_rule():
    """Loss-rise detector, same (M, B) as the MG rule, delta from the E5 no-event runs."""
    rule = json.load(open(HERE / "results_detector" / "chosen_rules.json"))["MG"]
    worst = []
    for f in sorted((HERE / "results_graded").glob("logs_*_s*.npz")):
        arm = f.stem[5:].rsplit("_s", 1)[0]
        if arm not in ("base", "batch_up"):
            continue
        x = np.load(f)["batch_loss"]
        g = pd.DataFrame([{"start": a, "end": a + W, "loss_median": float(np.median(x[a:a + W]))}
                          for a in range(0, len(x) - W + 1, S)])
        ds = [d for _, d in drop_series(g, "loss_median", rule["M"], rule["B"], -1) if np.isfinite(d)]
        worst.append(-min(ds))
    return {"M": rule["M"], "B": rule["B"], "delta": max(worst) + 0.02, "sign": -1}


def first_alarm(g, stat, r):
    for end, d in drop_series(g, stat, r["M"], r["B"], r["sign"]):
        if np.isfinite(d) and d < -r["delta"]:
            return end
    return None


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(RES / "windows.csv")
    dead = pd.read_csv(RES / "dead.csv")
    meta = pd.DataFrame(json.load(open(RES / "meta.json")))
    rules = json.load(open(HERE / "results_detector" / "chosen_rules.json"))
    rules["loss_rise"] = loss_rise_rule()
    stat_of = {"MG": "MG", "crossings": "crossings", "lag1": "lag1", "det_std": "det_std",
               "loss_rise": "loss_median"}

    # ground truth per training run
    gt = []
    for (arm, seed), g in dead.groupby(["arm", "seed"]):
        pre = g[(g.step >= 2000) & (g.step < EVENT)].dead.median()
        post = g[(g.step >= 5000) & (g.step <= 9000)].dead.median()
        m = meta[(meta.arm == arm) & (meta.seed == seed)].iloc[0]
        gt.append({"arm": arm, "seed": seed, "dead_pre": pre, "dead_post": post, "d_dead": post - pre,
                   "train_acc": m.get("train_acc", np.nan),
                   "diverged": bool(pd.notna(m.get("diverged_at", np.nan)))})
    gt = pd.DataFrame(gt)
    gt["net_death"] = gt.train_acc < 0.2
    gt["collapse"] = gt.d_dead >= 0.05
    for c in OBS_CTRL:          # observer controls share base's trajectory
        b = gt[gt.arm == "base"].copy(); b["arm"] = c; gt = pd.concat([gt, b])

    rows = []
    for (arm, seed), g in w.groupby(["arm", "seed"]):
        pre = g[(g.start >= 2000) & (g.end <= EVENT)].MG.median()
        post = g[(g.start >= 5000) & (g.start <= 9000)].MG.median()
        row = {"arm": arm, "seed": seed, "MG_change": post / pre - 1}
        for name, r in rules.items():
            t = first_alarm(g, stat_of[name], r)
            row[f"alarm_{name}"] = t
            row[f"hit_{name}"] = t is not None and EVENT < t <= EVENT + HORIZON
            row[f"early_{name}"] = t is not None and t <= EVENT
        rows.append(row)
    runs = pd.DataFrame(rows).merge(gt, on=["arm", "seed"], how="left")
    runs.to_csv(RES / "per_run.csv", index=False)

    col = runs[runs.collapse & ~runs.net_death]
    non = runs[~runs.collapse]
    shocks = runs[runs.arm.isin(["shock20", "shock30"]) & ~runs.collapse]
    out = {"n_runs": len(runs), "n_collapse": int(runs.collapse.sum()),
           "n_net_death": int(runs.net_death.sum()), "n_non_collapse": len(non),
           "loss_rise_rule": rules["loss_rise"]}
    for name in rules:
        out[name] = {"hit_collapse": f"{int(col[f'hit_{name}'].sum())}/{len(col)}",
                     "alarm_non_collapse": f"{int(non[f'alarm_{name}'].notna().sum())}/{len(non)}",
                     "alarm_harmless_shocks": f"{int(shocks[f'alarm_{name}'].notna().sum())}/{len(shocks)}",
                     "median_delay": float(np.nanmedian((col[f'alarm_{name}'] - EVENT)
                                                        .where(col[f'hit_{name}'])))
                     if col[f"hit_{name}"].any() else None}
    tr = runs[~runs.arm.isin(OBS_CTRL)]
    out["P1_hit_rate_MG"] = float(col.hit_MG.mean()) if len(col) else None
    out["P2_false_alarm_rate_MG"] = float(non.alarm_MG.notna().mean())
    out["P3_spearman_MGchange_vs_ddead"] = float(spearmanr(tr.MG_change, tr.d_dead, nan_policy="omit")[0])
    out["P4_shock_alarms_loss_vs_MG"] = [int(shocks.alarm_loss_rise.notna().sum()),
                                        int(shocks.alarm_MG.notna().sum()), len(shocks)]
    per_arm = runs.groupby("arm").agg(
        dead_pre=("dead_pre", "median"), dead_post=("dead_post", "median"),
        collapse=("collapse", "sum"), net_death=("net_death", "sum"), n=("seed", "count"),
        MG_change=("MG_change", "median"), MG_alarm=("alarm_MG", lambda s: int(s.notna().sum())),
        loss_alarm=("alarm_loss_rise", lambda s: int(s.notna().sum())),
        acc=("train_acc", "median"))
    per_arm["MG_change"] *= 100
    out["per_arm"] = per_arm.round(3).reset_index().to_dict("records")
    json.dump(out, open(RES / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(per_arm.round(3).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "per_arm"}, indent=1, default=float))
    figures(runs, dead, w)


def figures(runs, dead, w):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"base": "#666", "scale": "#aaa", "smooth": "#aaa", "shock20": "#997700", "shock30": "#c9a227",
           "kill15x3": "#9ecae1", "kill20x2": "#4292c6", "kill50": "#2171b5", "kill100": "#08306b"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    tr = runs[~runs.arm.isin(OBS_CTRL)]
    for arm, g in tr.groupby("arm"):
        ax[0].scatter(100 * g.d_dead, 100 * g.MG_change, color=col[arm], s=22, label=arm,
                      marker="x" if arm.startswith("shock") or arm == "base" else "o")
    ax[0].axhline(-8.8, color="#BB5566", ls="--", lw=1, label="alarm threshold")
    ax[0].axvline(5, color="k", ls=":", lw=.8)
    ax[0].set_xlabel("dead ReLU channels: change after spike, % of 80 (expensive)")
    ax[0].set_ylabel("MG of parameter-norm log: change, %")
    ax[0].legend(frameon=False, fontsize=7, ncol=2)
    for arm in ["base", "shock30", "kill20x2", "kill100"]:
        g = dead[dead.arm == arm].groupby("step").dead.median()
        ax[1].plot(g.index, 100 * g.values, color=col[arm], label=arm)
    ax[1].axvline(EVENT, color="k", ls=":", lw=1)
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("dead channels, % (median over seeds)")
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "collapse_scatter.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for arm in ["base", "shock20", "shock30", "kill20x2", "kill50", "kill100"]:
        g = w[w.arm == arm].groupby("start").MG.median()
        ax.plot(g.index + W / 2, g.values, color=col[arm], label=arm)
    ax.axvline(EVENT, color="k", ls=":", lw=1)
    ax.set_xlabel("training step"); ax.set_ylabel("MG, parameter norm\n(median over seeds)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "mg_time.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
