"""Analysis and figures for eos.py (E2). Writes results_eos/summary.json and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parent / "results_eos"
FIG = RES / "figures"
BURN = 2000          # windows starting earlier are the approach to the edge
STAT = ["MG", "MG_surr", "crossings", "rises", "spec_entropy", "lag1"]
BLUE, ROSE, GOLD, GREY = "#004488", "#BB5566", "#997700", "#666666"


def within_run_rho(d, x, y):
    out = []
    for _, g in d.groupby(["eta0", "eta1", "seed"]):
        g = g.dropna(subset=[x, y])
        if len(g) >= 5 and g[x].std() > 0 and g[y].std() > 0:
            out.append(spearmanr(g[x], g[y])[0])
    return float(np.median(out)) if out else float("nan"), len(out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(RES / "windows_all.csv")
    ck = pd.read_csv(RES / "checkpoints_all.csv")
    meta = json.load(open(RES / "meta_all.json"))
    w["rel"] = w.MG / w.MG_surr
    w["rel_smooth"] = w.MG_smooth / w.MG_surr_smooth
    w["ident"] = w.MG_2E / w.MG
    out = {"diverged": [m["tag"] for m in meta if m.get("diverged")]}

    # ---------------- sweep: level across eta
    sw = w[(w.stage == "sweep") & (w.start >= BURN)]
    lvl = sw.groupby(["eta0", "seed"]).median(numeric_only=True).reset_index()
    tab = lvl.groupby("eta0")[["n_unstable", "n_unstable_97", "traj_PR", "MG", "MG_surr",
                               "rel", "ident", "crossings", "rises", "spec_entropy", "lag1",
                               "loss_level"]].median()
    out["sweep_table"] = tab.round(3).reset_index().to_dict("records")
    out["sweep_rank_corr"] = {}
    for ref in ("n_unstable", "n_unstable_97", "traj_PR"):
        out["sweep_rank_corr"][ref] = {s: float(spearmanr(lvl[ref], lvl[s])[0]) for s in STAT}
    # window-level, pooled across all sweep runs after burn-in
    out["sweep_window_corr"] = {ref: {s: float(spearmanr(sw[ref], sw[s], nan_policy="omit")[0])
                                      for s in STAT}
                                for ref in ("n_unstable", "traj_PR")}
    # within-run: tracks progressive sharpening in time?
    allsw = w[w.stage == "sweep"]
    out["within_run_rho"] = {s: within_run_rho(allsw, "n_unstable", s) for s in STAT}
    out["within_run_rho_trajPR"] = {s: within_run_rho(allsw, "traj_PR", s) for s in STAT}

    # ---------------- observer control: smoothing the log, same trajectory
    out["smoothing"] = {
        "MG_smooth/MG": float((sw.MG_smooth / sw.MG).median()),
        "MG_surr_smooth/MG_surr": float((sw.MG_surr_smooth / sw.MG_surr).median()),
        "crossings_smooth/crossings": float((sw.crossings_smooth / sw.crossings).median()),
        "spec_entropy_smooth/spec_entropy": float((sw.spec_entropy_smooth / sw.spec_entropy).median()),
        "rises_smooth/rises": float((sw.rises_smooth / sw.rises).median()),
        "rank_corr_n_unstable_MG_smooth": float(spearmanr(lvl.n_unstable,
                                                          lvl.MG_smooth)[0]),
        "rank_corr_n_unstable_crossings_smooth": float(spearmanr(lvl.n_unstable,
                                                                 lvl.crossings_smooth)[0]),
    }

    # ---------------- switch: change across the mid-run switch
    half = w.start.max() + 1000
    half = int(round((w.start.max() + 1000) / 2))
    ctl = w[w.stage == "sweep"].copy()
    sws = w[w.stage == "switch"].copy()
    both = pd.concat([sws, ctl[ctl.eta0.isin(sorted(set(sws.eta0) | set(sws.eta1)))]])
    pre = both[(both.start + 1000 <= half) & (both.start >= half - 2000)]
    post = both[both.start >= half + 1000]
    cols = ["n_unstable", "traj_PR", "MG", "MG_surr", "rel", "crossings", "rises",
            "spec_entropy", "MG_smooth", "crossings_smooth"]
    r = (post.groupby(["eta0", "eta1", "seed"])[cols].median()
         / pre.groupby(["eta0", "eta1", "seed"])[cols].median())
    d = (post.groupby(["eta0", "eta1", "seed"])[cols].median()
         - pre.groupby(["eta0", "eta1", "seed"])[cols].median())
    out["switch_ratio"] = r.groupby(["eta0", "eta1"]).median().round(3).reset_index().to_dict("records")
    out["switch_diff_nunst"] = d.groupby(["eta0", "eta1"]).n_unstable.median().round(2).reset_index().to_dict("records")
    rr = r.reset_index()
    rr["dir"] = np.sign(np.log(rr.eta1 / rr.eta0))
    out["switch_sign_agreement"] = {
        s: float(np.mean(np.sign(np.log(rr[s])) == np.sign(np.log(rr.traj_PR))))
        for s in ["MG", "MG_surr", "crossings", "spec_entropy", "MG_smooth", "crossings_smooth"]}

    # ---------------- cost
    m = pd.DataFrame([x for x in meta if not x.get("diverged")])
    out["cost"] = {
        "t_MG_window_ms": float(w.t_MG.median() * 1e3),
        "t_trajPR_window_ms": float(w.t_PR.median() * 1e3),
        "t_lanczos_checkpoint_ms": float(ck.t_lanczos.median() * 1e3),
        "hvp_per_checkpoint": float(ck.n_hvp.median()),
        "t_train_run_s": float(m.t_train.median()),
        "t_lanczos_run_s": float(m.t_lanczos.median()),
        "traj_MB": float(m.traj_MB.median()), "log_KB": float(m.log_KB.median()),
        "P": int(m.P.iloc[0]),
    }
    json.dump(out, open(RES / "summary.json", "w"), indent=1, default=float)

    figures(w, ck, lvl, r.reset_index(), half)
    print(json.dumps(out, indent=1, default=float)[:6000])


def figures(w, ck, lvl, rr, half):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    etas = sorted(lvl.eta0.unique())
    # Fig 1: level vs eta
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
    for a, (col, lab, c) in zip(ax, [("n_unstable", "Hessian modes at 2/η\n(Lanczos, expensive)", GREY),
                                     ("traj_PR", "trajectory PR\n(all weights, expensive)", GREY),
                                     ("MG", "MG of loss log\n(cheap)", BLUE),
                                     ("crossings", "trend crossings\n(cheap, spectral)", GOLD)]):
        g = lvl.groupby("eta0")[col]
        a.errorbar(etas, g.median(), yerr=[g.median() - g.min(), g.max() - g.median()],
                   fmt="o-", color=c, capsize=2)
        a.set_xlabel("step size η"); a.set_title(lab, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "sweep_levels.png", dpi=200); plt.close(fig)

    # Fig 2: MG vs n_unstable scatter per run + smoothing control
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    ax[0].scatter(lvl.n_unstable, lvl.MG, c=BLUE, s=18, label="raw log")
    ax[0].scatter(lvl.n_unstable, lvl.MG_smooth, c=BLUE, s=18, marker="x", label="smoothed log")
    ax[0].set_xlabel("Hessian modes at 2/η"); ax[0].set_ylabel("MG"); ax[0].legend(frameon=False)
    ax[1].scatter(lvl.n_unstable, lvl.crossings, c=GOLD, s=18, label="raw log")
    ax[1].scatter(lvl.n_unstable, lvl.crossings_smooth, c=GOLD, s=18, marker="x", label="smoothed log")
    ax[1].set_xlabel("Hessian modes at 2/η"); ax[1].set_ylabel("trend crossings"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "mg_vs_modes.png", dpi=200); plt.close(fig)

    # Fig 3: time courses for one seed per eta
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(etas):
        g = w[(w.stage == "sweep") & (w.eta0 == e) & (w.seed == 0)]
        c = cmap(i / max(1, len(etas) - 1))
        ax[0].plot(g.centre, g.n_unstable, color=c, label=f"η={e:g}")
        ax[1].plot(g.centre, g.traj_PR, color=c)
        ax[2].plot(g.centre, g.MG, color=c)
    ax[0].set_ylabel("modes at 2/η"); ax[1].set_ylabel("trajectory PR"); ax[2].set_ylabel("MG (loss)")
    ax[2].set_xlabel("step (window centre)"); ax[0].legend(ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "time_courses.png", dpi=200); plt.close(fig)

    # Fig 4: switch runs
    sw = w[w.stage == "switch"]
    pairs = sorted(set(zip(sw.eta0, sw.eta1)))
    fig, ax = plt.subplots(3, len(pairs), figsize=(2.6 * len(pairs), 5.5), sharex=True)
    ax = np.atleast_2d(ax).reshape(3, -1)
    for j, (e0, e1) in enumerate(pairs):
        for s, g in sw[(sw.eta0 == e0) & (sw.eta1 == e1)].groupby("seed"):
            ax[0, j].plot(g.centre, g.n_unstable, color=GREY, alpha=.6)
            ax[1, j].plot(g.centre, g.traj_PR, color=GREY, alpha=.6)
            ax[2, j].plot(g.centre, g.MG, color=BLUE, alpha=.6)
        for i in range(3):
            ax[i, j].axvline(half, color=ROSE, ls="--", lw=1)
        ax[0, j].set_title(f"η {e0:g} → {e1:g}", fontsize=9)
    ax[0, 0].set_ylabel("modes at 2/η"); ax[1, 0].set_ylabel("trajectory PR"); ax[2, 0].set_ylabel("MG (loss)")
    fig.tight_layout(); fig.savefig(FIG / "switch.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
