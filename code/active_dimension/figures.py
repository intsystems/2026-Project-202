"""Figures.  Run after the experiments; each panel degrades to a note if its CSV is absent.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)

C = {"qp": "#1f77b4", "noise": "#d62728", "batch": "#ff7f0e", "batch_proj": "#9467bd",
     "mixed": "#2ca02c", "gd": "#8c564b", "ou": "#d62728", "colored": "#ff7f0e",
     "truth": "#444444", "surr": "#999999"}


def _load(p):
    p = RES / p
    return pd.read_csv(p) if p.exists() else None


def fig1_atlas():
    """Is r identifiable at all?  Three generator classes, one scalar observer."""
    df = _load("e0_atlas/atlas_raw.csv")
    if df is None:
        return
    d = df[(df.max_E == 20) & (df.observer == "generic")]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for a, fam, ttl in zip(ax, ("qp", "ou", "colored"),
                           ("deterministic torus\n(an r-manifold exists)",
                            "white-driven OU\n(no r-manifold exists)",
                            "band-limited noise\n(no r-manifold exists)")):
        s = d[d.family == fam]
        for key, g in s.groupby("N" if fam == "qp" else "tau_c"):
            m = g.groupby("r").MG.median()
            a.plot(m.index, m.values, "o-", ms=4, alpha=.85,
                   label=("N=%d" % key) if fam == "qp" else ("tau_c=%g" % key))
        rr = np.arange(1, 9)
        a.plot(rr, rr, "--", c=C["truth"], lw=1, label="truth")
        a.set_title(ttl, fontsize=9)
        a.set_xlabel("active dimension r")
        a.legend(fontsize=6, frameon=False)
        a.grid(alpha=.25)
    ax[0].set_ylabel("MG (max_E = 20)")
    fig.suptitle("E0 -- MG against the true active dimension, by generator class", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_atlas.png", dpi=150)
    plt.close(fig)


def fig2_sweep():
    """The main sweep: MG against the MEASURED active dimension, per arm."""
    df = _load("e2_rank_sweep/sweep_raw.csv")
    if df is None:
        return
    df = df[~df.flat]
    arms = [a for a in ("qp", "qp_slow", "mixed", "noise", "batch_proj", "batch", "gd")
            if a in set(df.arm)]
    obs = ["w_fro", "c_proj1", "loss_probe", "g_fro"]
    fig, ax = plt.subplots(len(obs), len(arms), figsize=(1.95 * len(arms), 2.0 * len(obs)),
                           sharex=True, sharey=True, squeeze=False)
    for i, o in enumerate(obs):
        for j, arm in enumerate(arms):
            a = ax[i][j]
            s = df[(df.arm == arm) & (df.observer == o)]
            lim = np.arange(0, 10)
            a.plot(lim, lim, ":", c=C["truth"], lw=1.2, zorder=1)
            if len(s):
                g = s.groupby("r").agg(mg=("MG", "median"), lo=("MG", "min"),
                                       hi=("MG", "max"), tr=("traj_PR", "median"))
                a.fill_between(g.tr, g.lo, g.hi, color=C.get(arm, "k"), alpha=.18, lw=0)
                a.plot(g.tr, g.mg, "o-", ms=4, c=C.get(arm, "k"))
            a.set_yscale("symlog", linthresh=10, linscale=0.6)
            a.set_ylim(0, 40)
            a.set_yticks([0, 2, 4, 6, 8, 10, 20, 40])
            if i == 0:
                a.set_title(arm, fontsize=9)
            if j == 0:
                a.set_ylabel(o, fontsize=8)
            if i == len(obs) - 1:
                a.set_xlabel("measured active dim", fontsize=7)
            a.grid(alpha=.25)
            a.tick_params(labelsize=6)
    fig.suptitle("E2 -- MG (band = min..max over 4 seeds) against the MEASURED active "
                 "dimension.  Dotted = truth.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_sweep.png", dpi=150)
    plt.close(fig)


def fig3_controls():
    """Effect size of each negative control against the real effect in E3."""
    d = _load("e4_controls/controls_scored.csv")
    t = _load("e3_transitions/transitions_raw.csv")
    if d is None:
        return
    q = d[d["mode"] == "qp"]
    order = (q.groupby("control").between.median().sort_values().index.tolist())
    fig, a = plt.subplots(figsize=(7.5, 3.6))
    for i, ctl in enumerate(order):
        v = q[q.control == ctl].between.dropna().values
        if not len(v):
            continue
        a.scatter(np.full(len(v), i) + np.random.uniform(-.12, .12, len(v)), v,
                  s=8, alpha=.5, c="#1f77b4")
        a.plot([i - .3, i + .3], [np.median(v)] * 2, c="k", lw=2)
    if t is not None:
        real = (t[t["mode"] == "qp"].level0 - t[t["mode"] == "qp"].level1).abs().median()
        a.axhline(real, color=C["noise"], ls="--",
                  label=f"E3 real change r_high->r_low = {real:.2f}")
        a.legend(fontsize=8, frameon=False)
    a.set_xticks(range(len(order)))
    a.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    a.set_ylabel("|MG shift| vs baseline")
    a.set_title("E4 -- how much MG moves when the number of active directions does not",
                fontsize=10)
    a.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_controls.png", dpi=150)
    plt.close(fig)


def fig5_tau():
    """How far MG moves across its own delay lag, with the system held fixed."""
    d = _load("e6_tau/tau_sensitivity.csv")
    if d is None:
        return
    d = d[d.max_E == 20]
    per = sorted(d.period.unique())
    fig, ax = plt.subplots(1, len(per), figsize=(5.2 * len(per), 3.8), sharey=True)
    ax = np.atleast_1d(ax)
    for a, pv in zip(ax, per):
        s2 = d[d.period == pv]
        for tau, g in s2.groupby("tau"):
            m = g.groupby("r").MG.median()
            a.plot(m.index, m.values, "o-", ms=4, alpha=.85, label=f"tau={tau}")
        m = s2.groupby("r").specPR0.median()
        a.plot(m.index, m.values, "s--", ms=5, c="#111111", lw=1.6, label="spectral PR")
        rr = np.arange(1, 9)
        a.plot(rr, rr, ":", c=C["truth"], lw=1.4, label="truth")
        a.set_title(f"oscillation period = {pv} samples", fontsize=10)
        a.set_xlabel("active dimension r")
        a.set_yscale("symlog", linthresh=10)
        a.grid(alpha=.25)
        a.legend(fontsize=6, frameon=False, ncol=2)
    ax[0].set_ylabel("MG (max_E = 20)")
    fig.suptitle("E6 -- the same torus, seen through different delay lags", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_tau_sensitivity.png", dpi=150)
    plt.close(fig)


def fig4_real():
    """Where the project's own logs sit on the two atlas axes."""
    r = _load("e5_real_logs/real_logs_summary.csv")
    if r is None:
        return
    fig, a = plt.subplots(figsize=(6.4, 4.4))
    for col, g in r.groupby("column"):
        a.scatter(g.roughness, g.ident, s=34, label=col, alpha=.85)
        for _, row in g.iterrows():
            a.annotate(row.run.replace("_s", " s"), (row.roughness, row.ident),
                       fontsize=5, alpha=.6, xytext=(2, 2), textcoords="offset points")
    a.axhspan(0.98, 1.05, color="#2ca02c", alpha=.12)
    a.axhspan(1.15, 1.60, color="#d62728", alpha=.10)
    a.text(a.get_xlim()[0], 1.01, " identifiable (E0 torus)", fontsize=7, color="#2ca02c")
    a.text(a.get_xlim()[0], 1.40, " no dimension exists (E0 OU / coloured)", fontsize=7,
           color="#d62728")
    a.set_xscale("log")
    a.set_xlabel("roughness  std(diff x)/std(x)")
    a.set_ylabel("identifiability ratio  MG(2E)/MG(E)")
    a.set_title("E5 -- the project's 120k-step logs placed on the atlas", fontsize=10)
    a.legend(fontsize=7, frameon=False)
    a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_real_logs.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1_atlas, fig2_sweep, fig3_controls, fig4_real, fig5_tau):
        try:
            f()
            print("ok  ", f.__name__)
        except Exception as e:                       # a missing arm must not kill the rest
            print("skip", f.__name__, "->", type(e).__name__, e)
    print("->", FIG)
