"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiment_runner.plotting import CODE, BAND, FAINT, POINTER, RECURRENT, STOCHASTIC, TRANSIENT, bounds
from experiment_runner.plotting import draw


def fig_regimes():
    """Recovery, and whether it can be told without ground truth, on the image-data system."""
    d = pd.read_csv(HERE / 'logs/sweep_raw.csv')
    d = d[(~d.eta_zero) & (~d.observer.isin(["acc_probe", "loss_step"]))]

    series = [
        ("qp", "recurrent, fast", RECURRENT, "-", "o"),
        ("qp_slow", "recurrent, slow", RECURRENT, "--", "s"),
        ("mixed", "recurrent + noise", STOCHASTIC, "--", "^"),
        ("noise", "stochastic", STOCHASTIC, "-", "v"),
        ("gd", "transient", TRANSIENT, ":", "D"),
    ]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.66),
                                 gridspec_kw={"width_ratios": [1.18, 1.0]})

    # two reference lines, each named by a six-character label at its own end:
    # E_max is not a bound (no clamping is applied) and the diagonal is the truth
    ax.axhline(20, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    ax.text(9.6, 20, r"$E_{\max}$", va="center", ha="left", **POINTER)
    ax.plot([0.9, 8.6], [0.9, 8.6], color=FAINT, lw=0.9, ls=(0, (3.5, 2.5)),
            zorder=1)
    ax.text(9.6, 8.6, "truth", va="center", ha="left", **POINTER)

    ident = {}
    for arm, label, c, ls, mk in series:
        g = d[d.arm == arm]
        if g.empty:
            continue
        t = g.groupby("r").traj_PR.median().values
        y = g.groupby("r").MG.median().values
        # interquartile band over the forty values behind each point (four seeds
        # x ten observers). Sorted in x only so that the transient, whose seven
        # points all sit at x ~ 1.05 in no particular order, fills a blob rather
        # than a self-intersecting polygon; the other arms are already monotone.
        q1 = g.groupby("r").MG.quantile(0.25).values
        q3 = g.groupby("r").MG.quantile(0.75).values
        o = np.argsort(t)
        ax.fill_between(t[o], q1[o], q3[o], color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(t, y, ls, color=c, marker=mk, ms=2.8, mec="white", mew=0.5,
                lw=1.1, label=label, zorder=4, clip_on=False)
        # rho_ident is populated for seed 0, three observers and r in {2, 6} only,
        # so panel (b) plots the six raw values rather than an aggregate.
        ident[arm] = g.dropna(subset=["ident_ratio"])[["observer", "r",
                                                       "ident_ratio"]]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 9.2)
    ax.set_ylim(0.6, 42)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlabel("measured effective rank")
    # The paper's own symbol, not "estimated dimension". Two reasons: the phrase runs
    # off the left edge at 1.66 in of panel height, and this panel plots the transient
    # and mini-batch arms, where section 3.3 says the value is a statistic of the window
    # and not a dimension at all. fig_tau keeps the phrase because every case in it is
    # recurrent.
    ax.set_ylabel(r"$\hat{d}_{\mathrm{MG}}$")
    ax.set_title("(a) the estimate against truth", loc="left")

    bx.axvspan(0.95, 1.10, color=BAND, alpha=0.08, lw=0, zorder=0)
    # Two sub-rows per arm, one per rank, because the rank is what connects this
    # panel to (a): the slow recurrent arm tracks the truth at r = 2 and saturates
    # by r = 6, and the diagnostic has to be seen changing across the same step or
    # the panel only repeats what the regime labels already say. Fill encodes the
    # rank (open r = 2, solid r = 6); the observers stay separated within a sub-row
    # so that coincident values remain countable.
    RANK_DY = {2: 0.18, 6: -0.18}
    OBS_DY = {"c_proj1": 0.07, "g_fro": 0.0, "w_fro": -0.07}
    for i, (arm, label, c, ls, mk) in enumerate(series):
        v = ident.get(arm)
        if v is None or v.empty:
            continue
        y = len(series) - 1 - i
        for rank, dy in RANK_DY.items():
            w = v[v.r == rank]
            if w.empty:
                continue
            bx.plot([w.ident_ratio.min(), w.ident_ratio.max()], [y + dy] * 2, "-",
                    color=c, lw=0.6, alpha=0.40, zorder=2)
            bx.plot(w.ident_ratio.values, y + dy + w.observer.map(OBS_DY).values,
                    mk, color=c, ms=3.0, mec=c if rank == 2 else "white",
                    mfc="white" if rank == 2 else c, mew=0.6, zorder=3,
                    clip_on=False)
    bx.set_yticks(range(len(series)))
    bx.set_yticklabels([lbl for _, lbl, _, _, _ in series][::-1])
    bx.set_ylim(-0.65, len(series) - 0.02)
    # The band has to stay in frame whatever the ratios do: the panel is about which
    # regimes fall inside it, and a frame that cut it off would answer the question by
    # cropping. Everything else follows the data.
    ratios = pd.concat(ident.values()).ident_ratio if ident else pd.Series([1.0])
    low, high = bounds(ratios, pad=0.05, step=0.1, include=[0.95, 1.10])
    bx.set_xlim(low, high)
    bx.set_xticks(np.arange(low, high + 1e-9, 0.2))
    bx.set_xlabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")
    bx.set_title("(b) the same cases", loc="left")
    bx.text(1.025, 4.42, "accurate cases", color=BAND, fontsize=7.0, ha="center",
            va="bottom")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=2.8, mec="white",
                      mew=0.5, label=lbl) for _, lbl, c, ls, mk in series]
    fig.tight_layout(rect=[0, 0.115, 1, 1], w_pad=1.4)
    # Five entries across 5.5 in is the widest legend in the article, and at 9 pt type it
    # is the one that runs out of room first. The handles and the gaps carry the reduction
    # rather than the type, which has to stay level with the other eleven figures.
    fig.legend(handles=handles, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=1.5,
               handletextpad=0.35, columnspacing=0.75)
    return fig


def fig_signal():
    """Four logs that look alike, and the four levels the estimator reads off them."""
    series = pd.read_csv(CODE / 'synthetic/delay_embedding_and_limits/delay_embedding/logs/curve_series.csv')
    windows = pd.read_csv(CODE / 'synthetic/delay_embedding_and_limits/delay_embedding/logs/curve_windows.csv')
    ranks = sorted(series.r.unique())
    marks = ["o", "s", "^", "D"]
    # The four curves belong to one dynamical regime, but they represent different
    # ranks. Give the ranks distinct, colour-blind-safe hues as well as markers; the
    # marker remains the primary encoding in print and in monochrome copies.
    rank_colours = ["#332288", "#117733", "#44AA99", "#CC6677"]

    # 400 samples, not the whole record: the drive fills one octave of a period near
    # sixteen samples, so ten thousand of them at an inch and a quarter wide is a block of
    # ink. The point of the row is that the four look alike, which needs them legible.
    span = 400

    fig = plt.figure(figsize=(5.5, 2.60))
    grid = fig.add_gridspec(2, 2 * len(ranks), height_ratios=[0.92, 1.0])

    for column, rank in enumerate(ranks):
        ax = fig.add_subplot(grid[0, 2 * column:2 * column + 2])
        one = series[(series.r == rank) & (series["sample"] < span)]
        ax.plot(one["sample"].to_numpy(), one.z.to_numpy(), "-",
                color=rank_colours[column], lw=0.6)
        ax.set_title(f"$r = {rank}$", pad=2.5)
        ax.set_ylim(-3.0, 3.0)
        ax.set_yticks([-2, 0, 2])
        ax.set_xticks([0, 200, 400])
        if column:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("standardised")
        ax.set_xlabel("sample", labelpad=1.5)

    # The lower panels carry the same four series; only the quantity differs. The estimator
    # separates the ranks and the roughness does not, which is the whole of the smoothness
    # control and is otherwise a yes/no column in a table. Rank is the marker, because
    # every one of these is the same regime and so the same colour.
    ex = fig.add_subplot(grid[1, :len(ranks)])
    rx = fig.add_subplot(grid[1, len(ranks):])
    handles = []
    for rank, mark, colour in zip(ranks, marks, rank_colours):
        one = windows[windows.r == rank].sort_values("centre")
        x = one.centre.to_numpy() / 1000.0
        for ax, column in ((ex, "MG"), (rx, "roughness")):
            ax.plot(x, one[column].to_numpy(), "-", color=colour, lw=0.9,
                    marker=mark, ms=2.6, markevery=5, mec="white", mew=0.3)
        handles.append(Line2D([], [], color=colour, lw=0.9, marker=mark, ms=2.6,
                              mec="white", mew=0.3, label=f"$r = {rank}$"))
        ex.axhline(float(one.truth.iloc[0]), color=FAINT, lw=0.7, ls=(0, (2, 2.5)),
                   zorder=0)

    ex.set_ylabel("components")
    rx.set_ylabel("roughness")
    ex.set_title("(a) the estimate", loc="left", pad=2.5)
    rx.set_title("(b) the roughness", loc="left", pad=2.5)
    ex.set_ylim(0.0, max(ranks) + 1.2)
    rx.set_ylim(*bounds(windows.roughness, pad=0.35))
    for ax in (ex, rx):
        ax.set_xlim(3.85, 6.15)
        ax.set_xticks([4, 5, 6])
        ax.set_xlabel("window centre ($10^3$)", labelpad=1.5)

    fig.tight_layout(rect=[0, 0.115, 1, 1], h_pad=0.9, w_pad=1.4)
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=1.8, columnspacing=1.6)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_regimes, fig_signal,))
