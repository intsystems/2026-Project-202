"""Regenerate the article's figures from the committed result files.

    python make_figures.py

Writes PDF (for LaTeX) and PNG (for preview) into figures/.

Design notes, so that later edits do not undo them:

* Every figure is drawn at 5.5 in, the exact \\textwidth of the ICOMP style, with
  8 pt type, so LaTeX never rescales it and the type stays at 8 pt on the page.
* Colour encodes the paper's three dynamical regimes and nothing else:
  recurrent, stochastic, transient. Line style distinguishes variants inside a
  regime and marker shape distinguishes experimental settings, so identity is
  never carried by colour alone.
* The palette passes the categorical checks (OKLCH lightness band, chroma floor,
  Machado severity-1.0 protan and deutan separation on all pairs, normal-vision
  floor, WCAG contrast against the page) with no warnings.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical palette, in fixed order: one hue per dynamical regime.
RECURRENT = "#0072B2"
STOCHASTIC = "#D55E00"
TRANSIENT = "#5D3A9B"
GREY = "#666666"
BAND = "#0072B2"

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#999999",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
})


def save(fig, stem):
    # No bbox="tight": it expands the canvas past 5.5 in when a legend is wider
    # than the axes, and LaTeX then scales the figure down and the type with it.
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print("wrote", stem)


# ------------------------------------------------------------------ figure 1
def fig_regimes():
    """Recovery and admissibility on the image-data system of section 5.4."""
    d = pd.read_csv(CODE / "active_dimension/results/e2_rank_sweep/sweep_raw.csv")
    d = d[(~d.eta_zero) & (~d.observer.isin(["acc_probe", "loss_step"]))]

    series = [
        ("qp", "recurrent, fast drive", RECURRENT, "-", "o"),
        ("qp_slow", "recurrent, slow drive", RECURRENT, "--", "s"),
        ("mixed", "recurrent $+$ weak noise", STOCHASTIC, "--", "^"),
        ("noise", "stochastic", STOCHASTIC, "-", "v"),
        ("gd", "transient", TRANSIENT, ":", "D"),
    ]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.10),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax.axhline(20, color=GREY, lw=0.8, ls=(0, (1, 2)), zorder=0)
    # a reference line, not a bound: no clamping is applied and a violated model
    # returns values above E. Placed away from x = 1, where the transient piles up.
    ax.text(2.3, 20.8, r"$E_{\max}$, not a bound", color=GREY, fontsize=6.5,
            ha="left", va="bottom")
    ax.plot([0.9, 8.3], [0.9, 8.3], color=GREY, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(8.6, 7.6, "truth", color=GREY, fontsize=6.5, ha="left", va="center")

    ident = {}
    for arm, label, c, ls, mk in series:
        g = d[d.arm == arm]
        if g.empty:
            continue
        t = g.groupby("r").traj_PR.median().values
        y = g.groupby("r").MG.median().values
        ax.plot(t, y, ls, color=c, marker=mk, ms=3.2, mec="white", mew=0.5,
                label=label, zorder=4, clip_on=False)
        ident[arm] = g.groupby("r").ident_ratio.median()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 9.5)
    ax.set_ylim(0.6, 42)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("measured active dimension")
    ax.set_ylabel("estimated dimension")
    ax.set_title("(a) what the estimator returns", loc="left", pad=6)

    bx.axvspan(0.95, 1.10, color=BAND, alpha=0.10, lw=0, zorder=0)
    for i, (arm, label, c, ls, mk) in enumerate(series):
        v = ident.get(arm)
        if v is None:
            continue
        y = len(series) - 1 - i
        bx.plot(v.values, np.full(len(v), y), mk, color=c, ms=4.2,
                mec="white", mew=0.6, zorder=3, clip_on=False)
    bx.set_yticks(range(len(series)))
    bx.set_yticklabels([lbl for _, lbl, _, _, _ in series][::-1])
    bx.set_ylim(-0.55, len(series) - 0.35)
    bx.set_xlim(0.92, 1.70)
    bx.set_xticks([1.0, 1.2, 1.4, 1.6])
    bx.set_xlabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")
    bx.set_title("(b) is it a count?", loc="left", pad=6)
    bx.text(1.025, 4.42, "admissible", color=BAND, fontsize=6.5, ha="center")
    # inside the axes, on the empty right of the transient row: outside it the
    # text ran into the spine and the x label
    bx.annotate("passes, yet returns 29\nwhere the measurement is 1",
                xy=(1.06, 0.0), xytext=(1.135, 0.0), fontsize=6.0,
                color=TRANSIENT, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=TRANSIENT, lw=0.9,
                                shrinkA=2, shrinkB=2))
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3.2, mec="white",
                      mew=0.5, label=lbl) for _, lbl, c, ls, mk in series]
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2, columnspacing=1.2)
    save(fig, "fig_regimes")


# ------------------------------------------------------------------ figure 2
def fig_dip():
    """The trajectory-rank collapse at generalisation, aligned on t_gen."""
    d = pd.read_csv(CODE / "active_rank/results_fine/rank_windows.csv")
    meta = {m["run"]: m for m in
            json.loads((CODE / "active_rank/results_fine/rank_milestones.json").read_text())}

    groks = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
    ctrls = ["mod_wd0", "s5_wd0"]
    panels = [("fn_PR_pos_det", "(a) function space", None),
              ("PR_pos_det", "(b) parameter space", None),
              ("move", "(c) displacement", "log")]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), sharex=True)
    grid = np.arange(-5000, 5200, 100)

    for ax, (col, title, scale) in zip(axes, panels):
        stack = []
        for r in groks:
            g = d[d.run == r].sort_values("right_step")
            x = 0.5 * (g.right_step + g.left_step) - meta[r]["t_gen"]
            ax.plot(x, g[col], color=RECURRENT, lw=0.7, alpha=0.30)
            stack.append(np.interp(grid, x, g[col], left=np.nan, right=np.nan))
        ax.plot(grid, np.nanmean(np.vstack(stack), axis=0), color=RECURRENT,
                lw=2.2, label="generalises (4 runs)", zorder=5)

        for r, ls in zip(ctrls, ["--", (0, (1, 1.6))]):
            g = d[d.run == r].sort_values("right_step")
            ref = meta["mod_wd1" if r.startswith("mod") else "s5_wd1"]["t_gen"]
            ax.plot(0.5 * (g.right_step + g.left_step) - ref, g[col], linestyle=ls,
                    color=STOCHASTIC, lw=1.2,
                    label="no weight decay (2 runs)" if r == "mod_wd0" else None)

        ax.axvline(0, color=GREY, lw=0.9, ls=(0, (2, 2)))
        if scale:
            ax.set_yscale(scale)
        ax.set_xlim(-5000, 5000)
        ax.set_xticks([-5000, 0, 5000])
        ax.set_xticklabels(["-5k", "$t_{gen}$", "5k"])
        ax.set_title(title, loc="left", pad=4)

    axes[0].set_ylabel("participation ratio")
    axes[2].set_ylabel("displacement")
    axes[1].set_xlabel("steps since generalisation")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2)
    save(fig, "fig_dip")


# ------------------------------------------------------------------ figure 3
def fig_map():
    """Where the training logs of section 7 fall in the admissibility plane."""
    tr = pd.read_csv(CODE / "active_dimension/results/e5_real_logs/real_logs_summary.csv")
    tr = tr[tr.column == "weight_norm"]
    gr = pd.concat([
        pd.read_csv(CODE / "gromov_arithmetic/results/arith/dimension_probe_summary.csv"),
        pd.read_csv(CODE / "gromov_polynomials/results/dimension_probe_summary.csv")])
    gr = gr[gr.column == "train_loss"]

    fig, ax = plt.subplots(figsize=(5.5, 2.35))

    # one highlighted target zone: stable in E and recurrent enough to embed
    ax.add_patch(mpl.patches.Rectangle(
        (8.0, 0.95), 500 - 8.0, 0.15, facecolor=BAND, alpha=0.10,
        edgecolor=BAND, lw=1.0, ls=(0, (4, 3)), zorder=0))
    ax.axvline(8.0, color=GREY, lw=0.8, ls=(0, (2, 3)), zorder=0)
    ax.axhline(1.10, color=GREY, lw=0.8, ls=(0, (2, 3)), zorder=0)

    hi = tr[tr.ident > 1.15]
    lo = tr[tr.ident <= 1.15]
    ax.plot(gr.osc, gr.ident, "s", mfc="none", mec=TRANSIENT, mew=1.2, ms=9,
            ls="none", label="perceptron, full batch", zorder=3)
    ax.plot(hi.osc, hi.ident, "o", color=STOCHASTIC, ms=5.5, mec="white", mew=0.8,
            ls="none", label="transformer, mini-batch", zorder=4)
    ax.plot(lo.osc, lo.ident, "o", color=TRANSIENT, ms=5.5, mec="white", mew=0.8,
            ls="none", label="transformer, no weight decay", zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(0.7, 500)
    ax.set_ylim(0.88, 1.74)
    ax.set_xlabel("recurrences per window")
    ax.set_ylabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")

    ax.text(90, 1.025, "admissible: no run is here", color=BAND,
            fontsize=7, ha="center", va="center")
    ax.text(12, 1.63, "stochastic", color=STOCHASTIC, fontsize=7,
            ha="left", va="center")
    ax.text(12, 1.555, "no invariant set exists", color=STOCHASTIC,
            fontsize=6.5, ha="left", va="center")
    ax.text(1.05, 1.34, "transient", color=TRANSIENT, fontsize=7,
            ha="left", va="center")
    ax.text(1.05, 1.245, "monotone, so stable\nfor the wrong reason",
            color=TRANSIENT, fontsize=6.5, ha="left", va="center")
    ax.text(7.0, 1.70, "too few recurrences", color=GREY, fontsize=6.5,
            ha="right", va="top")
    ax.text(440, 1.135, "unstable in $E$", color=GREY, fontsize=6.5,
            ha="right", va="bottom")

    fig.tight_layout(rect=[0, 0.16, 1, 1])
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.44), handletextpad=0.3, columnspacing=0.9)
    save(fig, "fig_map")


# ------------------------------------------------------------------ figure 4
def fig_pairs():
    """The label-matched pairs, and the training loss that explains them."""
    mg = pd.concat([
        pd.read_csv(CODE / "gromov_arithmetic/results/arith/dimension_probe_summary.csv"),
        pd.read_csv(CODE / "gromov_polynomials/results/dimension_probe_summary.csv")])
    mg = mg[mg.column == "train_loss"].set_index("run").MG

    pairs = [("g_p1_p97", "g_p1x_p97", r"$(4n_1{+}n_2^2)^3$"),
             ("g_p2_p97", "g_p2x_p97", r"$(2n_1{+}3n_2)^4$"),
             ("g_p3_p97", "g_p3x_p97", r"$(5n_1^3{+}2n_2^4)^2$"),
             ("a_add", "x_no_grok", r"$n{+}m$")]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.35),
                                 gridspec_kw={"width_ratios": [1.0, 1.05]})

    for i, (good, bad, name) in enumerate(pairs):
        y = len(pairs) - 1 - i
        ax.plot([mg[good], mg[bad]], [y, y], color=GREY, lw=1.0, zorder=1)
        ax.plot(mg[good], y, "o", color=RECURRENT, ms=5.5, mec="white", mew=0.8,
                zorder=3, label="generalises" if i == 0 else None)
        ax.plot(mg[bad], y, "o", color=STOCHASTIC, ms=5.5, mec="white", mew=0.8,
                zorder=3, label="does not" if i == 0 else None)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([n for _, _, n in pairs][::-1])
    ax.set_ylim(-0.7, len(pairs) - 0.3)
    ax.set_xlim(18.5, 23.6)
    ax.set_xlabel("estimate on the training loss")
    ax.set_title("(a) four label-matched pairs", loc="left", pad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


    for key, colour, ls, lbl in [
            ("g_p1_p97", RECURRENT, "-", "generalises"),
            ("g_p1x_p97", STOCHASTIC, "--", "does not")]:
        t = pd.read_csv(CODE / f"gromov_polynomials/results/{key}_train.csv")
        bx.plot(t.step, t.train_loss, ls, color=colour, lw=1.3, label=lbl)
    bx.set_yscale("log")
    bx.set_xlim(0, 100000)
    bx.set_xticks([0, 50000, 100000])
    bx.set_xticklabels(["0", "50k", "100k"])
    bx.set_xlabel("step")
    bx.set_ylabel("training loss")
    # panel (b) plots g_p1, which is the top row of panel (a)
    bx.set_title("(b) the top pair, training loss",
                 loc="left", pad=6)
    h, l = bx.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2)
    save(fig, "fig_pairs")


if __name__ == "__main__":
    fig_regimes()
    fig_dip()
    fig_map()
    fig_pairs()
