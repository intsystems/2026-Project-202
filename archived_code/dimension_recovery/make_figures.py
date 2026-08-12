"""Two figures for report_0808.md.

    fig1  the lowdata15 trace against a generalising run, with the running-maximum
          reference drawn, so the trigger can be read off rather than described
    fig2  MG's systematic offset against every nuisance parameter, which is the
          evidence separating "responds to k" from "measures k"

    python make_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension as mle                     # noqa: E402

PRED = HERE.parent / "prediction_improved" / "results"
CONF = HERE.parent / "edm_validation" / "results" / "conf"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
WIN, LOG_EVERY = 600, 10


def first_sustained(steps, acc, thr=0.95, w=5):
    a = pd.Series(acc).rolling(w, min_periods=1, center=True).mean().to_numpy()
    ok = a >= thr
    i = None
    for j in range(len(ok) - 1, -1, -1):
        if not ok[j]:
            break
        i = j
    return int(steps[i]) if i is not None else None


def trace(path):
    d = pd.read_csv(path)
    s, wn = d["step"].to_numpy(), d["weight_norm"].to_numpy()
    tm = first_sustained(s, d["train_acc"].to_numpy())
    tg = first_sustained(s, d["val_acc"].to_numpy())
    w = WIN // LOG_EVERY
    right = np.array([s[a + w - 1] for a in range(len(wn) - w + 1)])
    dims = np.array([mle(wn[a:a + w], tau=1, max_E=15, k_neighbors=5,
                         correction="mackay_ghahramani", theiler_window=0,
                         rng=np.random.default_rng(0))
                     for a in range(len(wn) - w + 1)])
    return right, dims, tm, tg


def _panel(ax, path, title, span=None, mark_gen=False):
    """Trace since t_mem, with the running maximum and the 30 % trigger level.

    `span` truncates to a fixed number of steps after t_mem so that two runs with very
    different plateau lengths are drawn on the same time axis. Without it the shorter
    run looks smooth purely because it is stretched over a tenth of the width, which is
    an artefact of the plot and not of the data.
    """
    r, d, tm, tg = trace(path)
    stop = tg if (mark_gen and tg) else r.max()
    m = (r - WIN + 1 >= tm) & (r <= stop)
    rr, dd = r[m], d[m]
    if span is not None:
        keep = rr <= rr[0] + span
        rr, dd = rr[keep], dd[keep]
    peak = np.maximum.accumulate(dd)
    ax.plot(rr - rr[0], dd, lw=1.1, color="#2b6cb0", label="MG on $\\|w\\|_2$")
    ax.plot(rr - rr[0], peak, lw=1.0, ls="--", color="#c05621", label="running maximum")
    ax.plot(rr - rr[0], 0.7 * peak, lw=1.0, ls=":", color="#9b2c2c",
            label="trigger (30 % below)")
    fired = np.flatnonzero(dd <= 0.7 * peak)
    if len(fired):
        ax.axvline(rr[fired[0]] - rr[0], color="#9b2c2c", lw=1.2, alpha=0.8)
        ax.annotate("fires", (rr[fired[0]] - rr[0], dd.min()), xytext=(4, 2),
                    textcoords="offset points", color="#9b2c2c", fontsize=8)
    if mark_gen and tg:
        ax.axvline(tg - rr[0], color="#276749", lw=1.4)
        ax.annotate("$t_{gen}$", (tg - rr[0], dd.max()), xytext=(-30, -6),
                    textcoords="offset points", color="#276749", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("steps since $t_{mem}$")
    ax.grid(alpha=0.25)
    return dd


def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    span = 1200
    a = _panel(axes[0], PRED / "lowdata15_train.csv",
               f"lowdata15, first {span} steps after $t_{{mem}}$\n(never generalises)",
               span=span)
    b = _panel(axes[1], CONF / "s3_i1.csv",
               "conf_s3_i1, whole plateau (1160 steps)\n(generalises)", mark_gen=True)
    c = _panel(axes[2], PRED / "lowdata15_train.csv",
               "lowdata15, all 18600 post-memorisation steps\n(same run as the left panel)")
    axes[0].set_ylabel("MG intrinsic-dimension estimate")
    axes[1].legend(fontsize=8, loc="lower left", framealpha=0.9)
    for ax in axes[:2]:
        ax.set_ylim(1.2, 2.3)
    fig.suptitle("At matched duration the two traces are not distinguishable by shape; "
                 "the rule fires on both.\n"
                 "The right panel is the same control over its full budget: the level "
                 "does not fall (Theil-Sen $-0.003$ per 1000 steps).", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_lowdata15_vs_conf.png", dpi=160)
    print(f"wrote {FIG / 'fig1_lowdata15_vs_conf.png'}")
    print(f"  matched-span sd: lowdata15 {a.std():.3f}   conf_s3_i1 {b.std():.3f}"
          f"   lowdata15 full {c.std():.3f}")


def fig2():
    f = pd.read_csv(HERE / "results" / "exp6_ofat.csv")
    factors = ["cycles", "window", "tau", "max_E", "amp", "kn", "snr", "n"]
    metrics = ["MG", "TwoNN", "LB", "PR"]
    colours = {"MG": "#2b6cb0", "TwoNN": "#276749", "LB": "#c05621", "PR": "#805ad5"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, fac in zip(axes.ravel(), factors):
        sub = f[f.factor == fac]
        lv = sorted(sub.level.unique())
        for m in metrics:
            y = [sub[(sub.metric == m) & (sub.level == l)].bias.iloc[0] for l in lv]
            ax.plot(range(len(lv)), y, "o-", ms=4, lw=1.4, color=colours[m], label=m)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-0.5, 0.5, color="0.85", zorder=0)
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([f"{l:g}" for l in lv], fontsize=8)
        rng = sub[sub.metric == "MG"].bias
        ax.set_title(f"{fac}   (MG range {rng.max() - rng.min():.2f})", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("systematic offset  mean($\\hat d - k$)")
    axes[1, 0].set_ylabel("systematic offset  mean($\\hat d - k$)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Absolute calibration is not stable: the offset moves with parameters "
                 "nobody can fix on a training log\n"
                 "(grey band = within half a component of the truth)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_bias_vs_parameters.png", dpi=160)
    print(f"wrote {FIG / 'fig2_bias_vs_parameters.png'}")


if __name__ == "__main__":
    fig1()
    fig2()
