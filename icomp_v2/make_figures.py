"""Generate every figure in the report.

    python make_figures.py            # -> figures/*.pdf and *.png

Palette: Okabe-Ito, the published qualitative palette designed and tested for
colour-vision deficiency. Hues are assigned to entities in a fixed order and never
cycled, so a series keeps its colour across figures.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE / "edm_validation"))

from ccm import ccm_convergence                             # noqa: E402
from forecast import recurrence_profile                     # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

BLUE, VERMILLION, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
ORANGE, SKY, GREY = "#E69F00", "#56B4E9", "#7F7F7F"
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": "#BBBBBB", "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def lorenz_x(n=4000, dt=0.01, s=10.0, r=28.0, b=8 / 3, burn=1000):
    state, out = np.array([1.0, 1.0, 1.0]), []
    for _ in range(n + burn):
        x, y, z = state
        state = state + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        out.append(state[0])
    return np.array(out[burn:])


# --- Figure 1: preconditions -----------------------------------------------

def figure_preconditions():
    grok = pd.read_csv(CODE / "prediction_improved" / "results" / "grok_train.csv")
    s5 = pd.read_csv(CODE / "grokking_analysis" / "grokking_logs"
                     / "grokking_modular_addition_logs_S_5_with_stochastic.csv")

    series = [
        ("Lorenz-63 (attractor)", lorenz_x(), BLUE, "-", 1.6),
        ("monotone ramp (transient)", np.linspace(0, 1, 4000) ** 2, GREY, ":", 1.4),
        (r"grokking: $\|w\|_2$", grok["weight_norm"].to_numpy(), VERMILLION, "-", 1.2),
        (r"grokking: $\mathcal{L}_{train}$", grok["train_loss"].to_numpy(), GREEN, "-", 1.2),
        (r"grokking: $\mathcal{L}_{val}$ ($S_5$)", s5["val_loss"].to_numpy(), PURPLE, "-", 1.2),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    for label, values, colour, style, width in series:
        profile, _ = recurrence_profile(values, E=5, tau=5)
        if not profile:
            continue
        windows = sorted(profile)
        rates = [profile[w] for w in windows]
        norm = [r / rates[0] if rates[0] else np.nan for r in rates]
        ax.plot(np.arange(len(windows)), norm, style, color=colour, lw=width,
                marker="o", ms=3.4, label=label)
        # Label only the two calibration references: the three grokking traces end
        # within 0.05 of each other and their labels would collide without adding
        # information the axis does not already carry.
        if label.startswith(("Lorenz", "monotone")):
            ax.annotate(f"{norm[-1]:.2f}", (len(windows) - 1, norm[-1]),
                        textcoords="offset points", xytext=(7, 0), fontsize=7.2,
                        color=colour, va="center", fontweight="bold")

    ax.set_xticks(range(6))
    ax.set_xticklabels(["0", "span", "N/100", "N/20", "N/10", "N/5"])
    ax.set_xlabel("temporal exclusion window")
    ax.set_ylabel("recurrence rate (relative to no exclusion)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(-0.25, 5.9)
    ax.legend(loc="lower left", ncol=1)
    save(fig, "fig_preconditions")


# --- Figure 2: driver recovery ---------------------------------------------

def figure_driver_recovery():
    a = pd.read_csv(CODE / "edm_validation" / "results" / "phase3_ccm.csv")
    b = pd.read_csv(CODE / "edm_validation" / "results" / "phase5_inject_ccm.csv")
    a = a.dropna(subset=["rho_raw"]).rename(columns={"rho_raw": "rho", "expected": "coupled"})
    b = b.rename(columns={"driver": "run", "expected": "coupled"})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.42})
    panels = [
        (axes[0], a, "(a) ResNet-18 / CIFAR-10", "run"),
        (axes[1], b, "(b) 1-layer transformer / modular addition", "run"),
    ]
    for ax, table, title, key in panels:
        table = table.sort_values(["coupled", "rho"], ascending=[False, True])
        y = np.arange(len(table))
        colours = [BLUE if c else VERMILLION for c in table.coupled]
        ax.barh(y, table.rho, height=0.62, color=colours, edgecolor="white", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(table[key], fontsize=7.3)
        ax.set_xlim(-0.08, 1.06)
        ax.axvline(0, color="#BBBBBB", lw=0.7)
        ax.set_xlabel(r"cross-map skill $\rho$")
        ax.set_title(title, fontsize=8.2, color=INK, pad=6)
        ax.set_axisbelow(True)
        ax.grid(axis="x")
        for yi, (rho, detected, coupled) in enumerate(
                zip(table.rho, table.detected, table.coupled)):
            if detected:
                text = "detected"
            elif coupled:
                text = "missed (periodic)"      # the stated limitation, not a silence
            else:
                text = "silent"
            ax.annotate(text, (max(rho, 0.02), yi), textcoords="offset points",
                        xytext=(4, 0), va="center", fontsize=6.6,
                        color=INK if detected else MUTED)

    handles = [plt.Line2D([], [], color=BLUE, lw=6),
               plt.Line2D([], [], color=VERMILLION, lw=6)]
    fig.legend(handles, ["driver applied to training", "driver logged, never applied"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(left=0.16, right=0.99)
    save(fig, "fig_driver_recovery")


# --- Figure 3: convergence --------------------------------------------------

def figure_convergence():
    base = CODE / "poisoned_batch"
    runs = [
        ("LogisticMap (applied)", base / "folder_for_raw_series"
         / "resnet_cifar_LogisticMap_logs.csv", BLUE, "-"),
        ("Random (applied)", base / "folder_for_raw_series"
         / "resnet_cifar_Random_logs.csv", SKY, "-"),
        ("Ghost Normal (never applied)", base / "ghost_raw_series_logs"
         / "resnet_cifar_Ghost_Normal_logs.csv", VERMILLION, "--"),
        ("Ghost Uniform (never applied)", base / "ghost_raw_series_logs"
         / "resnet_cifar_Ghost_Uniform_logs.csv", ORANGE, "--"),
    ]
    sizes = (20, 40, 80, 160, 320, 640, 1600, 4000, 7800)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.set_axisbelow(True)
    ax.grid()
    for label, path, colour, style in runs:
        frame = pd.read_csv(path)
        curve = ccm_convergence(frame["train_loss"].to_numpy(),
                                frame["poison_fraction"].to_numpy(),
                                E=3, tau=1, library_sizes=sizes)
        xs = sorted(curve)
        ys = [curve[x] for x in xs]
        ax.plot(xs, ys, style, color=colour, lw=1.6, marker="o", ms=3.2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("library size $L$")
    ax.set_ylabel(r"cross-map skill $\rho$")
    ax.set_ylim(-0.12, 1.04)
    ax.axhline(0, color="#BBBBBB", lw=0.7)
    ax.legend(loc="center left")
    save(fig, "fig_convergence")


# --- Figure 4: the confound-free test --------------------------------------

def figure_confound_free():
    seven = pd.read_csv(CODE / "edm_validation" / "results" / "phase6_manifold_sharing.csv")
    eight = pd.read_csv(CODE / "edm_validation" / "results" / "phase7_confound_free.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    ax = axes[0]
    medians = seven.groupby(["run", "generalizes"]).z.median().reset_index()
    delayed = {"grok", "grok_seed1", "grok_seed2"}
    medians["family"] = ["delayed transition" if r in delayed else "no delayed transition"
                         for r in medians.run]
    medians = medians.sort_values("z")
    colours = [BLUE if f == "delayed transition" else VERMILLION for f in medians.family]
    ax.barh(np.arange(len(medians)), medians.z, height=0.6, color=colours,
            edgecolor="white", lw=0.8)
    ax.set_yticks(np.arange(len(medians)))
    ax.set_yticklabels(medians.run, fontsize=7.3)
    ax.set_xlabel(r"median coupling $z$ (vs. surrogates)")
    ax.set_title("(a) across configurations: apparent separation",
                 fontsize=8.2, color=INK, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    handles = [plt.Line2D([], [], color=BLUE, lw=6),
               plt.Line2D([], [], color=VERMILLION, lw=6)]
    ax.legend(handles, ["delayed transition", "no delayed transition"],
              loc="lower right", fontsize=7)

    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid()
    ax.scatter(eight.gap, eight.z_median, s=34, color=BLUE, zorder=3,
               edgecolor="white", linewidth=0.8, label=r"median over all windows")
    valid = eight.dropna(subset=["z_plateau"])
    ax.scatter(valid.gap, valid.z_plateau, s=34, color=PURPLE, marker="^", zorder=3,
               edgecolor="white", linewidth=0.8, label="median over plateau windows")
    r = np.corrcoef(eight.gap, eight.z_median)[0, 1]
    ax.set_xlabel("grokking gap (optimization steps)")
    ax.set_ylabel(r"coupling $z$")
    ax.set_title("(b) within one configuration: none", fontsize=8.2, color=INK, pad=6)
    ax.set_ylim(-0.8, 3.5)                      # headroom so the note clears the points
    ax.annotate(f"Pearson $r={r:+.3f}$ (n=8)", (0.03, 0.94), xycoords="axes fraction",
                fontsize=7.4, color=INK, va="top")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_confound_free")


# --- Figure 5: the delay is not typical ------------------------------------

def figure_delay_distribution():
    sweep = pd.read_csv(CODE / "prediction_improved" / "results" / "sweep" / "summary.csv")
    gaps = sweep.gap.dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    rng = np.random.default_rng(0)
    ax.scatter(gaps, 1 + rng.uniform(-0.13, 0.13, len(gaps)), s=26, color=BLUE,
               alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3,
               label=f"30 (split, init) combinations of the same configuration")
    ax.scatter([11790], [1], s=90, marker="*", color=VERMILLION, zorder=4,
               edgecolor="white", linewidth=0.8,
               label="split and initialisation from one stream")
    ax.annotate("single stream\n11 790 steps", (11790, 1), textcoords="offset points",
                xytext=(0, 24), ha="center", fontsize=7.4, color=VERMILLION)
    ax.annotate(f"30 runs span {gaps.min():.0f} to {gaps.max():.0f}",
                (np.median(gaps), 0.78), ha="center", fontsize=7.4, color=INK)
    ax.set_yticks([])
    ax.set_ylim(0.62, 1.42)
    ax.set_xlim(0, 13200)
    ax.set_xlabel("memorization-to-generalization gap (optimization steps)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.02))
    for side in ("left",):
        ax.spines[side].set_visible(False)
    save(fig, "fig_delay_distribution")


# --- Figure: the dimension estimate is a constant of a straight line -------

def figure_dimension_artifact():
    lines = pd.read_csv(CODE / "edm_validation" / "results" / "phase8_line_constants.csv")
    controls = pd.read_csv(CODE / "edm_validation" / "results" / "phase8_control_plateaus.csv")
    scan = pd.read_csv(CODE / "edm_validation" / "results" / "phase8_identifiability.csv",
                       index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                             gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid()
    limit = [0.8, 13]
    ax.plot(limit, limit, "-", color="#CCCCCC", lw=1.0, zorder=1)
    ax.scatter(lines.closed_form, lines.measured_on_line, s=42, color=BLUE, zorder=3,
               edgecolor="white", linewidth=0.8, label="estimator on a synthetic line")
    ax.scatter(controls.closed_form, controls.measured_median, s=48, marker="s",
               color=VERMILLION, zorder=3, edgecolor="white", linewidth=0.8,
               label="$WD=0$ control runs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*limit)
    ax.set_ylim(*limit)
    ax.set_xticks([1, 2, 5, 10])
    ax.set_yticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.set_xlabel("closed form for a straight line")
    ax.set_ylabel(r"reported dimension $\hat{d}$")
    ax.set_title("(a) what the estimator reports", fontsize=8.2, color=INK, pad=6)
    ax.legend(loc="upper left")

    # (b) The identifiability diagnostic is length-dependent, so it can only be read
    # against a reference of the same length. An unmatched comparison, a 12000-sample
    # Lorenz against 2000-sample logs and the difference was carried by that alone.
    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid()
    length = pd.read_csv(CODE / "edm_validation" / "results" / "phase10_length_dependence.csv")
    logs = pd.read_csv(CODE / "edm_validation" / "results" / "phase10_log_ratios.csv")

    ax.axvspan(logs.n.min() * 0.82, logs.n.max() * 1.22, color=VERMILLION, alpha=0.10, lw=0)
    ax.annotate("length of\ntraining logs", (float(np.sqrt(logs.n.min() * logs.n.max())), 11.6),
                ha="center", va="top", fontsize=6.8, color=VERMILLION)
    ax.plot(length.n, length.lorenz, "-o", color=BLUE, lw=1.5, ms=3.6,
            label="Lorenz-63 (an attractor)")
    ax.plot(length.n, length.white_noise, ":v", color=GREY, lw=1.5, ms=3.6,
            label="white noise (no attractor)")
    ax.plot(logs.n, logs.ratio, "s", color=VERMILLION, ms=5.5, markeredgecolor="white",
            markeredgewidth=0.8, label="weight-norm logs")
    ax.axhline(1.0, color="#BBBBBB", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("series length $n$")
    ax.set_ylabel(r"growth of $\hat{d}$ with $E_{max}$")
    ax.set_title("(b) the diagnostic needs $n \\gtrsim 6000$", fontsize=8.2, color=INK, pad=6)
    ax.legend(loc="upper right", fontsize=6.8)
    save(fig, "fig_dimension_artifact")


# --- Figure: function-space velocity fails its control ---------------------

def figure_velocity():
    results = CODE / "prediction_improved" / "results"
    extra = CODE / "edm_validation" / "results" / "velocity"
    families = {
        "grok (WD=1, generalises)": (BLUE, ["grok", "grok_seed1", "grok_seed2"]),
        "lowdata (WD=1, never generalises)": (VERMILLION,
                                              ["lowdata15", "lowdata20",
                                               "lowdata15_s1", "lowdata15_s2",
                                               "lowdata20_s1", "lowdata20_s2"]),
        "no weight decay (never generalises)": (GREY, ["wd0"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                             gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.34})

    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid()
    summary = []
    for label, (colour, runs) in families.items():
        first = True
        for run in runs:
            path = results / f"{run}_probe.csv"
            if not path.exists():
                path = extra / f"{run}_probe.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            steps = frame["step"].to_numpy()
            velocity = frame["val_velocity"].to_numpy()
            edges = np.linspace(0, steps.max(), 21)
            centres, medians = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (steps >= lo) & (steps < hi)
                if mask.any():
                    centres.append((lo + hi) / 2)
                    medians.append(np.nanmedian(velocity[mask]))
            ax.plot(centres, medians, "-", color=colour, lw=1.4, alpha=0.85,
                    label=label if first else None)
            first = False
            late = np.isfinite(velocity) & (steps > steps.max() / 2)
            summary.append({"family": label, "run": run, "colour": colour,
                            "late": float(np.median(velocity[late]))})

    ax.set_yscale("log")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("median normalized-logit velocity")
    ax.set_title("(a) time course", fontsize=8.2, color=INK, pad=6)
    ax.legend(loc="lower left", fontsize=6.9)

    ax = axes[1]
    table = pd.DataFrame(summary)
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    order = table.sort_values("late")
    ax.barh(np.arange(len(order)), order.late, height=0.6, color=order.colour,
            edgecolor="white", lw=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order.run, fontsize=6.9)
    ax.set_xscale("log")
    ax.set_xlabel("velocity, second half of training")
    ax.set_title("(b) the control sits highest", fontsize=8.2, color=INK, pad=6)
    save(fig, "fig_velocity")


def figure_paper_settings():
    """The dimension signal at W=0, that is, with no temporal exclusion."""
    res = CODE / "edm_validation" / "results"
    summary = pd.read_csv(res / "phase9_paper_settings_summary.csv")
    families = {
        "grok (WD=1, generalises)": (BLUE, ["grok", "grok_s1", "grok_s2"]),
        "lowdata (WD=1, never generalises)": (VERMILLION, ["lowdata15", "lowdata20"]),
        "no weight decay (never generalises)": (GREY, ["wd0"]),
    }
    styles = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5),
                             gridspec_kw={"width_ratios": [1.3, 0.8, 1.0], "wspace": 0.45})

    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid()
    pooled = []
    for label, (colour, runs) in families.items():
        for i, run in enumerate(runs):
            frame = pd.read_csv(res / f"phase9_trace_{run}.csv")
            ax.plot(frame.step, frame.dim_W0, styles[i % len(styles)], color=colour,
                    lw=1.3, alpha=0.9, label=label if i == 0 else None)
            outcome = "generalises" if run.startswith("grok") else "never generalises"
            pooled.append(frame.assign(outcome=outcome, colour=colour))
    ax.set_xlabel("optimization step")
    ax.set_ylabel("estimated dimension, $W=0$")
    ax.set_title("(a) no Theiler window", fontsize=8.2, color=INK, pad=6)
    ax.legend(loc="upper left", fontsize=6.4)

    # (b) The claim under test is a fall. A slope graph shows the direction each run
    # actually takes, which is the quantity the claim is about.
    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    colour_of = {r: c for label, (c, runs) in families.items() for r in runs for c in [c]}
    for _, row in summary.iterrows():
        colour = colour_of.get(row.run, GREY)
        ax.plot([0, 1], [row.dim_early, row.dim_late], "-o", color=colour, lw=1.3,
                ms=4.0, alpha=0.9, markeredgecolor="white", markeredgewidth=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["early", "late"], fontsize=7.2)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("estimated dimension, $W=0$")
    ax.set_title("(b) direction varies by seed", fontsize=8.2, color=INK, pad=6)

    # (c) what the estimate tracks: a one-line roughness ratio, no embedding involved.
    ax = axes[2]
    ax.set_axisbelow(True)
    ax.grid()
    table = pd.concat(pooled)
    for outcome, colour in (("generalises", BLUE), ("never generalises", VERMILLION)):
        sub = table[table.outcome == outcome]
        ax.plot(sub.roughness, sub.dim_W0, ".", ms=2.2, color=colour, alpha=0.45,
                label=outcome)
    ax.set_xlabel(r"roughness  std$(\Delta x)/$std$(x)$")
    ax.set_ylabel("estimated dimension, $W=0$")
    ax.set_title("(c) tracks smoothness", fontsize=8.2, color=INK, pad=6)
    ax.legend(loc="lower right", fontsize=6.6, markerscale=3)
    save(fig, "fig_paper_settings")


def figure_directionality():
    """Forward against reverse cross mapping. Ground truth is unidirectional by design."""
    res = CODE / "edm_validation" / "results"
    # Both panels use a matched embedding dimension for the two directions, so the
    # comparison is symmetric. The main table additionally grants the reverse the best
    # of five embeddings; that stress test is reported in the text, not here.
    table = pd.read_csv(res / "phase12_matched_gains.csv")
    curves = pd.read_csv(res / "phase12_direction_curves.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.34})

    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid()
    colours = {"LogisticMap": BLUE, "Random": SKY, "Ghost_Normal": VERMILLION}
    for run, colour in colours.items():
        sub = curves[curves.run == run].sort_values("library")
        ax.plot(sub.library, sub.forward, "-o", color=colour, lw=1.5, ms=3.4)
        ax.plot(sub.library, sub.reverse, "--v", color=colour, lw=1.3, ms=3.4, alpha=0.75)
    ax.set_xscale("log")
    ax.axhline(0, color="#BBBBBB", lw=0.8)
    ax.set_xlabel("library size $L$")
    ax.set_ylabel(r"cross-map skill $\rho$")
    ax.set_title("(a) convergence, both directions", fontsize=8.2, color=INK, pad=6)
    handles = [plt.Line2D([], [], color=INK, lw=1.5, marker="o", ms=3.4),
               plt.Line2D([], [], color=INK, lw=1.3, ls="--", marker="v", ms=3.4)]
    handles += [plt.Line2D([], [], color=c, lw=4) for c in colours.values()]
    ax.legend(handles, ["loss xmap driver", "driver xmap loss"] + list(colours),
              loc="center right", fontsize=6.3)

    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid()
    limit = [-0.06, 0.40]
    ax.plot(limit, limit, "-", color="#CCCCCC", lw=1.0, zorder=1)
    for coupled, colour, label in ((True, BLUE, "driver applied"),
                                   (False, VERMILLION, "driver never applied")):
        sub = table[table.coupled == coupled]
        ax.scatter(sub.gain_rev, sub.gain_fwd, s=44, color=colour, zorder=3,
                   edgecolor="white", linewidth=0.8, label=label)
    outlier = table[table.run == "sinusoid"]
    if len(outlier):
        ax.annotate("sinusoid\n(periodic)",
                    (float(outlier.gain_rev.iloc[0]), float(outlier.gain_fwd.iloc[0])),
                    textcoords="offset points", xytext=(-16, 20), ha="right",
                    fontsize=6.6, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                    shrinkA=1, shrinkB=3))
    ax.set_xlim(*limit)
    ax.set_ylim(*limit)
    ax.set_xlabel("reverse convergence gain")
    ax.set_ylabel("forward convergence gain")
    ax.set_title("(b) convergence gain, matched $E$", fontsize=8.2, color=INK, pad=6)
    ax.legend(loc="upper left", fontsize=6.8)
    save(fig, "fig_directionality")


if __name__ == "__main__":
    print("generating figures ...")
    figure_dimension_artifact()
    figure_paper_settings()
    figure_velocity()
    figure_preconditions()
    figure_driver_recovery()
    figure_convergence()
    figure_directionality()
    figure_confound_free()
    figure_delay_distribution()
    print("done")
