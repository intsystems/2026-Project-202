"""Does the rank collapse of ``../active_rank/`` survive a matched control?

That result -- a transient one-dimensional bottleneck in the trajectory at
generalisation -- was measured on six transformer runs whose controls were the
*no-weight-decay* configurations.  In that run set "generalises" and "has weight decay"
move together, so its own report flags the gap:

    "The controls are not perfectly matched. `wd0` differs from `wd1` in more than
     whether it generalises."

Here every run has ``weight_decay = 0``, and each control differs from its partner by
**one monomial in the label function** and nothing else -- same width, same learning
rate, same training fraction, same step budget, both reaching 100% training accuracy.
So the two ways the earlier result could have been an artefact are closed at once: a
weight-decay detector must stay silent on all eight runs, and a
difficulty-of-optimisation detector cannot separate members of a pair.

The control has no ``t_gen`` of its own, so it is aligned on **its partner's** -- which
is the whole point.  If the dip tracked training progress rather than generalisation,
it would appear at the same place in both members.

    python rank_dip.py --indir ./results/rank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

_AR = Path(__file__).resolve().parent.parent / "active_rank"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))
from analyze_rank import milestones   # noqa: E402  -- one definition of t_mem/t_gen

PAIRS = [
    ("a_add", "x_no_grok", "n+m  vs  n^3+nm^2+m"),
    ("g_p1_p97", "g_p1x_p97", "(4n1+n2^2)^3  vs  +n1n2"),
    ("g_p2_p97", "g_p2x_p97", "(2n1+3n2)^4  vs  -n1^2"),
    ("g_p3_p97", "g_p3x_p97", "(5n1^3+2n2^4)^2  vs  -n2"),
]
STATS = ["fn_PR_pos_det", "fn_PR_step5", "PR_pos_det", "PR_step5"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="./results/rank")
    ap.add_argument("--halfwidth", type=int, default=4000,
                    help="steps either side of t_gen searched for the dip")
    args = ap.parse_args()

    root = Path(args.indir)
    win = pd.read_csv(root / "rank_windows.csv")
    win["centre"] = (win.left_step + win.right_step) / 2.0

    tgen = {}
    for run in win.run.unique():
        tgen[run] = milestones(pd.read_csv(root / f"{run}_train.csv"))[1]

    rows = []
    for grok_run, ctrl_run, label in PAIRS:
        if grok_run not in tgen or ctrl_run not in tgen:
            continue
        tg = tgen[grok_run]
        if tg is None:
            print(f"[{grok_run}] never generalised -- skipping its pair")
            continue
        for run, role in ((grok_run, "generalises"), (ctrl_run, "control")):
            g = win[win.run == run].sort_values("centre")
            for s in STATS + ["move"]:
                pre = g[(g.centre >= tg - args.halfwidth) & (g.centre < tg)][s]
                near = g[(g.centre >= tg - args.halfwidth) &
                         (g.centre <= tg + args.halfwidth)]
                if not len(pre) or not len(near):
                    continue
                i = near[s].idxmin()
                rows.append(dict(pair=label, run=run, role=role, stat=s, t_gen=tg,
                                 plateau=pre.median(), dip=near[s].min(),
                                 at=near.centre[i] - tg,
                                 depth=pre.median() / max(near[s].min(), 1e-9)))
    tab = pd.DataFrame(rows)
    tab.to_csv(root / "rank_dip_pairs.csv", index=False)

    for s in STATS + ["move"]:
        sub = tab[tab.stat == s]
        if sub.empty:
            continue
        print(f"\n=== {s} ===")
        print(sub[["pair", "role", "plateau", "dip", "at", "depth"]]
              .round(2).to_string(index=False))

    print("\n=== depth, generalising vs its own matched control ===")
    head = f"{'stat':<16}{'pair':<32}{'grok':>8}{'control':>9}{'ratio':>8}"
    print(head)
    print("-" * len(head))
    for s in STATS:
        for _, _, label in PAIRS:
            sub = tab[(tab.stat == s) & (tab.pair == label)]
            if len(sub) != 2:
                continue
            d_g = float(sub[sub.role == "generalises"].depth.iloc[0])
            d_c = float(sub[sub.role == "control"].depth.iloc[0])
            print(f"{s:<16}{label:<32}{d_g:>8.2f}{d_c:>9.2f}{d_g / max(d_c, 1e-9):>8.2f}")

    # ---- figure: every pair aligned on the generalising member's t_gen -----------
    fig, axes = plt.subplots(len(STATS), 1, figsize=(7.2, 2.5 * len(STATS)), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), STATS):
        for grok_run, ctrl_run, label in PAIRS:
            tg = tgen.get(grok_run)
            if tg is None:
                continue
            for run, style in ((grok_run, "-"), (ctrl_run, "--")):
                g = win[win.run == run].sort_values("centre")
                if g.empty:
                    continue
                ax.plot(g.centre - tg, g[s], style, lw=1.1, alpha=.85,
                        label=f"{run}" if s == STATS[0] else None)
        ax.axvline(0, color="k", lw=1, ls=":")
        ax.set_ylabel(s, fontsize=9)
        ax.grid(alpha=.25)
        ax.set_xlim(-args.halfwidth, args.halfwidth)
    np.atleast_1d(axes)[-1].set_xlabel("steps since the generalising member's t_gen")
    np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("Trajectory participation ratio, label-matched pairs, all weight_decay = 0\n"
                 "solid: generalises   dashed: its control, aligned on the partner's t_gen",
                 fontsize=10)
    fig.tight_layout()
    # Named after the input directory: the full-batch and mini-batch arms are the same
    # runs measured under different noise, and a shared filename means the second call
    # silently overwrites the first with a figure that looks just as plausible.
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    path = out / f"rank_dip_{root.name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n-> {path}")
    print(f"-> {root / 'rank_dip_pairs.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
