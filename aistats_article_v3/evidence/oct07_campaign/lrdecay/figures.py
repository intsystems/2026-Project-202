"""Figures for REPORT_ru.md (labels in English)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from main import CONDS  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": GRAY, "ytick.color": GRAY})


def response_curves():
    per = pd.read_csv(RES / "test_per_run.csv")
    fig, axs = plt.subplots(2, 5, figsize=(11, 4.4), sharey=False)
    for ax, name in zip(axs.flat, "ABCDEFGHIJ"):
        rs = [json.load(open(f)) for f in sorted(RES.glob(f"res_{name}_s*.json"))]
        rs = [r for r in rs if r["cond"]["split"] == "test" and r.get("done")]
        if not rs:
            continue
        T = rs[0]["cond"]["T"]
        G = rs[0]["grid"]
        acc = np.array([[r["branches"][str(t)]["test_acc"] for t in G] + [r["none"]["test_acc"]] for r in rs])
        xs = [t / T for t in G] + [1.0]
        ax.plot(xs, acc.mean(0), color=BLUE, lw=2, marker="o", ms=3, label="decay x10 at t (mean)")
        ax.axhline(np.mean([r["cosine"]["test_acc"] for r in rs]), color=GRAY, lw=1, ls="--", label="cosine")
        p = per[(per.cond == name)]
        for rule, col, mk in (("MG", ORANGE, "v"), ("fixed_step", INK, "s"), ("plateau_val_loss", AQUA, "^")):
            q = p[p.rule == rule]
            if len(q):
                d = np.where(q.decay.isna(), 1.0, q.decay / q["T"])
                ax.scatter(d, q.test_acc, color=col, marker=mk, s=18, zorder=3, label=rule)
        c = CONDS[name]
        ax.set_title(f"{name}: lr {c['lr']}, bs {c['bs']}, n {c['n']}, noise {c['noise']}, T {c['T']}", fontsize=7)
        ax.set_xlabel("decay time / budget (1.0 = never)")
        ax.set_ylabel("final test acc")
    axs.flat[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_response_curves.png", dpi=150)


def mg_traces(log="param_norm", stat="MG"):
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, name in zip(axs, ("A", "B", "F")):
        for s, col in ((0, BLUE), (1, ORANGE)):
            f = RES / f"win_{name}_s{s}.csv"
            if not f.exists():
                continue
            w = pd.read_csv(f)
            w = w[w.log == log]
            ax.plot(w.end, w[stat], color=col, lw=2, label=f"seed {s}")
        ax.set_title(f"{name}: lr {CONDS[name]['lr']}, bs {CONDS[name]['bs']} ({stat} of {log})", fontsize=8)
        ax.set_xlabel("window end, step")
        ax.set_ylabel(stat)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / f"fig_{stat}_{log}_calibration.png", dpi=150)


if __name__ == "__main__":
    response_curves()
    mg_traces("param_norm")
    mg_traces("batch_loss")
