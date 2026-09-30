"""E9 analysis: P1-P5 of run.py, tables and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
ORDER = ["T1", "T2", "T3", "T4", "H2", "H4", "M4", "chaos"]
LINES = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "H2": 2, "H4": 4, "M4": 4}


def load():
    rows = []
    for f in sorted(RES.glob("runs_s*.json")):
        rows += json.load(open(f))
    df = pd.DataFrame(rows)
    df["lines"] = df.arm.map(LINES)
    return df


def rho(a, b):
    return float(spearmanr(a, b)[0])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    L = df[df.learned]
    out = {"n_runs": len(df), "n_learned": int(df.learned.sum()),
           "unlearned": df[~df.learned & (df.arm != "chaos")][["arm", "seed", "fidelity"]]
           .to_dict("records")}

    t = L[L.arm.str.startswith("T")]
    out["P1_spearman_MG_d_T"] = rho(t.MG_n0, t.d)

    wide = df.pivot(index="seed", columns="arm", values="MG_n0")
    learned = df.pivot(index="seed", columns="arm", values="learned").fillna(False)

    def seeds_ok(cond, arms):
        ok = learned[arms].all(axis=1)
        return int((cond & ok).sum()), int(ok.sum())

    out["P2_H4<M4<T4"] = seeds_ok((wide.H4 < wide.M4) & (wide.M4 < wide.T4), ["H4", "M4", "T4"])
    out["P2_H2<T2"] = seeds_ok(wide.H2 < wide.T2, ["H2", "T2"])
    for c in ["peaks", "PR"]:
        wc = df.pivot(index="seed", columns="arm", values=c)
        out[f"P2_{c}_H4<M4<T4"] = seeds_ok((wc.H4 < wc.M4) & (wc.M4 < wc.T4), ["H4", "M4", "T4"])
        out[f"P2_{c}_H2<T2"] = seeds_ok(wc.H2 < wc.T2, ["H2", "T2"])

    out["P3_spearman_d"] = {c: rho(L[c], L.d) for c in ["MG_n0", "MG_n1", "MG_n2", "peaks", "PR", "n_zero"]}
    out["P4_spearman_MG_nzero"] = rho(L.MG_n0, L.n_zero)
    out["n_zero_equals_d"] = f"{int((L.n_zero == L.d).sum())}/{len(L)}"

    p5 = []
    for s in wide.index:
        ref = [a for a in ["T1", "T2", "H2", "H4"] if learned.loc[s].get(a, False)]
        p5.append(bool(all(wide.loc[s, "chaos"] > wide.loc[s, a] for a in ref)))
    out["P5_chaos_above"] = f"{sum(p5)}/{len(p5)}"

    per_arm = df.groupby("arm").agg(
        learned=("learned", "sum"), n=("seed", "count"), d=("d", "first"), lines=("lines", "first"),
        n_zero=("n_zero", "median"), lyap1=("lyap1", "median"), peaks=("peaks", "median"),
        PR=("PR", "median"), MG_n0=("MG_n0", "median"), MG_n0_min=("MG_n0", "min"),
        MG_n0_max=("MG_n0", "max"), MG_n1=("MG_n1", "median"), MG_n2=("MG_n2", "median"),
        t_train=("t_train", "median"), t_lyap=("t_lyap", "median"), t_MG=("t_MG", "median"))
    per_arm = per_arm.reindex([a for a in ORDER if a in per_arm.index])
    out["per_arm"] = per_arm.round(3).reset_index().to_dict("records")
    json.dump(out, open(RES / "summary.json", "w"), indent=1, default=float)
    df.drop(columns=[c for c in df.columns if c.endswith("windows") or c in ("lyap", "per_output")]) \
        .to_csv(RES / "runs.csv", index=False)
    pd.set_option("display.width", 220)
    print(per_arm.round(2).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "per_arm"}, indent=1, default=float))
    figures(df)


def figures(df):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"T1": "#c6dbef", "T2": "#6baed6", "T3": "#2171b5", "T4": "#08306b",
           "H2": "#fdae6b", "H4": "#d94801", "M4": "#41ab5d"}
    label = {"T1": "T1: 1 line", "T2": "T2: 2 lines", "T3": "T3: 3 lines", "T4": "T4: 4 lines",
             "H2": "H2: 2 harmonics", "H4": "H4: 4 harmonics", "M4": "M4: 2 bases x 2 harmonics"}
    L = df[df.learned]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    panels = [("MG_n0", "(a) MG of one neuron's rate (cheap)"),
              ("peaks", "(b) spectral peaks of the output"),
              ("PR", "(c) PCA participation ratio, all 1000 neurons")]
    for k, (c, lab) in enumerate(panels):
        a = ax[k]
        for arm in ["T1", "T2", "T3", "T4", "H2", "H4", "M4"]:
            g = L[L.arm == arm]
            jit = (np.arange(len(g)) - (len(g) - 1) / 2) * 0.035
            off = {"H2": -.2, "H4": .2, "M4": .2}.get(arm, 0)
            a.scatter(g.d + jit + off, g[c], color=col[arm], s=22, zorder=3,
                      label=label[arm] if k == 0 else None, edgecolor="k", linewidth=.3)
        a.set_xticks([1, 2, 3, 4])
        a.set_xlim(.6, 4.4)
        a.set_xlabel("active dimension d (independent phases)")
        a.set_title(lab, fontsize=9)
    ax[0].plot([1, 4], [1, 4], color="k", lw=.7, ls=":", label="MG = d")
    ch = df[df.arm == "chaos"].MG_n0
    ax[0].set_ylim(0, 9.5)
    ax[0].text(.62, 9.0, f"untrained chaotic network: MG {ch.min():.0f}-{ch.max():.0f} (off scale)",
               fontsize=7.5, color="#555", va="top")
    fig.legend(loc="upper center", ncol=8, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.0),
               handletextpad=.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(FIG / "mg_vs_dimension.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
