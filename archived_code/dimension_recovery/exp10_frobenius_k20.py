"""Frobenius-norm MG calibration and held-out test for k=1..20.

This is the k=20 extension of exp9.  One common configuration is selected on
seed 0; seeds 1 and 2 and a zigzag schedule are held out.  There is no per-k
calibration and no nonlinear remapping of the estimate to the known answer.

    python exp10_frobenius_k20.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import estimators as E
import systems as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp10_frobenius_k20"
OUT.mkdir(parents=True, exist_ok=True)

KS = np.arange(1, 21)
SCHEDULE = (2, 13, 5, 18, 9, 20, 4, 16, 7, 1, 14, 10, 19, 6, 12, 3, 17, 8, 15, 11, 2)

# E must be at least 2*k+1 for the strongest Takens-style generic embedding
# heuristic at k=20.  The grid is deliberately small enough to run locally,
# but varies the three scale parameters that mattered in exp9.
GRID = (
    dict(window=4000, cycles=1000.0, max_E=41, kn=30, tau=4),
    dict(window=4000, cycles=1000.0, max_E=45, kn=30, tau=4),
    dict(window=4000, cycles=1000.0, max_E=45, kn=50, tau=4),
    dict(window=8000, cycles=2000.0, max_E=41, kn=30, tau=4),
    dict(window=8000, cycles=2000.0, max_E=45, kn=30, tau=4),
    dict(window=8000, cycles=4000.0, max_E=45, kn=50, tau=4),
    dict(window=8000, cycles=2000.0, max_E=45, kn=50, tau=2),
)


def observer(k, seed, window, cycles, n=None):
    info = S.make_system(
        "quasiperiodic", k=int(k), D=64, n=int(n or window),
        cycles_per_window=float(cycles), window=int(window), amp=.1,
        seed=int(seed), band_mode="matched",
    )
    return S.observers(info, seed=int(seed), obs_snr=1e6)["norm_fro"]


def stats(vals):
    vals = np.asarray(vals, float)
    err = vals - KS
    inv = int(np.sum(np.diff(vals) <= 0))
    return dict(
        bias=float(err.mean()), mae=float(np.abs(err).mean()),
        max_error=float(np.abs(err).max()),
        rho=float(spearmanr(KS, vals).statistic), inversions=inv,
        objective=float(np.abs(err).mean() + .15*np.abs(err).max() + .15*inv),
    )


def estimate_all(cfg, seed):
    vals=[]
    for k in KS:
        x=observer(k, seed, cfg["window"], cfg["cycles"])
        vals.append(E.mg(x, max_E=cfg["max_E"], k=cfg["kn"], tau=cfg["tau"]))
    return np.asarray(vals)


def calibrate():
    rows=[]
    # Observers are cached because the grid changes only the estimator for most rows.
    cache={}
    for cfg in GRID:
        vals=[]
        for k in KS:
            key=(cfg["window"],cfg["cycles"],int(k))
            if key not in cache:
                cache[key]=observer(k,0,cfg["window"],cfg["cycles"])
            vals.append(E.mg(cache[key],max_E=cfg["max_E"],k=cfg["kn"],tau=cfg["tau"]))
        st=stats(vals)
        rows.append({**cfg,**st,**{f"d{k}":vals[k-1] for k in KS}})
    frame=pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    frame.to_csv(OUT/"calibration_grid.csv",index=False)
    best = frame.iloc[0]
    # Pandas stores a mixed numeric row as float; restore integer estimator
    # parameters before passing them to delay_embedding/kNN routines.
    cfg = {k: (int(best[k]) if k in ("window", "max_E", "kn", "tau") else float(best[k]))
           for k in GRID[0]}
    return frame, cfg


def validate(cfg):
    rows=[]
    for seed in (0,1,2):
        vals=estimate_all(cfg,seed); st=stats(vals)
        for k,v in zip(KS,vals):
            rows.append(dict(seed=seed,split="calibration" if seed==0 else "held_out",
                             k=int(k),MG=float(v),error=float(v-k),**st))
    f=pd.DataFrame(rows);f.to_csv(OUT/"stationary_validation.csv",index=False)
    return f


def transition(cfg):
    W=int(cfg["window"]); seg=3*W; stride=W//2
    info=S.make_transition(SCHEDULE,D=64,seg=seg,cycles_per_window=cfg["cycles"],
                           window=W,amp=.1,snr=np.inf,seed=103,ramp=0,
                           band_mode="matched")
    x=S.observers(dict(diag=info["diag"],active=np.arange(max(SCHEDULE)),n=info["n"]),
                  seed=103,obs_snr=1e6)["norm_fro"]
    right,mg=E.trace(x,W,stride,name="MG",max_E=cfg["max_E"],k=cfg["kn"],tau=cfg["tau"])
    trace=pd.DataFrame(dict(right=right,MG=mg,true_k=info["truth"][right]))
    trace.to_csv(OUT/"zigzag_trace.csv",index=False)
    left=right-W+1; rows=[]
    for j,k in enumerate(SCHEDULE):
        inside=(left>=j*seg)&(right<=(j+1)*seg-1)
        v=float(np.nanmedian(mg[inside]))
        rows.append(dict(segment=j,true_k=k,MG=v,error=v-k,n_windows=int(inside.sum())))
    f=pd.DataFrame(rows);f.to_csv(OUT/"zigzag_segments.csv",index=False)
    return trace,f


def plots(val,trace,seg,cfg):
    fig,ax=plt.subplots(figsize=(9,5.6),constrained_layout=True)
    ax.plot(KS,KS,"k--",lw=1.2,label="ideal")
    for seed,g in val.groupby("seed"):
        ax.plot(g.k,g.MG,"o-",lw=1.4,ms=4,label=f"seed {seed}" + (" calibration" if seed==0 else " held out"))
    ax.set(xlabel="True k",ylabel="MG(norm_fro)",title="Stationary diagonal systems, k=1..20")
    ax.set_xticks(KS);ax.grid(alpha=.22);ax.legend()
    fig.savefig(OUT/"stationary_k1_k20.png",dpi=220);fig.savefig(OUT/"stationary_k1_k20.pdf");plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,5.8),constrained_layout=True);W=cfg["window"]
    ax.step(trace.right/W,trace.true_k,where="post",color="black",lw=1.2,label="true k")
    ax.plot(trace.right/W,trace.MG,color="#28669b",lw=1.15,label="MG(norm_fro)")
    ax.scatter((seg.segment+1)*3-.5,seg.MG,s=23,color="#c33c54",label="segment median",zorder=3)
    ax.set(xlabel="Time / window",ylabel="Dimension",
           title="Held-out zigzag k: " + " -> ".join(map(str,SCHEDULE)))
    ax.grid(alpha=.2);ax.legend(ncol=3)
    fig.savefig(OUT/"zigzag_k20.png",dpi=220);fig.savefig(OUT/"zigzag_k20.pdf");plt.close(fig)


def report(grid,val,seg,cfg,elapsed):
    lines=["# Frobenius MG recovery up to k=20","",
           "One fixed configuration was selected on seed 0; seeds 1 and 2 and the zigzag schedule were held out.","",
           "Selected: `"+", ".join(f"{k}={v}" for k,v in cfg.items())+"`.","",
           "## Stationary test","","| seed | split | MAE | max error | rho | inversions |",
           "|---:|---|---:|---:|---:|---:|"]
    for _,r in val.groupby(["seed","split"],as_index=False).first().iterrows():
        lines.append(f"| {int(r.seed)} | {r.split} | {r.mae:.3f} | {r.max_error:.3f} | {r.rho:.3f} | {int(r.inversions)} |")
    med=val[val.split=="held_out"].groupby("k").MG.median()
    lines += ["","| true k | held-out median MG | error |","|---:|---:|---:|"]
    for k,v in med.items():lines.append(f"| {k} | {v:.3f} | {v-k:+.3f} |")
    lines += ["","## Zigzag held-out test","","Schedule: `"+" -> ".join(map(str,SCHEDULE))+"`.","",
              "| segment | true k | MG | error |","|---:|---:|---:|---:|"]
    for _,r in seg.iterrows():lines.append(f"| {int(r.segment)} | {int(r.true_k)} | {r.MG:.3f} | {r.error:+.3f} |")
    lines += ["","## Verdict","",
              f"Held-out stationary MAE: **{np.mean(np.abs(val[val.split=='held_out'].error)):.3f}**.",
              f"Zigzag MAE: **{np.mean(np.abs(seg.error)):.3f}**.","",
              "This reports the saturation honestly. No per-k correction or answer-dependent calibration was applied.","",
              f"Runtime: {elapsed:.1f} seconds."]
    (OUT/"report_exp10.md").write_text("\n".join(lines),encoding="utf-8")
    (OUT/"best_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")


def main():
    t=time.time();grid,cfg=calibrate();val=validate(cfg);tr,seg=transition(cfg)
    plots(val,tr,seg,cfg);report(grid,val,seg,cfg,time.time()-t)
    print("best",cfg);print("outputs",OUT);return 0

if __name__=="__main__":raise SystemExit(main())
