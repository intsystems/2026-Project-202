"""Checkpoint time-course for the existing smooth-policy experiment.

No training is performed. One fixed held-out reset is evaluated for each
checkpoint and arm, then MG is computed on the action norm.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from motion import Actor, rollout
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research_walker_phase_wide"))
from features import measure

H = Path(__file__).resolve().parent
STEPS = [0, 262144, 524288, 786432, 1048576]
RESET = 62001
W = 2048
TAU = 8

def action_mg(actions):
    x = np.linalg.norm(actions, axis=1)
    r = measure(x[-W:], W, TAU)
    return float(r["MG"]) if np.isfinite(r.get("MG", np.nan)) and not r.get("degenerate", True) else np.nan

def main():
    rows=[]
    for seed in range(221, 226):
        values={}
        for arm in [0,1]:
            label=f"seed{seed}_lambda{arm}"
            values[arm]={}
            for step in STEPS:
                cp=H/label/f"step{step:07d}"
                root=cp/f"reset{RESET}"
                traj=root/"trajectory.npz"
                if traj.exists():
                    try:
                        with np.load(traj) as zcheck:
                            bad=zcheck["actions"].ndim != 2 or zcheck["actions"].shape[0] < W
                    except Exception:
                        bad=True
                    if bad:
                        for name in ["metrics.json","trajectory.npz"]:
                            (root/name).unlink(missing_ok=True)
                result=rollout(Actor(cp), cp, RESET)
                data=np.load(cp/f"reset{RESET}/trajectory.npz")
                valid=data["actions"].ndim == 2 and data["actions"].shape[0] >= W
                mg=action_mg(data["actions"]) if valid else np.nan
                j1=float(np.mean(np.diff(data["actions"],axis=0)**2)) if valid else np.nan
                j2=float(np.mean(np.diff(data["actions"],n=2,axis=0)**2)) if valid else np.nan
                values[arm][step]=dict(MG=mg,J1=j1,J2=j2,healthy=bool(result["eligible"]))
                print("DONE",seed,arm,step,flush=True)
        for step in STEPS:
            c,s=values[0][step],values[1][step]
            rows.append(dict(seed=seed,step=step,MG_control=c["MG"],MG_smooth=s["MG"],
                             MG_ratio=s["MG"]/c["MG"] if np.isfinite(c["MG"]) and np.isfinite(s["MG"]) else np.nan,
                             J1_ratio=s["J1"]/c["J1"] if np.isfinite(c["J1"]) and np.isfinite(s["J1"]) else np.nan,
                             J2_ratio=s["J2"]/c["J2"] if np.isfinite(c["J2"]) and np.isfinite(s["J2"]) else np.nan,
                             healthy_control=c["healthy"],
                             healthy_smooth=s["healthy"]))
    out=pd.DataFrame(rows)
    out.to_csv(H/"checkpoint_timecourse.csv",index=False)
    summary=out.groupby("step").agg(MG_ratio=("MG_ratio","median"),J1_ratio=("J1_ratio","median"),J2_ratio=("J2_ratio","median"),
                                      seeds=("seed","count")).reset_index()
    summary.to_csv(H/"checkpoint_timecourse_summary.csv",index=False)
    print(summary.to_string(index=False))

if __name__=="__main__":
    with threadpool_limits(limits=1): main()
