"""Serial CPU timing on the same trajectory, one numerical thread.

Input preparation and acquisition excluded. MG includes scalar normalisation
and estimation; state reference includes its state reduction. Warmup + 5 trials.
"""
from pathlib import Path
import sys, time, platform, json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
from motion import state_reference
sys.path.insert(0,str(H.parent/'research_walker_phase_wide'))
from features import measure
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate

def main():
    source=H/'seed231_lambda0/step1048576/reset62001/trajectory.npz'
    z=np.load(source)
    a=np.asarray(z['actions'],dtype=float)
    qpos,qvel=z['qpos'].copy(),z['qvel'].copy()
    z.close()
    x=np.linalg.norm(a,axis=1)
    cfg=EstimatorConfig(window=2048,max_E=20,tau=8,k_neighbors=20,theiler=312,theiler_cap=312)
    funcs={
      'J1_J2_4096':lambda: (np.mean(np.diff(a,axis=0)**2),np.mean(np.diff(a,n=2,axis=0)**2)),
      'state_R_D_2048':lambda: state_reference(qpos[-2048:],qvel[-2048:]),
      'state_R_D_4096':lambda: state_reference(qpos,qvel),
      'MG20_one_window':lambda: estimate(x[-2048:],cfg,seed=123),
      'MG20_plus_E40_one_window':lambda: measure(x[-2048:],2048,8),
      'MG20_plus_E40_three_windows':lambda: [measure(x[e-2048:e],2048,8) for e in [2048,3072,4096]],
    }
    rows=[]
    with threadpool_limits(limits=1):
      for name, fn in funcs.items():
        fn()
        for rep in range(5):
          start=time.perf_counter();fn();duration=time.perf_counter()-start
          rows.append(dict(method=name,repeat=rep,seconds=duration))
    d=pd.DataFrame(rows);d.to_csv(H/'timing_raw.csv',index=False)
    s=d.groupby('method').seconds.agg(['median','min','max']).reset_index()
    s.to_csv(H/'timing_summary.csv',index=False)
    (H/'timing_environment.json').write_text(json.dumps(dict(platform=platform.platform(),processor=platform.processor(),python=sys.version,threads=1,source=str(source),repeats=5),indent=2))
    print(s.to_string(index=False))
if __name__=='__main__':main()
