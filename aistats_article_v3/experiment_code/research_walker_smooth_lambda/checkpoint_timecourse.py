"""All coefficients at five fixed checkpoints, one fixed reset, no selection."""
from pathlib import Path
import concurrent.futures
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from motion import Actor, rollout
from action_mg import analyze, label, COEFS, SEEDS, H
import sys
sys.path.insert(0,str(H.parent/'research_walker_phase_wide'))
from features import measure
STEPS=[0,262144,524288,786432,1048576]
RESET=62001

def run(job):
    seed,coef,step=job
    torch.set_num_threads(1)
    cp=H/label(seed,coef)/f'step{step:07d}'
    cache=cp/f'reset{RESET}'/'timecourse_MG_windows.csv'
    with threadpool_limits(limits=1):
      result=rollout(Actor(cp),cp,RESET)
      mg=np.nan
      if result['eligible']:
        source=cp/f'reset{RESET}'/'action_MG_windows.csv'
        if source.exists():
            d=pd.read_csv(source);d=d[d.signal=='action_norm']
        elif cache.exists():d=pd.read_csv(cache)
        else:
            with np.load(cp/f'reset{RESET}'/'trajectory.npz') as z:
                x=np.linalg.norm(z['actions'].astype(float),axis=1)
            d=pd.DataFrame([dict(end=e,**measure(x[e-2048:e],2048,8)) for e in [2048,3072,4096]])
            d.to_csv(cache,index=False)
        if (d.MG.notna() & ~d.degenerate).all():mg=float(d.MG.median())
    return dict(seed=seed,coef=coef,step=step,MG=mg,eligible=result['eligible'],
                J1=result['J1'],J2=result['J2'],padded_reward=result['padded_reward'])

def main():
    jobs=[(s,c,t) for s in SEEDS for c in COEFS for t in STEPS]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
      for i,r in enumerate(pool.map(run,jobs),1):
        rows.append(r)
        if i%10==0:print(f'TIMECOURSE {i}/{len(jobs)}',flush=True)
    d=pd.DataFrame(rows);d.to_csv(H/'timecourse_raw.csv',index=False)
    b=d[d.coef==0].drop(columns='coef').rename(columns={k:k+'_base' for k in ['MG','J1','J2','eligible','padded_reward']})
    x=d.merge(b,on=['seed','step'],validate='many_to_one')
    ok=x.eligible & x.eligible_base
    for k in ['MG','J1','J2']:
        x[k+'_ratio']=np.where(ok,x[k]/x[k+'_base'],np.nan)
    x.to_csv(H/'timecourse_pairs.csv',index=False)
    a=x.groupby(['coef','step']).agg(n=('MG_ratio','count'),MG_ratio=('MG_ratio','median'),J1_ratio=('J1_ratio','median'),J2_ratio=('J2_ratio','median')).reset_index()
    a.to_csv(H/'timecourse_summary.csv',index=False)
    print(a.to_string(index=False))
if __name__=='__main__':main()
