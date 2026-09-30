from pathlib import Path
import argparse,json,sys,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate

def measure(x,window,tau):
    config=EstimatorConfig(window=window,max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,theiler_cap=39*tau)
    started=time.perf_counter()
    try:
        a=estimate(x,config,seed=123);elapsed=time.perf_counter()-started
    except (ValueError,RuntimeError) as e:
        return dict(MG=None,degenerate=True,MG40=None,diagnostic_degenerate=True,ident=None,
            MG_seconds=time.perf_counter()-started,error=str(e))
    try:
        b=estimate(x,config.replace(max_E=40),seed=123)
        return dict(MG=a.MG,degenerate=a.degenerate,MG40=b.MG,diagnostic_degenerate=b.degenerate,
            ident=b.MG/a.MG if a.MG else None,MG_seconds=elapsed,error=None)
    except (ValueError,RuntimeError) as e:
        return dict(MG=a.MG,degenerate=a.degenerate,MG40=None,diagnostic_degenerate=True,ident=None,
            MG_seconds=elapsed,error='E40 diagnostic only: '+str(e))

def analyze(seed):
    selection=json.loads((H/'selection.json').read_text());tau=selection['tau'];rows=[]
    settings=list(dict.fromkeys([(2048,tau),(1024,tau),(4096,tau),(2048,max(1,tau//2)),(2048,2*tau)]))
    for path in sorted((H/f'seed{seed}').glob('step*/reset*/metrics.json')):
        metrics=json.loads(path.read_text())
        if not metrics['eligible']:continue
        root=path.parent;data=np.load(root/'trajectory.npz');output=root/'MG_windows.csv'
        if not output.exists():
            windows=[]
            for sensor,index in [('right_knee',4),('left_knee',7)]:
                for w,delay in settings:
                    if sensor=='left_knee' and (w,delay)!=(2048,tau):continue
                    signal=data['qpos'][:,index]
                    for end in range(w,len(signal)+1,512):
                        windows.append(dict(sensor=sensor,window=w,tau=delay,end=end,**measure(signal[end-w:end],w,delay)))
            pd.DataFrame(windows).to_csv(output,index=False)
        frame=pd.read_csv(output)
        for (sensor,w,delay),part in frame.groupby(['sensor','window','tau']):
            valid=np.isfinite(part.MG)&~part.degenerate
            rows.append(dict(seed=seed,step=int(root.parent.name[4:]),reset=int(root.name[5:]),sensor=sensor,
                window=int(w),tau=int(delay),all_valid=bool(valid.all()),valid_count=int(valid.sum()),window_count=len(part),
                MG=float(part.loc[valid,'MG'].median()) if valid.any() else None,
                ident_min=float(part.ident.min()) if part.ident.notna().any() else None,
                ident_max=float(part.ident.max()) if part.ident.notna().any() else None))
    pd.DataFrame(rows).to_csv(H/f'seed{seed}'/'MG_summary.csv',index=False)
    print('MG',seed,'traces/settings',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);a=p.parse_args()
    estimate(np.sin(np.arange(2048)/13)+np.cos(np.arange(2048)/7),EstimatorConfig(window=2048,max_E=20,tau=4,k_neighbors=20,theiler=156,theiler_cap=156),seed=123)
    with threadpool_limits(limits=1):analyze(a.seed)
