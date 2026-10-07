from pathlib import Path
import sys,json,time,argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent; OLD=H.parent/'research_walker_repair'
sys.path.insert(0,str(OLD))
from motion import reduced,Actor
from features import measure

def period_info(qp,qv):
    x=reduced(qp,qv)
    def calc(x):
        var=np.mean(np.sum((x-x.mean(0))**2,axis=1))
        e=np.array([np.mean(np.sum((x[p:]-x[:-p])**2,axis=1))/(2*max(var,1e-30)) for p in range(20,251)])
        candidates=find_peaks(-e)[0]; bound=1.1*e.min()+.005
        good=candidates[e[candidates]<=bound]
        chosen=int(good[0]+20) if len(good) else int(e.argmin()+20)
        return chosen,e,candidates
    p,e,c=calc(x);halves=[calc(y)[0] for y in np.array_split(x,2)]
    return dict(period=p,global_period=int(e.argmin()+20),halves=halves,
        stable=bool(max(abs(z/p-1) for z in halves)<=.2),
        minima=[dict(lag=int(i+20),error=float(e[i])) for i in c],
        error_selected=float(e[p-20]),error_global=float(e.min())),e

def mg_settings(data,settings=('fixed','cycles','resampled','left')):
    pi,_=period_info(data['qpos'],data['qvel']);p=pi['period'];rows=[]
    for mode in settings:
        w=2048 if mode in ('fixed','left') else 14*p
        tau=8 if mode in ('fixed','left') else max(1,round(p/19))
        signal=data['qpos'][:,7 if mode=='left' else 4]
        if len(signal)<w:
            rows.append(dict(mode=mode,end=-1,MG=np.nan,degenerate=True,period=p,window=w,tau=tau,error='short record'));continue
        for end in np.unique(np.linspace(w,len(signal),3).astype(int)):
            part=signal[end-w:end]
            if mode=='resampled':part=np.interp(np.linspace(0,w-1,1792),np.arange(w),part);tau=7
            rows.append(dict(mode=mode,end=int(end),period=p,period_stable=pi['stable'],window=len(part),native_window=w,tau=tau,**measure(part,len(part),tau)))
    return pd.DataFrame(rows)

def run_old():
    out=H/'diagnostic';out.mkdir(exist_ok=True);rows=[];summaries=[]
    for seed in range(211,216):
        for step in [0,1048576]:
            for path in sorted((OLD/f'seed{seed}'/f'step{step:07d}').glob('reset51*/metrics.json')):
                m=json.loads(path.read_text())
                if not m['eligible']:continue
                reset=m['reset'];label=f'{"anchor" if step==0 else "seed"+str(seed)}_r{reset}'
                d=np.load(path.parent/'trajectory.npz');pi,e=period_info(d['qpos'],d['qvel'])
                (out/f'{label}_period.json').write_text(json.dumps(pi,indent=2))
                rows.append(dict(seed=seed,step=step,reset=reset,**{k:v for k,v in pi.items() if k not in ('minima','halves')},half1=pi['halves'][0],half2=pi['halves'][1]))
                cached=out/f'{label}_MG.csv'
                if not cached.exists():mg_settings(d,('fixed','cycles','resampled')).to_csv(cached,index=False)
                df=pd.read_csv(cached)
                if 'fixed' not in set(df['mode']):
                    df=pd.concat([df,mg_settings(d,('fixed',))],ignore_index=True);df.to_csv(cached,index=False)
                for mode,g in df.groupby('mode'):
                    ok=np.isfinite(g.MG)&~g.degenerate
                    summaries.append(dict(seed=seed,step=step,reset=reset,mode=mode,valid=bool(ok.all()),MG=float(g.loc[ok,'MG'].median()),ident_max=float(g.ident.max())))
        print('DIAGNOSTIC',seed,flush=True)
    pd.DataFrame(rows).to_csv(out/'periods.csv',index=False);pd.DataFrame(summaries).to_csv(out/'MG_summary.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(limits=1):run_old()
