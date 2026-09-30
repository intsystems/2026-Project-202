from pathlib import Path
import argparse,json,sys,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from reference import pr
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'code'))
from actdim.estimator.mle import estimate
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.surrogates import iaaft

def mg(x,tau=1):
    cfg=EstimatorConfig(max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,theiler_cap=39*tau,window=len(x))
    t=time.perf_counter();a=estimate(x,cfg,seed=123);t1=time.perf_counter()
    b=estimate(x,cfg.replace(max_E=40),seed=123);t2=time.perf_counter()
    return dict(MG=a.MG,MG40=b.MG,ident=b.MG/a.MG,degenerate=a.degenerate or b.degenerate,
        mg_seconds=t1-t,checks_seconds=t2-t)

def cheap(x):
    t=time.perf_counter();y=x-x.mean();p=np.abs(np.fft.rfft(y))[1:]**2;p/=max(p.sum(),1e-30)
    z=x-np.polyval(np.polyfit(np.arange(len(x)),x,1),np.arange(len(x)))
    return dict(mean=x.mean(),std=x.std(),entropy=-np.sum(p*np.log(np.maximum(p,1e-30)))/np.log(len(p)),
        crossings=np.mean(z[:-1]*z[1:]<0),acf1=np.corrcoef(y[:-1],y[1:])[0,1],cheap_seconds=time.perf_counter()-t)

def analyze(root):
    allrows=[];sensitivity=[];controls=[]
    for path in sorted(root.rglob('logs.csv')):
        d=path.parent;log=pd.read_csv(path);ref=pd.read_csv(d/'reference.csv').set_index('end')
        tag=str(d.relative_to(root)).replace('\\','/');meta=json.loads((d/'meta.json').read_text())
        for signal in ['loss','projection','grad_norm']:
            x=log[signal].to_numpy()
            for end in range(512,len(x)+1,256):
                seg=x[end-512:end]
                allrows.append(dict(run=tag,seed=meta['seed'],arm=meta['arm'],signal=signal,end=end,
                    **ref.loc[end].to_dict(),**mg(seg),**cheap(seg)))
            traj=np.load(d/'trajectory.npy',mmap_mode='r')
            for end in [2048,4096]:
                for w,tau in [(256,1),(1024,1),(512,4)]:
                    v,var=pr(traj[end-w:end]);small,_=pr(traj[end-w:end,::max(1,traj.shape[1]//128)])
                    sensitivity.append(dict(run=tag,signal=signal,end=end,window=w,tau=tau,pr=v,small_pr=small,**mg(x[end-w:end],tau)))
                seg=x[end-512:end];a=mg(seg)['MG'];b=mg(10*seg)['MG'];assert np.isclose(a,b,rtol=1e-7)
                vals=[mg(iaaft(seg,rng=np.random.default_rng(s),match=False))['MG'] for s in [8,9,10]]
                smooth=mg(np.convolve(seg,np.ones(8)/8,mode='valid'))['MG']
                controls.append(dict(run=tag,signal=signal,end=end,MG=a,scale=b,smooth=smooth,
                    surrogate=float(np.median(vals)),ratio=a/np.median(vals)))
        print('Analyzed',tag,flush=True)
        pd.DataFrame(allrows).to_csv(root/'windows.csv',index=False)
        pd.DataFrame(sensitivity).to_csv(root/'sensitivity.csv',index=False)
        pd.DataFrame(controls).to_csv(root/'controls.csv',index=False)
    df=pd.DataFrame(allrows)
    pre=df[df.end.between(1024,2048)].groupby(['run','signal']).median(numeric_only=True)
    post=df[df.end>=3072].groupby(['run','signal']).median(numeric_only=True)
    ratio=post[['pr','small_pr','MG','std','entropy','crossings']]/pre[['pr','small_pr','MG','std','entropy','crossings']]
    ratio.to_csv(root/'ratios.csv');print(ratio[['pr','MG']].to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):analyze(a.root)
