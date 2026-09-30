"""Frozen MG protocol and cheap baselines; no adjustment based on reference."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'code'))
from actdim.estimator.mle import estimate
from actdim.estimator.config import EstimatorConfig

def score(x,tau=1):
    cfg=EstimatorConfig(max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,theiler_cap=39*tau,window=len(x))
    t=time.perf_counter();a=estimate(x,cfg,seed=123);t1=time.perf_counter()
    b=estimate(x,cfg.replace(max_E=40),seed=123);t2=time.perf_counter()
    return dict(MG=a.MG,MG40=b.MG,ident=b.MG/a.MG if a.MG>0 else np.nan,
                degenerate=a.degenerate or b.degenerate,mg_seconds=t1-t,checks_seconds=t2-t)

def cheap(x):
    t=time.perf_counter();n=len(x);y=x-x.mean()
    power=np.abs(np.fft.rfft(y))[1:]**2;power/=max(power.sum(),1e-30)
    entropy=-np.sum(power*np.log(np.maximum(power,1e-30)))/np.log(len(power))
    detr=x-np.polyval(np.polyfit(np.arange(n),x,1),np.arange(n))
    return dict(mean=float(x.mean()),std=float(x.std()),entropy=float(entropy),
                acf1=float(np.corrcoef(y[:-1],y[1:])[0,1]),
                crossings=float(np.mean(np.sign(detr[:-1])*np.sign(detr[1:])<0)),cheap_seconds=time.perf_counter()-t)

def iaaft(x,seed,iterations=100):
    rng=np.random.default_rng(seed);r=rng.permutation(x);values=np.sort(x);amp=np.abs(np.fft.rfft(x))
    for _ in range(iterations):
        v=np.fft.irfft(amp*np.exp(1j*np.angle(np.fft.rfft(r))),n=len(x))
        order=np.argsort(v);r=np.empty(len(x));r[order]=values
    return r

def analyze(root):
    rows=[];sensitivity=[];controls=[]
    for d in sorted(root.iterdir()):
        if not (d/'logs.csv').exists():continue
        log=pd.read_csv(d/'logs.csv');ref=pd.read_csv(d/'reference.csv')
        meta=__import__('json').loads((d/'metadata.json').read_text())
        endpoints=ref.step.to_numpy(dtype=int);switch=meta['switch']
        for signal in ['g_loss','d_loss']:
            x=log[signal].to_numpy()
            for end in endpoints[endpoints>=512]:
                y=x[end-512:end];v=score(y);q=cheap(y)
                row=dict(run=d.name,arm=meta['arm'],seed=meta['seed'],step=end,signal=signal,
                         **v,**q);rows.append(row)
            for end in [switch,len(x)]:
                if end>len(x):continue
                for w,tau in [(256,1),(1024,1),(512,4)]:
                    if end<w:continue
                    sensitivity.append(dict(run=d.name,step=end,signal=signal,window=w,tau=tau,**score(x[end-w:end],tau)))
        for end in [switch,len(log)]:
            if end>len(log):continue
            x=log.g_loss.to_numpy()[end-512:end]
            baseline=score(x)['MG'];scaled=score(10*x)['MG'];assert np.isclose(baseline,scaled,atol=1e-8)
            smooth=np.convolve(x,np.ones(8)/8,mode='valid')
            surrogate=[score(iaaft(x,seed))['MG'] for seed in [10,11,12]]
            controls.append(dict(run=d.name,step=end,MG=baseline,scaled=scaled,smoothed=score(smooth)['MG'],
                                 surrogate=float(np.median(surrogate)),surrogate_ratio=baseline/np.median(surrogate)))
        print('Analyzed',d.name,flush=True)
    pd.DataFrame(rows).to_csv(root/'mg_windows.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(root/'sensitivity.csv',index=False)
    pd.DataFrame(controls).to_csv(root/'controls.csv',index=False)
    return pd.DataFrame(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=H/'mnist_pilot');a=p.parse_args()
    with threadpool_limits(limits=1):analyze(a.root)
