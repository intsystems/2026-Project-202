"""Execute BASELINE_PROTOCOL.md on saved trajectories; no training."""
from pathlib import Path
import json, hashlib, platform, time, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
SEEDS=range(231,236)
COEFS=[0,.25,1,4]
ENDS=[2048,3072,4096]
METRICS=['MG','entropy','recurrence','increments']

def entropy(x):
    y=x-x.mean()
    if np.mean(y*y)<=1e-30:return np.nan
    p=np.abs(np.fft.rfft(y))[1:]**2
    p=p/p.sum();p=p[p>0]
    return float(-np.sum(p*np.log(p))/np.log(len(x)//2))

def recurrence(x, details=False):
    v=np.var(x)
    if v<=1e-30:return (np.nan,0) if details else np.nan
    e=np.array([np.mean((x[p:]-x[:-p])**2)/(2*v) for p in range(20,251)])
    j=int(e.argmin())
    return (float(e[j]),20+j) if details else float(e[j])

def increments(x):
    v=np.var(x)
    return float(np.mean(np.diff(x)**2)/(2*v)) if v>1e-30 else np.nan

def signals(a):
    return dict(action_norm=np.linalg.norm(a,axis=1),
                delta_action_norm=np.linalg.norm(np.diff(a,axis=0,prepend=a[:1]),axis=1),
                mean_action=a.mean(axis=1))

def aggregate(d, group):
    return d.groupby(group).agg(seeds=('ratio','count'),ratio=('ratio','median'),
        minimum=('ratio','min'),maximum=('ratio','max'),
        decrease=('ratio',lambda x:int((x<1-1e-10).sum())),
        increase=('ratio',lambda x:int((x>1+1e-10).sum()))).reset_index()

def analyze():
    rows=[]
    for seed in SEEDS:
      for coef in COEFS:
        folder=H/f'seed{seed}_lambda{coef:g}'
        tests=pd.read_csv(folder/'test.csv')
        for reset in tests.loc[tests.eligible,'reset'].astype(int):
          root=folder/'step1048576'/f'reset{reset}'
          with np.load(root/'trajectory.npz') as z:a=z['actions'].astype(float)
          assert a.shape==(4096,6)
          cached=pd.read_csv(root/'action_MG_windows.csv').set_index(['signal','end'])
          for name,x in signals(a).items():
            for end in ENDS:
              y=x[end-2048:end];m=cached.loc[(name,end)]
              r,p=recurrence(y,True)
              rows.append(dict(seed=seed,coef=coef,reset=reset,signal=name,end=end,
                MG=m.MG if not m.degenerate else np.nan,entropy=entropy(y),
                recurrence=r,lag=p,lag_boundary=p in [20,250],increments=increments(y)))
        print('BASELINES',seed,coef,flush=True)
    w=pd.DataFrame(rows);w.to_csv(H/'baselines_windows.csv',index=False)
    assert len(w)==169*3*3
    assert np.isfinite(w[METRICS]).all().all()
    rec=w.groupby(['seed','coef','reset','signal'])[METRICS].median().reset_index()
    rec.to_csv(H/'baselines_records.csv',index=False)
    b=rec[rec.coef==0].drop(columns='coef')
    paired=rec.merge(b,on=['seed','reset','signal'],suffixes=('','_control'),validate='many_to_one')
    out=[]
    for metric in METRICS:
        g=paired[['seed','coef','reset','signal']].copy()
        g['metric']=metric;g['ratio']=paired[metric]/paired[metric+'_control'];out.append(g)
    pairs=pd.concat(out,ignore_index=True)
    pairs.to_csv(H/'baselines_pairs.csv',index=False)
    seedrows=pairs.groupby(['seed','coef','signal','metric']).agg(n=('ratio','count'),ratio=('ratio','median')).reset_index()
    seedrows.to_csv(H/'baselines_by_seed.csv',index=False)
    agg=aggregate(seedrows,['coef','signal','metric']);agg.to_csv(H/'baselines_summary.csv',index=False)
    # Same eligible resets in all four arms, preventing composition changes.
    keys=rec.groupby(['seed','reset','signal']).coef.nunique()
    keys=keys[keys==4].reset_index()[['seed','reset','signal']]
    common=pairs.merge(keys,on=['seed','reset','signal'],validate='many_to_one')
    same=common.groupby(['seed','coef','signal','metric']).agg(n=('ratio','count'),ratio=('ratio','median')).reset_index()
    same.to_csv(H/'baselines_common4_by_seed.csv',index=False)
    aggregate(same,['coef','signal','metric']).to_csv(H/'baselines_common4_summary.csv',index=False)
    # Direct excessive versus moderate comparison, same triplets 0,1,4.
    common0=rec[rec.coef==0][['seed','reset','signal']]
    m=rec[rec.coef==4].merge(rec[rec.coef==1],on=['seed','reset','signal'],suffixes=('_4','_1'),validate='one_to_one').merge(common0,on=['seed','reset','signal'],validate='one_to_one')
    rr=[]
    for metric in METRICS:
      g=m[['seed','reset','signal']].copy();g['metric']=metric;g['ratio']=m[metric+'_4']/m[metric+'_1'];rr.append(g)
    rr=pd.concat(rr,ignore_index=True);rr.to_csv(H/'baselines_rebound_pairs.csv',index=False)
    ss=rr.groupby(['seed','signal','metric']).agg(n=('ratio','count'),ratio=('ratio','median')).reset_index()
    ss.to_csv(H/'baselines_rebound_by_seed.csv',index=False)
    aggregate(ss,['signal','metric']).to_csv(H/'baselines_rebound_summary.csv',index=False)
    print(agg[agg.signal=='action_norm'].to_string(index=False),flush=True)
    print('MATCHED REBOUND',aggregate(ss,['signal','metric']).to_string(index=False),flush=True)
    # Regression check against the previous cached MG aggregation.
    previous=pd.read_csv(H/'action_mg_by_seed.csv')
    cmp=seedrows[seedrows.metric=='MG'].merge(previous,on=['seed','coef','signal'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(cmp)==60 and np.allclose(cmp.ratio_new,cmp.ratio_old,rtol=1e-12)
    # Scale/translation invariance must hold for these normalized companions.
    x=np.sin(np.arange(2048)*.07)+.3*np.cos(np.arange(2048)*.123)
    for fn in [entropy,recurrence,increments]:assert np.isclose(fn(x),fn(3*x+10),rtol=1e-10)

def benchmark():
    sys.path.insert(0,str(H.parent/'code'))
    from actdim.estimator.config import EstimatorConfig
    from actdim.estimator.mle import estimate
    with np.load(H/'seed231_lambda0/step1048576/reset62001/trajectory.npz') as z:
        x=signals(z['actions'].astype(float))['action_norm'][-2048:]
    config=EstimatorConfig(window=2048,max_E=20,tau=8,k_neighbors=20,theiler=312,theiler_cap=312)
    funcs=dict(entropy=lambda:entropy(x),recurrence=lambda:recurrence(x),increments=lambda:increments(x),
        MG=lambda:estimate(x,config,seed=123),
        MG_with_E40=lambda:(estimate(x,config,seed=123),estimate(x,config.replace(max_E=40),seed=123)))
    for f in funcs.values():f()
    rows=[];rng=np.random.default_rng(921)
    for rep in range(7):
      for name in rng.permutation(list(funcs)):
        # Batching removes timer resolution overhead on fast O(N) baselines.
        batch=100 if name in ['entropy','increments'] else 1
        start=time.perf_counter()
        for _ in range(batch):funcs[name]()
        rows.append(dict(metric=name,repeat=rep,batch=batch,seconds=(time.perf_counter()-start)/batch))
    d=pd.DataFrame(rows);d.to_csv(H/'baselines_timing_raw.csv',index=False)
    summary=d.groupby('metric').seconds.agg(['median','min','max']).reset_index()
    summary.to_csv(H/'baselines_timing.csv',index=False);print(summary.to_string(index=False))
    (H/'baselines_provenance.json').write_text(json.dumps(dict(platform=platform.platform(),python=sys.version,threads=1,
        protocol_sha256=hashlib.sha256((H/'BASELINE_PROTOCOL.md').read_bytes()).hexdigest(),
        source='seed231_lambda0/step1048576/reset62001/trajectory.npz',window=2048,repeats=7,
        io_excluded=True,secondary_analysis=True),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):analyze();benchmark()
