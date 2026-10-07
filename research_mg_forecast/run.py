from pathlib import Path
import json,sys,time,argparse
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG=EstimatorConfig(max_E=20,tau='acorr',k_neighbors=20,theiler='autocorr',theiler_cap=320)
ARMS=['T1','T2','T3','T4','H2','H4','M4','chaos']
def period(x):
    tail=x[600:];origin=x[:-600];v=np.var(x)
    errs=[np.mean((x[p:]-x[:-p])**2)/v for p in range(30,601)]
    p=30+np.argmin(errs)
    def loss(lag):return np.mean((np.interp(np.arange(len(origin))+lag,np.arange(len(x)),x)-origin)**2)/v
    fit=minimize_scalar(loss,bounds=(max(30,p-1),min(600,p+1)),method='bounded')
    return fit.x,fit.fun
def periodic(x,horizon):
    p,_=period(x)
    def design(t):
        ph=2*np.pi*np.outer(t,np.arange(1,9))/p
        return np.column_stack([np.ones(len(t)),np.sin(ph),np.cos(ph)])
    coef=np.linalg.lstsq(design(np.arange(len(x))),x,rcond=None)[0]
    return design(np.arange(len(x),len(x)+horizon))@coef
def ar(x,horizon):
    lag=64;win=np.lib.stride_tricks.sliding_window_view(x,lag+1)
    X=win[:,:lag];y=win[:,lag];g=X.T@X
    w=np.linalg.solve(g+np.eye(lag)*1e-3*np.trace(g)/lag,X.T@y)
    z=list(x)
    for _ in range(horizon):
        val=float(np.dot(z[-lag:],w))
        if not np.isfinite(val) or abs(val)>1e5:return np.full(horizon,np.nan)
        z.append(val)
    return np.asarray(z[-horizon:])
def loss(pred,y):
    return min(1000.,float(np.mean((pred-y)**2))) if np.isfinite(pred).all() else 1000.
def features(x):
    r=estimate(x,CFG,seed=123);p=np.abs(np.fft.rfft(x-x.mean()))[1:]**2;p/=p.sum()
    return dict(MG=r.MG if not r.degenerate else np.nan,
        entropy=-float(np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p))),
        increments=float(np.mean(np.diff(x)**2)/(2*np.var(x))),recurrence=float(period(x)[1]))
def collect(seeds):
    rows=[]
    for seed in seeds:
        for arm in ARMS:
            raw=np.load(R/f'research_generator/results/obs_{arm}_s{seed}.npz')['obs'][:,0]
            x=(raw[:4096]-raw[:4096].mean())/raw[:4096].std();y=(raw[4096:4608]-raw[:4096].mean())/raw[:4096].std()
            row=dict(seed=seed,arm=arm)
            tic=time.perf_counter();row.update(features(x));row['features_seconds']=time.perf_counter()-tic
            for name,fn in [('periodic',periodic),('ar',ar)]:
                tic=time.perf_counter();pred=fn(x,512);row[name+'_seconds']=time.perf_counter()-tic
                row[name]=loss(pred,y);row[name+'_valid']=bool(np.isfinite(pred).all())
                row[name+'_holdout']=loss(fn(x[:-512],512),x[-512:])
            rows.append(row);print(seed,arm,round(row['periodic'],3),round(row['ar'],3),flush=True)
    return pd.DataFrame(rows)
def fit(d):
    rules={}
    for feat in ['MG','entropy','increments','recurrence']:
        vals=np.sort(d[feat].dropna().unique());thr=np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf]
        candidates=[]
        for sign in [1,-1]:
            for t in thr:
                choose=(d[feat]<=t) if sign==1 else (d[feat]>t)
                score=np.where(choose,d.periodic,d.ar).mean()
                candidates.append((score,dict(feature=feat,threshold=float(t),sign=sign)))
        rules[feat]=min(candidates,key=lambda r:r[0])[1]
    return rules
def main():
    pilot=collect([1,2]);pilot.to_csv(H/'pilot.csv',index=False)
    rules=fit(pilot);(H/'rules.json').write_text(json.dumps(rules,indent=2))
    test=collect([3,4,5]);test.to_csv(H/'confirmation.csv',index=False)
    rows=[]
    for split,d in [('pilot',pilot),('confirmation',test)]:
        for row in d.to_dict('records'):
            choose={key:('periodic' if ((row[key]<=rule['threshold']) if rule['sign']==1 else (row[key]>rule['threshold'])) else 'ar') for key,rule in rules.items()}
            choose.update(fixed_ar='ar',fixed_periodic='periodic',holdout='periodic' if row['periodic_holdout']<row['ar_holdout'] else 'ar',oracle='periodic' if row['periodic']<row['ar'] else 'ar')
            for method,model in choose.items():rows.append(dict(split=split,seed=row['seed'],arm=row['arm'],method=method,model=model,error=row[model]))
    records=pd.DataFrame(rows);records.to_csv(H/'decisions.csv',index=False)
    summary=records.groupby(['split','method']).error.agg(['mean','median']);summary.to_csv(H/'summary.csv');print(summary)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
