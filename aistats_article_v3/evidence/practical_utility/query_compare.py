from query_budget import H,sampled,MODELS,forecast,CONFIG,estimate
from routing_v2 import load,SETS,VAL
from geometric_audit import get,SETS as GEOMSETS
import numpy as np,pandas as pd,pickle,json,time
from threadpoolctl import threadpool_limits
def main():
 d=get(range(681,701)).reset_index(drop=True);rows=[]
 with (H/'routing_v2_frozen.pkl').open('rb') as f:models=pickle.load(f)[(2,32)]
 with (H/'geometry_frozen.pkl').open('rb') as f:gm=pickle.load(f)
 candidates={**{name:(m,SETS[name]) for name,m in models['models'].items()},**{name:(m,GEOMSETS[name]) for name,m in gm.items()}}
 Y=d[['loss_'+m for m in MODELS]].to_numpy();choices={name:m.predict(d[cols].to_numpy()).argmin(1) for name,(m,cols) in candidates.items()}
 choices.update(validation=d[VAL].to_numpy().argmin(1),fixed=np.full(len(d),models['fixed']),oracle=Y.argmin(1))
 for i,r in d.iterrows():
  z=np.load(H/f'fresh_seed{int(r.seed)}/{r.arm}.npz')['obs'];x=z[4096:6144,2];y=z[6144:6176,2];flat=x.std()<1e-10
  for method,ids in choices.items():
   k=0 if flat else int(ids[i]);err=float(np.mean((x[-1]-y)**2)/max(x.var(),1e-12)) if flat else Y[i,k]
   rows.append(dict(seed=int(r.seed),arm=r.arm,method=method,model=MODELS[k],error=err))
 for r in pd.read_csv(H/'query_fresh.csv').itertuples():rows.append(dict(seed=r.seed,arm=r.arm,method=f'MG_q{r.q}',model=r.model,error=r.error))
 a=pd.DataFrame(rows);a.to_csv(H/'query_fresh_comparisons.csv',index=False);summary=a.groupby('method').error.agg(['mean','median']);summary.to_csv(H/'query_fresh_summary.csv');print(summary.sort_values('mean').round(5).to_string())
 # Repeated interleaved end-to-end benchmark with identical inputs and classifier, one thread.
 tim=[]
 for seed in [681,690,700]:
  for arm in ['T1','T2','H4','T4']:
   x=np.load(H/f'fresh_seed{seed}/{arm}.npz')['obs'][4096:6144,2];rule=models['models']['MG']
   def route(q):
    value=estimate(x,CONFIG,seed=123).MG if q is None else sampled(x,q)
    k=int(rule.predict([[value]]).argmin());return forecast(x,32,MODELS[k])
   def validation():
    losses=[np.mean([np.mean((forecast(x[:o],32,m)-x[o:o+32])**2) for o in range(1920,2017,32)]) for m in MODELS]
    return forecast(x,32,MODELS[int(np.argmin(losses))])
   funcs={'MG':lambda:route(None),'MG_q128':lambda:route(128),'MG_q64':lambda:route(64),'validation':validation}
   for fn in funcs.values():fn()
   for rep in range(7):
    for name in np.random.default_rng(rep).permutation(list(funcs)):
     tic=time.perf_counter();funcs[name]();tim.append(dict(seed=seed,arm=arm,rep=rep,method=name,seconds=time.perf_counter()-tic))
 t=pd.DataFrame(tim);t.to_csv(H/'query_paired_timing.csv',index=False);p=t.groupby(['seed','arm','method']).seconds.median().unstack();p['speedup_vs_full']=p.MG/p.MG_q128;p['speedup_vs_validation']=p.validation/p.MG_q128;p.to_csv(H/'query_paired_summary.csv');print(p.median())
 p=a.groupby(['seed','method']).error.mean().unstack();rng=np.random.default_rng(811);stats=[]
 for baseline in ['MG','validation','cheap_val','TwoNN','all_nonMG']:
  diff=p[baseline]-p.MG_q128;ci=np.quantile(rng.choice(diff.to_numpy(),size=(20000,len(diff))).mean(1),[.025,.975]);stats.append(dict(baseline=baseline,mean_improvement=diff.mean(),ci_low=ci[0],ci_high=ci[1],n=len(diff),wins=int((diff>1e-12).sum()),ties=int((abs(diff)<=1e-12).sum())))
 pd.DataFrame(stats).to_csv(H/'query_intervals.csv',index=False)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
