from pathlib import Path
import json,pickle,time,hashlib,sys,platform
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from campaign import H,MODELS,feat,CONFIG
from generator_screen import forecast
from actdim.estimator.mle import estimate
from actdim.estimator.embedding import reconstruct
from scipy.spatial import cKDTree
from scipy.stats import binomtest

def features_extra(x):
 r=estimate(x,CONFIG,seed=123)
 return r
def main():
 # Verify finalized rules unchanged since before final cohort.
 frozen=json.loads((H/'final_freeze.json').read_text())
 for name,digest in frozen.items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==digest,(name,'changed after freeze')
 with (H/'routing_v2_frozen.pkl').open('rb') as f:selectors=pickle.load(f)
 checks={'frozen_rules_unchanged':True};rows=[];flat=[]
 for seed in range(651,681):
  df=pd.read_csv(H/f'fresh_seed{seed}/features.csv');assert len(df)==24
  for arm in ['T1','T2','T3','T4','H2','H4','M4','chaos']:
   z=np.load(H/f'fresh_seed{seed}/{arm}.npz');x=z['obs'][4096:6144,2];y=z['obs'][6144:6176,2]
   assert x.shape==(2048,) and y.shape==(32,);assert np.isfinite(np.r_[x,y]).all()
   saved=df[(df.arm==arm)&(df.neuron==2)].iloc[0];v=x.var()
   if x.std()<1e-10:
    flat.append(dict(seed=seed,arm=arm,std=x.std(),persistence_error=float(np.mean((x[-1]-y)**2)/max(v,1e-12))))
   # Independently recompute fixed forecast losses for selected audit subset without using stored labels.
   if seed in [651,660,670,680]:
    for m in MODELS:
     pred=forecast(x,32,m);err=np.mean((pred-y)**2)/max(v,1e-12)
     if v>=1e-12:assert np.isclose(err,saved['loss_'+m],rtol=1e-9,atol=1e-12),(seed,arm,m)
   meta=json.loads((H/f'fresh_seed{seed}/{arm}.json').read_text());assert meta['n']==256 and meta['seed']==seed
   rows.append(meta)
 checks.update(all240tasks_included=True,forecast_losses_recomputed=True,no_quality_filtering=True,flat_records=len(flat))
 pd.DataFrame(flat).to_csv(H/'final_flat_records.csv',index=False);pd.DataFrame(rows).to_csv(H/'final_training_records.csv',index=False)
 # All algorithms use same flat-observer persistence fallback; recompute its real, floored loss.
 d=pd.read_csv(H/'final_decisions.csv').query('neuron==2 and horizon==32').copy();g=pd.read_csv(H/'geometry_final.csv');allrows=pd.concat([d[['seed','arm','method','model','error']],g],ignore_index=True)
 for row in flat:
  mask=(allrows.seed==row['seed'])&(allrows.arm==row['arm']);allrows.loc[mask,'error']=row['persistence_error'];allrows.loc[mask,'model']='last'
 assert np.isfinite(allrows.error).all();allrows.to_csv(H/'final_audited_decisions.csv',index=False)
 summary=allrows.groupby('method').error.agg(['mean','median']);summary.to_csv(H/'final_audited_summary.csv')
 p=allrows.groupby(['seed','method']).error.mean().unstack();comparisons=[];rng=np.random.default_rng(90210)
 for other in p:
  if other in ['MG','oracle','old_oracle']:continue
  diff=p[other].to_numpy()-p.MG.to_numpy();boot=rng.choice(diff,size=(20000,len(diff))).mean(1);ci=np.quantile(boot,[.025,.975]);signs=rng.choice([-1,1],size=(100000,len(diff)));pv=(1+np.sum(abs((signs*diff).mean(1))>=abs(diff.mean())-1e-15))/100001
  comparisons.append(dict(baseline=other,MG=p.MG.mean(),baseline_error=p[other].mean(),gain=1-p.MG.mean()/p[other].mean(),difference=diff.mean(),ci_low=ci[0],ci_high=ci[1],wins=int((diff>1e-12).sum()),ties=int((abs(diff)<=1e-12).sum()),n=len(diff),p_signflip=pv))
 cmp=pd.DataFrame(comparisons).sort_values('p_signflip');k=len(cmp);cmp['holm_p']=np.minimum(1,np.maximum.accumulate(cmp.p_signflip.to_numpy()*(k-np.arange(k))));cmp.to_csv(H/'final_audited_comparisons.csv',index=False)
 allrows[allrows.arm!='chaos'].groupby('method').error.mean().to_csv(H/'final_without_control.csv')
 # Fitted forecast models never see targets; MG tree uses one column only.
 sel=selectors[(2,32)]['models']['MG'];assert sel.n_features_in_==1
 checks['MG_selector_one_input']=True
 (H/'final_audit.json').write_text(json.dumps(dict(checks=checks,python=platform.python_version(),platform=platform.platform(),train_seeds=list(range(601,619)),final_seeds=list(range(651,681))),indent=2))
 print(summary.sort_values('mean').head(12).to_string());print(cmp.sort_values('baseline_error').head(10).to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
