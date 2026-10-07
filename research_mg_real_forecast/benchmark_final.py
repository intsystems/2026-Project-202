from pathlib import Path
import json,pickle,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from generator_screen import forecast
from campaign import H,CONFIG,MODELS,feat
from actdim.estimator.mle import estimate
def main():
 with (H/'routing_v2_frozen.pkl').open('rb') as f:rules=pickle.load(f)
 selector=rules[(2,32)]['models']['MG'];fixed=rules[(2,32)]['fixed'];rows=[];timeseeds=[651,665,680]
 for seed in timeseeds:
  for arm in ['T1','T2','T3','T4','H2','H4','M4','chaos']:
   x=np.load(H/f'fresh_seed{seed}/{arm}.npz')['obs'][4096:6144,2]
   if x.std()<1e-10:continue
   def mg_route():
    r=estimate(x,CONFIG,seed=123);value=r.MG if not r.degenerate else np.nan;k=int(selector.predict([[value]]).argmin());return forecast(x,32,MODELS[k])
   def validation():
    scores=[]
    for model in MODELS:scores.append(np.mean([np.mean((forecast(x[:o],32,model)-x[o:o+32])**2) for o in range(1920,2017,32)]))
    return forecast(x,32,MODELS[int(np.argmin(scores))])
   funcs={'MG_route':mg_route,'validation':validation,'fixed':lambda:forecast(x,32,MODELS[fixed]),
          'MG_feature':lambda:estimate(x,CONFIG,seed=123)}
   for fn in funcs.values():fn()
   for repeat in range(5):
    for name in np.random.default_rng(repeat).permutation(list(funcs)):
     tic=time.perf_counter();funcs[name]();rows.append(dict(seed=seed,arm=arm,repeat=repeat,method=name,seconds=time.perf_counter()-tic))
 d=pd.DataFrame(rows);d.to_csv(H/'final_timing.csv',index=False);p=d.groupby(['seed','arm','method']).seconds.median().unstack();p['speedup']=p.validation/p.MG_route;p.to_csv(H/'final_timing_paired.csv');print(p.round(4).to_string());print('Median speedup',p.speedup.median())
 (H/'final_timing_protocol.json').write_text(json.dumps(dict(seeds=timeseeds,replicates=5,threads=1,excludes_disk=True,includes_selected_forecaster_fit=True,validation_fits=33,MG_fits=1,meta_selector_training_excluded=True),indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
