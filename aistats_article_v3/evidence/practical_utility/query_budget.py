from campaign import H,CONFIG,MODELS
from generator_screen import forecast
from actdim.estimator.mle import estimate
from actdim.estimator.embedding import reconstruct
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import sys,json,pickle,time
import numpy as np,pandas as pd
def sampled(x,q=128,backend='blocked'):
 rec=reconstruct(x,CONFIG,seed=123)
 if not rec.usable:return np.nan
 X=rec.points;n=len(X);k=CONFIG.k_neighbors;T=rec.theiler
 if n-(2*T+1)<k:return np.nan
 ids=np.unique(np.linspace(0,n-1,min(n,q)).astype(int))
 if backend=='tree':
  tree=KDTree(X);dist,idx=tree.query(X[ids],k=k+2*T+1)
  valid=abs(idx-ids[:,None])>T;order=np.argsort(~valid,axis=1,kind='stable');dist=np.take_along_axis(dist,order,axis=1)[:,:k]
 else:
  chunks=[]
  for start in range(0,len(ids),64):
   selected=ids[start:start+64];dd=cdist(X[selected],X,metric='sqeuclidean');dd[abs(selected[:,None]-np.arange(n)[None,:])<=T]=np.inf
   nearest=np.sort(np.partition(dd,k-1,axis=1)[:,:k],axis=1);chunks.append(np.sqrt(nearest))
  dist=np.concatenate(chunks,axis=0)
 dist=np.maximum(dist,CONFIG.floor_distance);sums=np.sum(np.log(dist[:,-1:]/dist[:,:-1]),axis=1)
 if np.mean(dist<=CONFIG.floor_distance*1.000001)>.01 or np.mean(sums<=CONFIG.floor_ratio_sum)>.01:return np.nan
 return (len(ids)*(k-1)-1)/np.maximum(sums,CONFIG.floor_ratio_sum).sum()
def main():
 seeds=range(681,701) if '--fresh' in sys.argv else range(651,681);tag='query_fresh' if '--fresh' in sys.argv else 'query_development'
 with (H/'routing_v2_frozen.pkl').open('rb') as f:rule=pickle.load(f)[(2,32)]['models']['MG']
 rows=[];times=[];checks=0
 for seed in seeds:
  d=pd.read_csv(H/f'fresh_seed{seed}/features.csv').query('neuron==2').set_index('arm')
  for arm,r in d.iterrows():
   z=np.load(H/f'fresh_seed{seed}/{arm}.npz')['obs'];x=z[4096:6144,2];y=z[6144:6176,2];flat=x.std()<1e-10
   if seed==min(seeds) and not flat:
    original=estimate(x,CONFIG,seed=123);full=sampled(x,10000)
    assert np.isclose(original.MG,full,rtol=1e-12) or (original.degenerate and not np.isfinite(full));checks+=1
   for q in [64,128,256]:
    tic=time.perf_counter();value=sampled(x,q) if not flat else np.nan;idx=int(rule.predict([[value]]).argmin()) if not flat else 0;forecast(x,32,MODELS[idx]);sec=time.perf_counter()-tic
    error=float(r['loss_'+MODELS[idx]]) if not flat else float(np.mean((x[-1]-y)**2)/max(x.var(),1e-12))
    rows.append(dict(seed=seed,arm=arm,q=q,MG=value,model=MODELS[idx],error=error,seconds=sec))
    if seed in [min(seeds),min(seeds)+10,max(seeds)] and not flat:
     for rep in range(5):
      tic=time.perf_counter();v=sampled(x,q);k=int(rule.predict([[v]]).argmin());forecast(x,32,MODELS[k]);times.append(dict(seed=seed,arm=arm,q=q,repeat=rep,seconds=time.perf_counter()-tic))
 pd.DataFrame(rows).to_csv(H/(tag+'.csv'),index=False);pd.DataFrame(times).to_csv(H/(tag+'_timing.csv'),index=False)
 print(pd.DataFrame(rows).groupby('q')[['error','seconds']].mean());print('exact_full_query_checks',checks)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
