from query_budget import H,sampled,reconstruct,CONFIG,estimate
from generator_screen import forecast,MODELS
import hashlib,json,pickle
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
def main():
 freeze=json.loads((H/'query_freeze.json').read_text())
 for name,value in freeze.items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==value
 full=pd.read_csv(H/'query_fresh_comparisons.csv').query('method=="MG"');sample=pd.read_csv(H/'query_fresh.csv').query('q==128')
 pair=full.merge(sample,on=['seed','arm'],suffixes=('_full','_query'),validate='one_to_one');assert len(pair)==160
 assert (pair.model_full==pair.model_query).all();assert np.allclose(pair.error_full,pair.error_query,rtol=1e-12,atol=1e-15)
 with (H/'routing_v2_frozen.pkl').open('rb') as f:rule=pickle.load(f)[(2,32)]['models']['MG']
 checks=0
 for seed in [681,690,700]:
  for arm in ['T1','T2','M4','T4']:
   raw=np.load(H/f'fresh_seed{seed}/{arm}.npz')['obs'];x=raw[4096:6144,2]
   a=sampled(x,128,'tree');b=sampled(x,128,'blocked');assert np.isclose(a,b,rtol=1e-10), (seed,arm,a,b)
   r=estimate(x,CONFIG,seed=123);allq=sampled(x,10000);assert np.isclose(r.MG,allq,rtol=1e-10)
   c=sampled(3*x+7,128);assert np.isclose(c,b,rtol=1e-5)
   idx=int(rule.predict([[b]]).argmin());saved=pair[(pair.seed==seed)&(pair.arm==arm)].iloc[0];assert MODELS[idx]==saved.model_query
   checks+=1
 assert np.isnan(sampled(np.ones(2048),128));assert np.isnan(sampled(np.full(2048,np.nan),128))
 (H/'query_audit.json').write_text(json.dumps(dict(frozen_hashes_match=True,n_forecasts=len(pair),same_decisions=int((pair.model_full==pair.model_query).sum()),same_errors=True,tree_block_distance_checks=checks,full_query_exact_checks=checks,affine_checks=checks,constant_and_nonfinite_rejected=True,scope='Agreement on these records, not guaranteed for arbitrary systems.'),indent=2))
 print('Query approximation audit passed:',len(pair),'matched decisions,',checks,'kernel/reconstruction checks')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
