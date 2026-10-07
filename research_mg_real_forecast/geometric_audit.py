from routing_v2 import H,load,model,MODELS,CHEAP,VAL
from campaign import CONFIG
import sys,json,time,pickle
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from actdim.estimator.mle import estimate
from actdim.estimator.embedding import reconstruct
EXTRA=['LB','TwoNN','PRdelay','sample_entropy','permutation_entropy']
SETS={k:[k] for k in EXTRA};SETS['all_nonMG']=CHEAP+VAL+EXTRA;SETS['all_withMG']=CHEAP+VAL+EXTRA+['MG']
def extract(seeds):
 rows=[]
 for seed in seeds:
  for arm in ['T1','T2','T3','T4','H2','H4','M4','chaos']:
   x=np.load(H/f'fresh_seed{seed}/{arm}.npz')['obs'][4096:6144,2];row=dict(seed=seed,arm=arm)
   if x.std()<1e-10:row.update({k:np.nan for k in EXTRA})
   else:
    tic=time.perf_counter();r=estimate(x,CONFIG,seed=123);row.update(LB=r.LB if not r.degenerate else np.nan,TwoNN=r.TwoNN if not r.degenerate else np.nan,dimension_seconds=time.perf_counter()-tic)
    rec=reconstruct(x,CONFIG,seed=123);ev=np.maximum(0,np.linalg.eigvalsh(np.cov(rec.points.T)));row['PRdelay']=ev.sum()**2/max(np.sum(ev**2),1e-30)
    z=(x-x.mean())/x.std();pat=np.argsort(np.lib.stride_tricks.sliding_window_view(z,5),axis=1,kind='stable');_,c=np.unique(pat,axis=0,return_counts=True);p=c/c.sum();row['permutation_entropy']=-np.sum(p*np.log(p))/np.log(120)
    a=np.lib.stride_tricks.sliding_window_view(z,3);counts=[]
    for b in [a[:,:2],a]:counts.append(int(np.sum(cKDTree(b).query_ball_point(b,.2,p=np.inf,return_length=True))-len(b)))
    row['sample_entropy']=-np.log(counts[1]/counts[0]) if min(counts)>0 else np.nan
   rows.append(row)
 print('extra',min(seeds),max(seeds),flush=True);return pd.DataFrame(rows)
def get(seeds):
 p=H/f'geometry_{min(seeds)}_{max(seeds)}.csv'
 if not p.exists():extract(seeds).to_csv(p,index=False)
 d=load(seeds).query('neuron==2 and horizon==32');return d.merge(pd.read_csv(p),on=['seed','arm'],validate='one_to_one')
def main():
 if '--test-only' in sys.argv:
  with (H/'geometry_frozen.pkl').open('rb') as f:saved=pickle.load(f)
  test=get(range(651,681));Y=test[['loss_'+m for m in MODELS]].to_numpy();rows=[]
  for name,m in saved.items():
   ids=m.predict(test[SETS[name]].to_numpy()).argmin(1);ids[test['std'].to_numpy()<1e-10]=0
   for i,k in enumerate(ids):rows.append(dict(seed=int(test.seed.iloc[i]),arm=test.arm.iloc[i],method=name,model=MODELS[k],error=Y[i,k]))
  pd.DataFrame(rows).to_csv(H/'geometry_final.csv',index=False);print(pd.DataFrame(rows).groupby('method').error.mean());return
 d=get(range(601,619));Y=d[['loss_'+m for m in MODELS]].to_numpy();saved={};cvrows=[]
 for name,cols in SETS.items():
  errors=[]
  for i in range(4):
   e=[]
   for tr,te in GroupKFold(3).split(d,Y,d.seed):
    m=model(i).fit(d.iloc[tr][cols].to_numpy(),Y[tr]);ids=m.predict(d.iloc[te][cols].to_numpy()).argmin(1);e.extend(Y[te,ids])
   errors.append(np.mean(e));cvrows.append(dict(method=name,classifier=i,error=errors[-1]))
  saved[name]=model(int(np.argmin(errors))).fit(d[cols].to_numpy(),Y)
 with (H/'geometry_frozen.pkl').open('wb') as f:pickle.dump(saved,f)
 pd.DataFrame(cvrows).to_csv(H/'geometry_cv.csv',index=False)
 test=get(range(631,651));Y=test[['loss_'+m for m in MODELS]].to_numpy();rows=[]
 for name,m in saved.items():
  ids=m.predict(test[SETS[name]].to_numpy()).argmin(1);ids[test['std'].to_numpy()<1e-10]=0
  for i,k in enumerate(ids):rows.append(dict(seed=int(test.seed.iloc[i]),arm=test.arm.iloc[i],method=name,model=MODELS[k],error=Y[i,k]))
 result=pd.DataFrame(rows);result.to_csv(H/'geometry_test.csv',index=False);print(result.groupby('method').error.mean())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
