from pathlib import Path
import json,time,sys,zipfile,io,pickle,hashlib
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_approximation import RBFSampler
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
COLS=['T1','T3','T6','RH_1','RH_3','RH_6'];MODELS=['last','daily','mean','ridge8','ridge48','knn8','knn48','rff48']
CHEAP=['entropy','increments','lag1','recurrence','peak','trend','std','level'];SETS={f:[f] for f in ['MG']+CHEAP};SETS.update(cheap=CHEAP,cheap_MG=CHEAP+['MG'])
CONFIG=EstimatorConfig(max_E=20,tau='acorr',k_neighbors=20,theiler='autocorr',theiler_cap=150)
def load():
 with zipfile.ZipFile(H/'energy.raw') as z:return pd.read_csv(io.BytesIO(z.read('energydata_complete.csv')))
def feat(x):
 row={};tic=time.perf_counter();r=estimate(x,CONFIG,seed=123);row['MG']=float(r.MG) if not r.degenerate else np.nan;row['MG_sec']=time.perf_counter()-tic
 tic=time.perf_counter();z=(x-x.mean())/x.std();p=np.abs(np.fft.rfft(z))[1:]**2;p/=p.sum()
 row.update(entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p))),increments=float(np.mean(np.diff(z)**2)/2),
 lag1=float(np.corrcoef(z[:-1],z[1:])[0,1]),recurrence=float(min(np.mean((z[l:]-z[:-l])**2)/2 for l in range(6,289))),peak=float(p.max()),
 trend=float(np.polyfit(np.linspace(-1,1,len(z)),z,1)[0]),std=float(x.std()),level=float(x.mean()))
 row['cheap_sec']=time.perf_counter()-tic;return row
def predict(x,h,model):
 if model=='last':return np.full(h,x[-1])
 if model=='daily':return x[-144:][:h].copy()
 if model=='mean':return np.full(h,x.mean())
 lag=int(''.join(c for c in model if c.isdigit()));mu=x.mean();sd=max(x.std(),1e-12);z=(x-mu)/sd
 w=np.lib.stride_tricks.sliding_window_view(z,lag+h);X=w[:,:lag];Y=w[:,lag:];q=z[-lag:][None]
 if model.startswith('knn'):
  dist=np.sum((X-q)**2,axis=1);ids=np.argsort(dist)[:5];pred=Y[ids].mean(0)
 else:
  if model.startswith('rff'):
   rbf=RBFSampler(gamma=1/lag,n_components=64,random_state=42);X=rbf.fit_transform(X);q=rbf.transform(q)
  X=np.column_stack([X,np.ones(len(X))]);q=np.column_stack([q,np.ones(len(q))]);reg=np.eye(X.shape[1]);reg[-1,-1]=1e-8
  beta=np.linalg.solve(X.T@X+reg,X.T@Y);pred=(q@beta)[0]
 return np.clip(pred,-10,10)*sd+mu
def collect(channel,h,split):
 path=H/f'bank_{channel}_{h}_{split}.csv'
 if path.exists():return pd.read_csv(path)
 x=load()[channel].to_numpy(float);mid=len(x)//2;rows=[]
 for origin in range(1024,len(x)-h,144):
  if split=='dev' and origin+h>mid:continue
  if split=='test' and origin-1024<mid:continue
  pref=x[origin-1024:origin];future=x[origin:origin+h];v=pref.var();row=dict(channel=channel,horizon=h,origin=origin,**feat(pref))
  if v<1e-12:continue
  for model in MODELS:
   tic=time.perf_counter();pred=predict(pref,h,model);row['cost_'+model]=time.perf_counter()-tic
   row['loss_'+model]=float(np.mean((pred-future)**2)/v)
   tic=time.perf_counter();vl=[]
   for o in range(896,1024-h+1,h):vl.append(np.mean((predict(pref[:o],h,model)-pref[o:o+h])**2)/v)
   row['val_'+model]=float(np.mean(vl));row['vcost_'+model]=time.perf_counter()-tic
  rows.append(row)
 d=pd.DataFrame(rows);d.to_csv(path,index=False);print('bank',channel,h,split,len(d),flush=True);return d
def fit(d,cols):
 X=d[cols];med=X.median().fillna(0);a=X.fillna(med).to_numpy();a=np.column_stack([a,X.isna().any(axis=1)])
 model=DecisionTreeRegressor(max_depth=2,min_samples_leaf=8,random_state=42).fit(a,d[['loss_'+m for m in MODELS]])
 return model,med
def decisions(d,models,fixed):
 losses=d[['loss_'+m for m in MODELS]].to_numpy();rows=[]
 choices={'fixed':np.full(len(d),fixed),'validation':d[['val_'+m for m in MODELS]].to_numpy().argmin(1)}
 for name,(model,med) in models.items():
  X=d[SETS[name]];a=np.column_stack([X.fillna(med).to_numpy(),X.isna().any(axis=1)]);choices[name]=model.predict(a).argmin(1)
 for m in MODELS:choices[m]=np.full(len(d),MODELS.index(m))
 choices['oracle']=losses.argmin(1)
 for name,ids in choices.items():
  for i,k in enumerate(ids):rows.append(dict(channel=d.channel.iloc[i],horizon=int(d.horizon.iloc[i]),origin=int(d.origin.iloc[i]),method=name,model=MODELS[k],loss=losses[i,k]))
 return pd.DataFrame(rows)
def develop():
 rows=[];frozen={}
 for ch in COLS:
  for h in [6,24]:
   d=collect(ch,h,'dev');cut=int(len(d)*2/3);valid=d.iloc[cut:];train=d[d.origin+1024+h<valid.origin.min()]
   models={name:fit(train,cols) for name,cols in SETS.items()};fixed=int(train[['loss_'+m for m in MODELS]].mean().argmin())
   v=decisions(valid,models,fixed);rows.append(v)
   frozen[ch+'_'+str(h)]=({name:fit(d,cols) for name,cols in SETS.items()},int(d[['loss_'+m for m in MODELS]].mean().argmin()))
 pd.concat(rows).to_csv(H/'development_decisions.csv',index=False)
 with (H/'frozen_models.pkl').open('wb') as f:pickle.dump(frozen,f)
 avg=pd.concat(rows).groupby(['channel','horizon','method']).loss.mean().unstack()
 baseline=[x for x in avg.columns if x not in ['MG','cheap_MG','oracle']]
 avg['gain_MG']=1-avg.MG/avg[baseline].min(1);avg['gain_augmented']=1-avg.cheap_MG/avg[baseline].min(1)
 avg.to_csv(H/'development_summary.csv');print(avg[['MG','cheap_MG','validation','gain_MG','gain_augmented']].round(3))
 selected=avg[['gain_MG','gain_augmented']].max(axis=1).nlargest(3)
 (H/'frozen_selection.json').write_text(json.dumps(dict(selected=[list(i) for i in selected.index],model_sha256=hashlib.sha256((H/'frozen_models.pkl').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((H/'CAMPAIGN.md').read_bytes()).hexdigest()),indent=2))
def confirm():
 with (H/'frozen_models.pkl').open('rb') as f:frozen=pickle.load(f)
 rows=[]
 for ch in COLS:
  for h in [6,24]:
   d=collect(ch,h,'test');model,fixed=frozen[ch+'_'+str(h)];rows.append(decisions(d,model,fixed))
 d=pd.concat(rows);d.to_csv(H/'confirmation_decisions.csv',index=False)
 avg=d.groupby(['channel','horizon','method']).loss.mean().unstack();baseline=[x for x in avg.columns if x not in ['MG','cheap_MG','oracle']]
 avg['gain_MG']=1-avg.MG/avg[baseline].min(1);avg['gain_augmented']=1-avg.cheap_MG/avg[baseline].min(1);avg.to_csv(H/'confirmation_summary.csv')
 print(avg[['MG','cheap_MG','validation','gain_MG','gain_augmented']].round(3))
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if '--confirm' in sys.argv:confirm()
  else:develop()
