from pathlib import Path
import pickle,json,sys,hashlib
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from campaign import H,MODELS,CHEAP
SCENARIOS=[(0,512),(1,512),(2,32)];VAL=['val_'+m for m in MODELS]
SETS={'MG':['MG'],'cheap':CHEAP,'cheap_MG':CHEAP+['MG'],'val':VAL,'val_MG':VAL+['MG'],'cheap_val':CHEAP+VAL,'cheap_val_MG':CHEAP+VAL+['MG']}
def clean(d):
 d=d.copy();flat=d['std']<1e-10
 for m in MODELS:d.loc[flat,'loss_'+m]=d.loc[flat,'loss_last'].fillna(0)
 return d.replace([np.inf,-np.inf],np.nan)
def load(seeds):return clean(pd.concat([pd.read_csv(H/f'fresh_seed{s}/features.csv') for s in seeds],ignore_index=True))
def model(i):
 m=DecisionTreeRegressor(max_depth=[2,3][i],min_samples_leaf=8,random_state=42) if i<2 else RandomForestRegressor(n_estimators=100,max_depth=[3,6][i-2],min_samples_leaf=5,random_state=42,n_jobs=1)
 return make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),m)
def fit():
 allrows=load(range(601,619));saved={};cvrows=[]
 for n,h in SCENARIOS:
  d=allrows[(allrows.neuron==n)&(allrows.horizon==h)].reset_index(drop=True);Y=d[['loss_'+m for m in MODELS]].to_numpy();group=d.seed.to_numpy()
  saved[(n,h)]={'fixed':int(np.mean(Y,axis=0).argmin()),'models':{}}
  for name,cols in SETS.items():
   scores=[]
   for i in range(4):
    errors=[]
    for tr,te in GroupKFold(3).split(d,Y,group):
     m=model(i).fit(d.iloc[tr][cols].to_numpy(),Y[tr]);pred=m.predict(d.iloc[te][cols].to_numpy()).argmin(1);errors.extend(Y[te,pred])
    scores.append(np.mean(errors));cvrows.append(dict(neuron=n,horizon=h,method=name,classifier=i,error=scores[-1]))
   selected=int(np.argmin(scores));saved[(n,h)]['models'][name]=model(selected).fit(d[cols].to_numpy(),Y)
  print('fit',n,h,flush=True)
 pd.DataFrame(cvrows).to_csv(H/'routing_v2_cv.csv',index=False)
 with (H/'routing_v2_frozen.pkl').open('wb') as f:pickle.dump(saved,f)
 (H/'routing_v2_manifest.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256((H/'ROUTING_V2.md').read_bytes()).hexdigest(),model_sha256=hashlib.sha256((H/'routing_v2_frozen.pkl').read_bytes()).hexdigest(),train_seeds=list(range(601,619)),test_seeds=list(range(619,631))),indent=2))
def test():
 with (H/'routing_v2_frozen.pkl').open('rb') as f:saved=pickle.load(f)
 d0=load(range(619,631));rows=[]
 for n,h in SCENARIOS:
  d=d0[(d0.neuron==n)&(d0.horizon==h)].reset_index(drop=True);Y=d[['loss_'+m for m in MODELS]].to_numpy();choices={name:m.predict(d[SETS[name]].to_numpy()).argmin(1) for name,m in saved[(n,h)]['models'].items()}
  choices.update(fixed=np.full(len(d),saved[(n,h)]['fixed']),validation=d[VAL].to_numpy().argmin(1),oracle=Y.argmin(1))
  for j,name in enumerate(MODELS):choices['fixed_'+name]=np.full(len(d),j)
  for name,ids in choices.items():
   ids[d['std'].to_numpy()<1e-10]=0
   for i,k in enumerate(ids):rows.append(dict(neuron=n,horizon=h,seed=int(d.seed.iloc[i]),arm=d.arm.iloc[i],method=name,model=MODELS[k],error=float(Y[i,k])))
 df=pd.DataFrame(rows);df.to_csv(H/'routing_v2_test.csv',index=False);s=df.groupby(['neuron','horizon','method']).error.mean().unstack();s.to_csv(H/'routing_v2_summary.csv');print(s.round(4).to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):test() if '--test' in sys.argv else fit()
