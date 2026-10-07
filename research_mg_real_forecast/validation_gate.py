from generator_screen import H,MODELS
import numpy as np,pandas as pd,json,pickle,sys
from sklearn.tree import DecisionTreeRegressor
FEATURES=['MG','entropy','increments','recurrence','peak','vmin','vmargin']
SCENARIOS=[(0,512),(1,512),(2,32)]
def prepare(d,fixed):
 d=d.copy();val=d[['val_'+m for m in MODELS]].to_numpy();loss=d[['loss_'+m for m in MODELS]].to_numpy()
 d['vmin']=np.min(val,axis=1);d['vmargin']=np.sort(val,axis=1)[:,1]-np.min(val,axis=1)
 d['lv']=loss[np.arange(len(d)),val.argmin(1)];d['lf']=loss[:,fixed]
 return d
def fit():
 result={}
 for n,h in SCENARIOS:
  d=pd.read_csv(H/f'g_bank_n{n}_h{h}_s1.csv');fixed=int(d[['loss_'+m for m in MODELS]].mean().argmin());d=prepare(d,fixed);rules={}
  for feat in FEATURES:
   x=d[feat].to_numpy();vals=np.unique(x[np.isfinite(x)]);thresholds=np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf];opts=[]
   for sign in [1,-1]:
    for t in thresholds:
     use=(x<=t) if sign==1 else(x>t);err=np.where(use,d.lv,d.lf).mean();opts.append((err,dict(feature=feat,sign=sign,t=float(t))))
   rules[feat]=min(opts,key=lambda x:x[0])[1]
  models={}
  for name,cols in [('cheap_tree',FEATURES[1:]),('cheap_MG_tree',FEATURES)]:
   X=d[cols];med=X.median().fillna(0);model=DecisionTreeRegressor(max_depth=1,min_samples_leaf=3,random_state=42).fit(X.fillna(med),d.lv-d.lf);models[name]=(cols,med,model)
  result[(n,h)]=(fixed,rules,models)
 with (H/'v_frozen.pkl').open('wb') as f:pickle.dump(result,f)
 (H/'v_rules.json').write_text(json.dumps({str(k):dict(fixed=v[0],rules=v[1]) for k,v in result.items()},indent=2))
def test(seeds):
 with (H/'v_frozen.pkl').open('rb') as f:rules=pickle.load(f)
 rows=[]
 for seed in seeds:
  raw=pd.read_csv(H/f'fresh_seed{seed}/features.csv')
  for (n,h),(fixed,thresholds,models) in rules.items():
   d=prepare(raw[(raw.neuron==n)&(raw.horizon==h)],fixed).reset_index(drop=True);choices=dict(validation=np.ones(len(d),bool),fixed=np.zeros(len(d),bool))
   for name,r in thresholds.items():choices[name]=(d[name]<=r['t']).to_numpy() if r['sign']==1 else(d[name]>r['t']).to_numpy()
   for name,(cols,med,model) in models.items():choices[name]=model.predict(d[cols].fillna(med))<0
   for name,use in choices.items():
    for i in range(len(d)):rows.append(dict(seed=seed,arm=d.arm.iloc[i],neuron=n,horizon=h,method=name,used_validation=bool(use[i]),loss=float(d.lv.iloc[i] if use[i] else d.lf.iloc[i])))
 df=pd.DataFrame(rows);suffix=f'{min(seeds)}_{max(seeds)}';df.to_csv(H/f'v_decisions_{suffix}.csv',index=False);a=df.groupby(['neuron','horizon','method']).loss.mean().unstack();a.to_csv(H/f'v_summary_{suffix}.csv');print(a.round(4).to_string())
if __name__=='__main__':
 if not (H/'v_frozen.pkl').exists():fit()
 test(list(range(601,619)))
