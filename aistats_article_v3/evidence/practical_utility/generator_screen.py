from campaign import H,feat,predict,MODELS,CHEAP,SETS
R=H.parent
from sklearn.tree import DecisionTreeRegressor
import pandas as pd,numpy as np,json,pickle,time,sys
from threadpoolctl import threadpool_limits
ARMS=['T1','T2','T3','T4','H2','H4','M4','chaos']
def forecast(x,h,m):
 if m=='daily':return np.resize(x[-314:],h)
 return predict(x,h,m)
def bank(neuron,h,seeds):
 file=H/f'g_bank_n{neuron}_h{h}_s{seeds[0]}.csv'
 if file.exists():return pd.read_csv(file)
 rows=[]
 for seed in seeds:
  for arm in ARMS:
   raw=np.load(R/f'research_generator/results/obs_{arm}_s{seed}.npz')['obs'][:,neuron];origin=6144;x=raw[4096:origin];y=raw[origin:origin+h];v=x.var()
   row=dict(seed=seed,arm=arm,neuron=neuron,horizon=h,**feat(x));span=max(128,h)
   for m in MODELS:
    tic=time.perf_counter();p=forecast(x,h,m);row['time_'+m]=time.perf_counter()-tic;row['loss_'+m]=float(np.mean((p-y)**2)/v)
    tic=time.perf_counter();values=[np.mean((forecast(x[:o],h,m)-x[o:o+h])**2)/v for o in range(len(x)-span,len(x)-h+1,h)]
    row['val_'+m]=float(np.mean(values));row['vtime_'+m]=time.perf_counter()-tic
   rows.append(row)
 d=pd.DataFrame(rows);d.to_csv(file,index=False);print('bank',neuron,h,seeds,flush=True);return d
def fit(d,cols):
 X=d[cols];median=X.median().fillna(0);x=np.column_stack([X.fillna(median),X.isna().any(axis=1)])
 m=DecisionTreeRegressor(max_depth=2,min_samples_leaf=3,random_state=42).fit(x,d[['loss_'+k for k in MODELS]])
 return m,median
def evaluate(d,models,fixed):
 losses=d[['loss_'+k for k in MODELS]].to_numpy();choices={k:np.full(len(d),j) for j,k in enumerate(MODELS)}
 choices.update(fixed=np.full(len(d),fixed),validation=d[['val_'+k for k in MODELS]].to_numpy().argmin(1),oracle=losses.argmin(1))
 for name,(m,median) in models.items():
  X=d[SETS[name]];xx=np.column_stack([X.fillna(median),X.isna().any(axis=1)]);choices[name]=m.predict(xx).argmin(1)
 rows=[]
 for name,choice in choices.items():
  for i,k in enumerate(choice):rows.append(dict(seed=int(d.seed.iloc[i]),arm=d.arm.iloc[i],neuron=int(d.neuron.iloc[i]),horizon=int(d.horizon.iloc[i]),method=name,model=MODELS[k],loss=losses[i,k]))
 return pd.DataFrame(rows)
def main():
 if '--test' not in sys.argv:
  rows=[];frozen={}
  for n in range(3):
   for h in [8,32,128,512]:
    d=bank(n,h,[1,2]);frozen[f'{n}_{h}']=({name:fit(d,cols) for name,cols in SETS.items()},int(d[['loss_'+m for m in MODELS]].mean().argmin()))
    for seed in [1,2]:
     train=d[d.seed!=seed];test=d[d.seed==seed];models={name:fit(train,cols) for name,cols in SETS.items()};fixed=int(train[['loss_'+m for m in MODELS]].mean().argmin());rows.append(evaluate(test,models,fixed))
  with (H/'g_frozen.pkl').open('wb') as f:pickle.dump(frozen,f)
  df=pd.concat(rows);df.to_csv(H/'g_development_decisions.csv',index=False)
  avg=df.groupby(['neuron','horizon','method']).loss.mean().unstack();b=[c for c in avg.columns if c not in ['MG','cheap_MG','oracle']];avg['gain']=1-avg[['MG','cheap_MG']].min(axis=1)/avg[b].min(axis=1)
  avg.to_csv(H/'g_development_summary.csv');(H/'g_selected.json').write_text(json.dumps([list(x) for x in avg.gain.nlargest(3).index]));print(avg[['MG','cheap_MG','validation','gain']])
 else:
  with (H/'g_frozen.pkl').open('rb') as f:frozen=pickle.load(f)
  rows=[]
  for n in range(3):
   for h in [8,32,128,512]:
    d=bank(n,h,[3,4,5]);models,fixed=frozen[f'{n}_{h}'];rows.append(evaluate(d,models,fixed))
  df=pd.concat(rows);df.to_csv(H/'g_test_decisions.csv',index=False)
  avg=df.groupby(['neuron','horizon','method']).loss.mean().unstack();b=[c for c in avg.columns if c not in ['MG','cheap_MG','oracle']];avg['gain_MG']=1-avg.MG/avg[b].min(axis=1);avg['gain_aug']=1-avg.cheap_MG/avg[b].min(axis=1)
  avg.to_csv(H/'g_test_summary.csv');print(avg[['MG','cheap_MG','validation','gain_MG','gain_aug']])
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
