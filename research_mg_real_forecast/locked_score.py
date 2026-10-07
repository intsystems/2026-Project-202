from routing_v2 import H,load,SCENARIOS,MODELS,VAL,SETS
from generator_screen import evaluate
import pickle,json,sys
import pandas as pd,numpy as np
def main():
 with (H/'routing_v2_frozen.pkl').open('rb') as f:saved=pickle.load(f)
 with (H/'g_frozen.pkl').open('rb') as f:old=pickle.load(f)
 final='--final' in sys.argv
 raw=load(range(651,681) if final else range(631,651));rows=[]
 for n,h in SCENARIOS:
  d=raw[(raw.neuron==n)&(raw.horizon==h)].reset_index(drop=True);Y=d[['loss_'+m for m in MODELS]].to_numpy();choices={name:m.predict(d[SETS[name]].to_numpy()).argmin(1) for name,m in saved[(n,h)]['models'].items()}
  choices.update(fixed=np.full(len(d),saved[(n,h)]['fixed']),validation=d[VAL].to_numpy().argmin(1),oracle=Y.argmin(1))
  choices.update({'fixed_'+m:np.full(len(d),i) for i,m in enumerate(MODELS)})
  oldmodels,fixed=old[f'{n}_{h}'];oldrows=evaluate(d,oldmodels,fixed);oldrows.method='old_'+oldrows.method;oldrows=oldrows.rename(columns={'loss':'error'})
  for name,ids in choices.items():
   ids[d['std'].to_numpy()<1e-10]=0
   for i,k in enumerate(ids):rows.append(dict(neuron=n,horizon=h,seed=int(d.seed.iloc[i]),arm=d.arm.iloc[i],method=name,model=MODELS[k],error=float(Y[i,k])))
  rows.extend(oldrows.to_dict('records'))
 prefix='final' if final else 'locked'
 df=pd.DataFrame(rows);df.to_csv(H/(prefix+'_decisions.csv'),index=False)
 a=df.groupby(['neuron','horizon','method']).error.mean().unstack();a.to_csv(H/(prefix+'_summary.csv'));print(a[['MG','cheap','cheap_MG','cheap_val','validation']].round(5).to_string())
 comparisons=[];rng=np.random.default_rng(314159)
 for n,h in SCENARIOS:
  p=df[(df.neuron==n)&(df.horizon==h)].groupby(['seed','method']).error.mean().unstack()
  for other in p:
   if other=='MG':continue
   diff=(p[other]-p.MG).to_numpy();boot=rng.choice(diff,size=(20000,len(diff))).mean(1);ci=np.quantile(boot,[.025,.975]);signs=rng.choice([-1,1],size=(20000,len(diff)));pv=(1+np.sum(np.abs((signs*diff).mean(1))>=abs(diff.mean())))/20001
   comparisons.append(dict(neuron=n,horizon=h,baseline=other,gain=1-p.MG.mean()/p[other].mean(),mean_improvement=diff.mean(),ci_low=ci[0],ci_high=ci[1],wins=int((diff>1e-12).sum()),ties=int((abs(diff)<=1e-12).sum()),n=len(diff),p_signflip=pv))
 pd.DataFrame(comparisons).to_csv(H/(prefix+'_comparisons.csv'),index=False)
 print(pd.DataFrame(comparisons).query('neuron==2').sort_values('gain').head(8).to_string(index=False))
if __name__=='__main__':main()
