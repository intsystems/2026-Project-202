from pathlib import Path
import json,numpy as np,pandas as pd
H=Path(__file__).resolve().parent;FEATURES=['MG','entropy','increments','slope','level','std','lag1']
def load(seed,noise):
 d=pd.read_csv(H/f'seed{seed}_noise{noise:g}/features.csv');m=pd.read_csv(H/f'seed{seed}_noise{noise:g}/metrics.csv');return d,m
def candidate_rows(seed,noise):
 f,m=load(seed,noise);q=f.groupby('step').first().reset_index().merge(m,on='step',suffixes=('','_metric'));return q
def fit(noise):
 d=pd.concat([candidate_rows(s,noise).assign(seed=s) for s in [0,1,2]],ignore_index=True);rules={}
 for feat in FEATURES:
  v=np.sort(d[feat].dropna().unique());best=None
  for sign in [1,-1]:
   for t in np.r_[-np.inf,(v[:-1]+v[1:])/2,np.inf]:
    stop=[]
    for seed,g in d.groupby('seed'):
     g=g.sort_values('step');use=(g[feat]<=t) if sign==1 else(g[feat]>t);s=g.loc[use,'step'];stop.append(int(s.iloc[0]) if len(s) else 2048)
    score=np.mean([float(d[(d.seed==seed)&(d.step==st)].val_acc.iloc[0])-.02*st/2048 for seed,st in zip([0,1,2],stop)])
    if best is None or score>best[0]:best=(score,dict(feature=feat,sign=sign,threshold=float(t)))
  rules[feat]=best[1]
 return rules
def apply(q,rule):
 q=q.sort_values('step');use=(q[rule['feature']]<=rule['threshold']) if rule['sign']==1 else(q[rule['feature']]>rule['threshold']);s=q.loc[use,'step'];return int(s.iloc[0]) if len(s) else 2048
def main():
 rules={str(noise):fit(noise) for noise in [0,.4,.6]};(H/'stop_rules.json').write_text(json.dumps(rules,indent=2));rows=[]
 for noise in [0,.4,.6]:
  for seed in range(10,18):
   q=candidate_rows(seed,noise)
   for name,rule in rules[str(noise)].items():
    step=apply(q,rule);r=q[q.step==step].iloc[0];rows.append(dict(seed=seed,noise=noise,method=name,step=step,test_acc=r.test_acc,val_acc=r.val_acc,train_noisy_acc=r.train_noisy_acc))
   for name,step in [('fixed512',512),('fixed1024',1024),('fixed1536',1536),('oracle',int(q.loc[q.val_acc.idxmax(),'step']))]:
    r=q[q.step==step].iloc[0];rows.append(dict(seed=seed,noise=noise,method=name,step=step,test_acc=r.test_acc,val_acc=r.val_acc,train_noisy_acc=r.train_noisy_acc))
 d=pd.DataFrame(rows);d.to_csv(H/'stop_decisions.csv',index=False);s=d.groupby(['noise','method']).agg(test_acc=('test_acc','mean'),step=('step','mean'),val_acc=('val_acc','mean')).reset_index();s.to_csv(H/'stop_summary.csv',index=False);print(s.round(4).to_string(index=False))
if __name__=='__main__':main()
