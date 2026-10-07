from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent
FEATURES=['MG','entropy','increments','recurrence','output_error','training_error']
def replay(d,rule):
    cost=0.;calls=0;last=None
    for i,r in enumerate(d.sort_values('step').itertuples()):
        last=r;feature=rule.get('feature')
        if feature:
            if feature!='training_error':cost+=r.rollout_seconds
            cost+=getattr(r,feature+'_seconds');value=getattr(r,feature)
            trigger=np.isfinite(value) and rule['sign']*value<=rule['threshold']
        elif rule['kind']=='periodic':trigger=i%rule['every']==0
        else:trigger=r.step>=rule['first']
        if trigger or r.step==8000:
            # Full assessment uses a rollout even when no feature was observed.
            if not feature or feature=='training_error':cost+=r.rollout_seconds
            cost+=r.reference_seconds;calls+=1
            if r.qualified:break
    qualified=d[d.qualified]
    earliest=int(qualified.step.min()) if len(qualified) else 8000
    return dict(seed=int(last.seed),stop=int(last.step),success=bool(last.qualified),calls=calls,
                seconds=cost+last.training_seconds,extra_steps=max(0,int(last.step)-earliest))
def load(seeds):return pd.concat([pd.read_csv(H/f'seed{s}/records.csv') for s in seeds])
def freeze():
    d=load(range(100,104));policies={}
    for feature in FEATURES:
        opts=[];values=d[feature].dropna()
        for sign in [1,-1]:
            for threshold in np.unique(np.r_[-np.inf,np.quantile(sign*values,np.linspace(0,1,11)),np.inf]):
                rule=dict(kind='threshold',feature=feature,sign=sign,threshold=float(threshold))
                out=pd.DataFrame([replay(g,rule) for _,g in d.groupby('seed')])
                if out.extra_steps.mean()<=1000:opts.append((out.seconds.mean(),out.extra_steps.mean(),rule))
        policies[feature]=min(opts,key=lambda x:x[:2])[2]
    for every in [1,2,3]:policies[f'periodic{every}']=dict(kind='periodic',every=every)
    opts=[]
    for first in [0,200,500,1000,2000,4000,8000]:
        rule=dict(kind='fixed',first=first);out=pd.DataFrame([replay(g,rule) for _,g in d.groupby('seed')])
        if out.extra_steps.mean()<=1000:opts.append((out.seconds.mean(),rule))
    policies['fixed_pilot']=min(opts,key=lambda x:x[0])[1]
    (H/'frozen_rules.json').write_text(json.dumps(policies,indent=2))
    print(policies)
def score():
    rules=json.loads((H/'frozen_rules.json').read_text());rows=[]
    for split,seeds in [('pilot',range(100,104)),('confirmation',range(110,118))]:
        if not all((H/f'seed{s}/records.csv').exists() for s in seeds):continue
        d=load(seeds)
        for method,rule in rules.items():
            for _,g in d.groupby('seed'):rows.append(dict(split=split,method=method,**replay(g,rule)))
    a=pd.DataFrame(rows);a.to_csv(H/'decisions.csv',index=False)
    summary=a.groupby(['split','method']).agg(seconds=('seconds','mean'),calls=('calls','mean'),steps=('stop','mean'),extra_steps=('extra_steps','mean'),success=('success','mean'))
    summary.to_csv(H/'summary.csv');print(summary.round(3).to_string())
if __name__=='__main__':
    if not (H/'frozen_rules.json').exists():freeze()
    score()
