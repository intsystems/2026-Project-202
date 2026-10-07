"""Frozen selectors and paired evaluation, with all failures retained."""
from pathlib import Path
import json,itertools,hashlib
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent
METRICS=['MG','entropy','increments','recurrence','J1']

def candidates(d,keys):
    n=d[d.split=='nominal'].copy()
    # Old exploratory runs calculated features on incomplete records: ignore them.
    q=n.groupby(keys).agg(reward=('reward','mean'),complete=('complete','sum'))
    q=q.join(n[n.complete].groupby(keys)[METRICS].median())
    q=q.sort_index()
    e=q[(q.complete>=2)&(q.reward>=.9*q.reward.max())]
    if len(e)==0:e=q.loc[[q.reward.idxmax()]]
    return q,e

def evaluate(d,seed,keys,fixed):
    q,e=candidates(d,keys)
    target=d[d.split=='target']
    if 'delay' in d.columns:target=target[target.delay>0]
    target=target.groupby(keys).agg(reward=('reward','mean'),complete=('complete','mean'))
    selections={'reward':e.reward.idxmax()}
    for metric in METRICS:
        valid=e[metric].dropna()
        selections[metric]=valid.idxmin() if len(valid) else selections['reward']
    selections['fixed_pilot']=fixed if fixed in e.index else selections['reward']
    selections['oracle_eligible']=target.loc[e.index].reward.idxmax()
    selections['oracle_all']=target.reward.idxmax()
    rows=[]
    for method,key in selections.items():
        v=target.loc[key]
        kd=dict(zip(keys,key if isinstance(key,tuple) else (key,)))
        rows.append(dict(seed=seed,method=method,**kd,reward=float(v.reward),complete=float(v.complete),
                         regret=float(target.loc[e.index].reward.max()-v.reward)))
    rows.append(dict(seed=seed,method='random_eligible',reward=float(target.loc[e.index].reward.mean()),
                     complete=float(target.loc[e.index].complete.mean()),regret=float(target.loc[e.index].reward.max()-target.loc[e.index].reward.mean())))
    return rows,q,e

def summarize(rows,name):
    df=pd.DataFrame(rows);df.to_csv(H/f'{name}_selections.csv',index=False)
    test=df[df.seed> (230 if name=='delay' else 231)]
    summary=test.groupby('method').agg(reward=('reward','mean'),survival=('complete','mean'),regret=('regret','mean')).reset_index()
    base=test[test.method=='MG'].set_index('seed')
    comparisons=[]
    for metric in test.method.unique():
        if metric=='MG':continue
        other=test[test.method==metric].set_index('seed');diff=(base.reward-other.reward).to_numpy()
        signs=np.array(list(itertools.product([-1,1],repeat=len(diff))))
        p=float(np.mean(np.abs((signs*diff).mean(1))>=abs(diff.mean())-1e-12))
        comparisons.append(dict(method=metric,MG_minus_baseline=float(diff.mean()),wins=int((diff>1e-9).sum()),ties=int((abs(diff)<=1e-9).sum()),n=len(diff),two_sided_signflip_p=p))
    summary.to_csv(H/f'{name}_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(H/f'{name}_comparisons.csv',index=False)
    print(name,summary.round(4).to_string(index=False))

def main():
    pilot=pd.DataFrame(json.loads((H/'seed230.json').read_text())['rows'])
    _,eligible=candidates(pilot,['coef'])
    fixed=pilot[(pilot.split=='target')&(pilot.delay>0)].groupby('coef').reward.mean().loc[eligible.index].idxmax()
    delay_fixed=float(fixed)
    rows=[];raw=[]
    for seed in range(230,236):
        d=pd.DataFrame(json.loads((H/f'seed{seed}.json').read_text())['rows']);raw.append(d)
        r,_,_=evaluate(d,seed,['coef'],fixed);rows+=r
    pd.concat(raw).to_csv(H/'delay_records.csv',index=False);summarize(rows,'delay')
    file=H/'checkpoint_selection_s231.csv'
    if not file.exists():return
    d=pd.read_csv(file);_,e=candidates(d,['coef','step'])
    fixed=d[d.split=='target'].groupby(['coef','step']).reward.mean().loc[e.index].idxmax()
    rows=[];raw=[]
    for seed in range(231,236):
        f=H/f'checkpoint_selection_s{seed}.csv'
        if not f.exists():return
        d=pd.read_csv(f);raw.append(d)
        r,q,e=evaluate(d,seed,['coef','step'],fixed);rows+=r
        q.to_csv(H/f'candidates_s{seed}.csv')
    pd.concat(raw).to_csv(H/'checkpoint_records.csv',index=False)
    summarize(rows,'checkpoint')
    (H/'selection_manifest.json').write_text(json.dumps(dict(fixed_delay=delay_fixed,fixed_checkpoint=[float(fixed[0]),int(fixed[1])],
        protocol_sha256=hashlib.sha256((H/'CHECKPOINT_PROTOCOL.md').read_bytes()).hexdigest(),
        interpretation='New deployment evaluation of historical checkpoints, not causal curriculum or new training.'),indent=2))

if __name__=='__main__':main()
