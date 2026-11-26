"""Frozen selectors and paired evaluation, with all failures retained."""
import argparse
from pathlib import Path
import itertools
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

def summarize(rows, output):
    df=pd.DataFrame(rows);df.to_csv(output/'checkpoint_selections.csv',index=False)
    test=df[df.seed > 231]
    summary=test.groupby('method').agg(reward=('reward','mean'),survival=('complete','mean'),regret=('regret','mean')).reset_index()
    base=test[test.method=='MG'].set_index('seed')
    comparisons=[]
    for metric in test.method.unique():
        if metric=='MG':continue
        other=test[test.method==metric].set_index('seed');diff=(base.reward-other.reward).to_numpy()
        signs=np.array(list(itertools.product([-1,1],repeat=len(diff))))
        p=float(np.mean(np.abs((signs*diff).mean(1))>=abs(diff.mean())-1e-12))
        comparisons.append(dict(method=metric,MG_minus_baseline=float(diff.mean()),wins=int((diff>1e-9).sum()),ties=int((abs(diff)<=1e-9).sum()),n=len(diff),two_sided_signflip_p=p))
    summary.to_csv(output/'checkpoint_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(output/'checkpoint_comparisons.csv',index=False)
    print(summary.round(4).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=H / "logs")
    parser.add_argument("--output", type=Path, default=H.parents[2] / "runs" / H.name / "scored")
    args = parser.parse_args()
    episodes = {
        seed: pd.read_csv(args.input / f"checkpoint_selection_s{seed}.csv", float_precision="round_trip")
        for seed in range(231, 236)
    }
    keys = ["coef", "step"]
    pilot = episodes[231]
    _, eligible = candidates(pilot, keys)
    pilot_reward = pilot[pilot.split == "target"].groupby(keys).reward.mean()
    fixed = pilot_reward.loc[eligible.index].idxmax()
    rows = []
    for seed, data in episodes.items():
        scored, _, _ = evaluate(data, seed, keys, fixed)
        rows.extend(scored)
    args.output.mkdir(parents=True, exist_ok=True)
    summarize(rows, args.output)

if __name__=='__main__':main()
