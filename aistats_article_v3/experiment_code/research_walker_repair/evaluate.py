import argparse,json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,rollout
VALIDATION=list(range(41001,41006));TEST=list(range(51001,51011));FINAL=1048576

def evaluate(label,split,steps=None):
    out=H/label;resets=VALIDATION if split=='validation' else TEST
    for p in sorted(out.glob('step*/policy.zip')):
        step=int(p.parent.name[4:])
        if steps is not None and step not in steps:continue
        actor=None
        for reset in resets:
            if not (p.parent/f'reset{reset}/metrics.json').exists() and actor is None:actor=Actor(p.parent)
            rollout(actor,p.parent,reset)
        print('EVAL',label,split,step,flush=True)
    rows=[]
    for p in sorted(out.glob('step*/reset*/metrics.json')):
        m=json.loads(p.read_text())
        if m['reset'] not in resets:continue
        rows.append(dict(step=int(p.parent.parent.name[4:]),walking=bool(m['complete'] and m['mean_speed']>=.5),**{k:v for k,v in m.items() if k!='cheap'}))
    frame=pd.DataFrame(rows);frame.to_csv(out/f'{split}.csv',index=False)
    return frame

def gate(label):
    out=H/label;frame=pd.read_csv(out/'validation.csv');base=frame[frame.step==0]
    baseline=float(base.loc[base.walking,'mean_reward'].median());checks=[]
    for step in [FINAL-262144,FINAL-131072,FINAL]:
        part=frame[frame.step==step];walk=part.walking.sum();reward=float(part.loc[part.walking,'mean_reward'].median())
        checks.append(dict(step=step,walking=int(walk),median_reward=reward,ratio=reward/baseline,pass_check=bool(len(part)==5 and walk>=4 and reward>=.9*baseline)))
    result=dict(label=label,baseline_reward=baseline,checks=checks,passed=all(r['pass_check'] for r in checks),
        reward_relative_range=float((max(r['median_reward'] for r in checks)-min(r['median_reward'] for r in checks))/baseline))
    (out/'gate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result

def pair(label):
    out=H/label;frame=pd.read_csv(out/'test.csv');early=frame[frame.step==0];late=frame[frame.step==FINAL]
    common=sorted(set(early[early.eligible].reset)&set(late[late.eligible].reset))
    result=dict(label=label,early=0,late=FINAL,common_resets=[int(x) for x in common],usable=len(common)>=8,
        walking_early=int(early.walking.sum()),walking_late=int(late.walking.sum()),test_resets=TEST)
    (out/'pair.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='seed210');p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--steps',nargs='*',type=int);p.add_argument('--gate',action='store_true');p.add_argument('--pair',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        if a.gate:gate(a.label)
        elif a.pair:pair(a.label)
        else:evaluate(a.label,a.split,a.steps)
