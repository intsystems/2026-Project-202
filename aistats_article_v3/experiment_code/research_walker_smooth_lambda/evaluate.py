import argparse,json
import torch,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from motion import H,Actor,rollout
from diagnostic import period_info
FINAL=1048576

def evaluate(label,split):
    root=H/label;rows=[]
    steps=[FINAL-262144,FINAL-131072,FINAL] if split=='validation' else [FINAL]
    resets=range(61001,61006) if split=='validation' else range(62001,62011)
    if label=='anchor':steps=[0]
    for step in steps:
        p=root/f'step{step:07d}';actor=Actor(p)
        for reset in resets:
            m=rollout(actor,p,reset)
            if m['eligible']:
                d=np.load(p/f'reset{reset}'/'trajectory.npz');pi,_=period_info(d['qpos'],d['qvel']);(p/f'reset{reset}'/'period.json').write_text(json.dumps(pi,indent=2))
            rows.append(dict(label=label,step=step,**{k:v for k,v in m.items() if k!='cheap'}))
        print('EVAL',label,split,step,flush=True)
    pd.DataFrame(rows).to_csv(root/f'{split}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--split',choices=['validation','test'],required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):evaluate(a.label,a.split)
