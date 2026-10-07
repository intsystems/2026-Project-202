import argparse,json
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,rollout,reduced
from reference import Orbit
FINAL=1048576

def evaluate(label,split):
    root=H/label;rows=[];orbit=Orbit()
    steps=[FINAL-262144,FINAL-131072,FINAL] if split=='validation' else [FINAL]
    resets=range(71001,71006) if split=='validation' else range(72001,72011)
    for step in steps:
        cp=root/f'step{step:07d}';actor=Actor(cp)
        for reset in resets:
            m=rollout(actor,cp,reset);p=cp/f'reset{reset}'
            for key,value in dict(D_section=None,section_count=0,section_eligible=False,orbit_distance2=None,orbit_loss=None).items():m.setdefault(key,value)
            if m['eligible']:
                data=np.load(p/'trajectory.npz');r,sec,t=orbit.metrics(reduced(data['qpos'],data['qvel']))
                np.savez_compressed(p/'section.npz',points=sec,times=t);m.update(r)
            (p/'metrics.json').write_text(json.dumps(m,indent=2))
            rows.append(dict(label=label,step=step,**{k:v for k,v in m.items() if k!='cheap'}))
        print('EVAL',label,split,step,flush=True)
    pd.DataFrame(rows).to_csv(root/f'{split}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--split',choices=['validation','test'],required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):evaluate(a.label,a.split)
