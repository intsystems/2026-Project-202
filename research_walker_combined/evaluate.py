import argparse,json
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,rollout,reduced
from environment import RHO
from whole_cycle import metric as whole_cycle
FINAL=1048576

def strobe(data):
    x=reduced(data['qpos'],data['qvel']);phase=data['phase'];P=152;cross=np.flatnonzero(np.diff(phase)<0);sections=[]
    for i in cross:
        frac=(P-phase[i])/RHO;assert 0<=frac<=1+1e-8
        sections.append(x[i]+frac*(x[i+1]-x[i]))
    sec=np.asarray(sections).reshape(-1,17);var=float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    disp=float(np.mean(np.sum((sec-sec.mean(0))**2,axis=1))/max(var,1e-30)) if len(sec)>=2 else None
    return dict(D_strobe=disp,section_count=len(sec),section_eligible=len(sec)>=8),sec

def evaluate(label,split):
    root=H/label;rows=[];steps=[FINAL-262144,FINAL-131072,FINAL] if split=='validation' else [FINAL]
    resets=range(77001,77006) if split=='validation' else range(78001,78011)
    for step in steps:
        cp=root/f'step{step:07d}';actor=Actor(cp)
        for reset in resets:
            m=rollout(actor,cp,reset);p=cp/f'reset{reset}'
            m.update(D_strobe=None,section_count=0,section_eligible=False,C_cycle=None,mean_curve_drift=None)
            if m['eligible']:
                data=np.load(p/'trajectory.npz');r,sec=strobe(data);m.update(r);m.update(whole_cycle(data));np.savez_compressed(p/'section.npz',points=sec)
            (p/'metrics.json').write_text(json.dumps(m,indent=2));rows.append(dict(label=label,step=step,**{k:v for k,v in m.items() if k!='cheap'}))
        print('EVAL',label,split,step,flush=True)
    pd.DataFrame(rows).to_csv(root/f'{split}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--split',choices=['validation','test'],required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):evaluate(a.label,a.split)
