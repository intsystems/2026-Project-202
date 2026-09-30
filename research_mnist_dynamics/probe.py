"""Exploratory fixed-observer correction; evaluate archived weights without retraining."""
import argparse,time,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run import H,data,model
from analyze import mg,cheap

def detrend(x):return x-np.polyval(np.polyfit(np.arange(len(x)),x,1),np.arange(len(x)))

def evaluate(root):
    x,y,tx,ty,_=data();rng=np.random.default_rng(10299)
    train_ids=np.concatenate([rng.choice(np.flatnonzero(y.numpy()==c),10,False) for c in range(10)])
    test_ids=np.concatenate([rng.choice(np.flatnonzero(ty.numpy()==c),10,False) for c in range(10)])
    px=torch.cat([x[train_ids],tx[test_ids]]);py=torch.cat([y[train_ids],ty[test_ids]])
    net=model();rows=[]
    for path in root.rglob('trajectory.npy'):
        d=path.parent;traj=np.load(path,mmap_mode='r');log=pd.read_csv(d/'logs.csv');tag=str(d.relative_to(root)).replace('\\','/')
        # Recompute after a rerun: a pre-existing CSV could describe older weights.
        out=[]
        with torch.no_grad():
            for i,theta in enumerate(traj):
                v=torch.from_numpy(theta.copy());torch.nn.utils.vector_to_parameters(v,net.parameters())
                t=time.perf_counter();loss=torch.nn.functional.cross_entropy(net(px),py,reduction='none')
                out.append(dict(step=i+1,train_probe=float(loss[:100].mean()),test_probe=float(loss[100:].mean()),
                    seconds=time.perf_counter()-t))
        p=pd.DataFrame(out);p.to_csv(d/'probe.csv',index=False)
        for signal in ['train_probe','test_probe','projection']:
            obs=(log if signal=='projection' else p)[signal].to_numpy()
            for preprocess in ['raw','detrended']:
                for end in range(512,len(obs)+1,256):
                    z=obs[end-512:end];z=detrend(z) if preprocess=='detrended' else z
                    rows.append(dict(run=tag,signal=signal,preprocess=preprocess,end=end,**mg(z),**cheap(z)))
        pd.DataFrame(rows).to_csv(root/'probe_windows.csv',index=False)
        print('Probe',tag,flush=True)
    df=pd.DataFrame(rows)
    a=df[df.end.between(1024,2048)].groupby(['run','signal','preprocess']).MG.median()
    b=df[df.end>=3072].groupby(['run','signal','preprocess']).MG.median()
    ratio=b/a;ratio.to_csv(root/'probe_ratios.csv');print(ratio.to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):evaluate(a.root)
