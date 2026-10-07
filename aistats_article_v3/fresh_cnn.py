"""Fresh CNN confirmation runs with event times fixed before training."""
from pathlib import Path
import json, sys, time
import numpy as np
import torch
import torch.nn as nn

H=Path(__file__).resolve().parent
ROOT=H.parent
sys.path.insert(0,str(ROOT/'research_trajectory_reference'))
from cifar_events import load, model

SEEDS=list(range(300,310))
EVENTS=json.loads((H/'new_results/event_assignment.json').read_text())
ARMS=('base','lr10','freeze_head','prune80')
OUT=H/'new_results/fresh_cnn';OUT.mkdir(parents=True,exist_ok=True)
TOTAL_BASE=14000
POST=5000

def train(arm,seed,event,data):
    X,y,_,_,Xt,yt=data
    net=model(seed); params=list(net.parameters()); names=[n for n,_ in net.named_parameters()]
    opt=torch.optim.SGD(params,lr=.02,momentum=.9,weight_decay=5e-4)
    lossf=nn.CrossEntropyLoss(); rng=np.random.default_rng(1000+seed)
    steps=TOTAL_BASE if arm=='base' else event+POST
    logs={k:np.empty(steps,dtype=np.float64) for k in ['param_norm','batch_loss','update_norm','moving_frac']}
    previous=None; masks=None; t0=time.perf_counter()
    for t in range(steps):
        if arm!='base' and t==event:
            if arm=='lr10':
                for g in opt.param_groups:g['lr']/=10
            elif arm=='freeze_head':
                keep=lambda n:n.startswith('10.')
                for n,p in zip(names,params):p.requires_grad_(keep(n))
                opt=torch.optim.SGD([p for n,p in zip(names,params) if keep(n)],lr=.02,momentum=.9,weight_decay=5e-4)
            elif arm=='prune80':
                weights=[p for p in params if p.dim()>1]
                allw=torch.cat([p.detach().abs().reshape(-1) for p in weights])
                threshold=torch.quantile(allw,.8); masks=[(p.detach().abs()>threshold).float() for p in weights]
                with torch.no_grad():
                    for p,m in zip(weights,masks):p.mul_(m)
        idx=torch.as_tensor(rng.integers(0,len(X),64))
        opt.zero_grad(); loss=lossf(net(X[idx]),y[idx]); loss.backward(); opt.step()
        if masks is not None:
            with torch.no_grad():
                for p,m in zip([p for p in params if p.dim()>1],masks):p.mul_(m)
        with torch.no_grad():
            flat=torch.cat([p.detach().reshape(-1) for p in params])
        if previous is None: delta=torch.zeros_like(flat)
        else: delta=flat-previous
        previous=flat.clone()
        logs['param_norm'][t]=flat.norm().item(); logs['batch_loss'][t]=loss.item()
        logs['update_norm'][t]=delta.norm().item(); logs['moving_frac'][t]=float((delta.abs()>1e-7).float().mean())
    with torch.no_grad():acc=(net(Xt).argmax(1)==yt).float().mean().item()
    meta=dict(arm=arm,seed=seed,event=event,steps=steps,test_acc=float(acc),wall_s=time.perf_counter()-t0)
    return logs,meta

def main():
    torch.set_num_threads(6)
    data=load(); allmeta=[]
    for seed in SEEDS:
        event=int(EVENTS[str(seed)])
        for arm in ARMS:
            path=OUT/f'logs_{arm}_s{seed}.npz';mp=OUT/f'meta_{arm}_s{seed}.json'
            if path.exists() and mp.exists():
                allmeta.append(json.loads(mp.read_text()));continue
            logs,meta=train(arm,seed,event,data)
            np.savez_compressed(path,**logs);mp.write_text(json.dumps(meta,indent=2));allmeta.append(meta)
            print(f'{arm:12s} seed {seed} event {event} test {meta["test_acc"]:.3f} time {meta["wall_s"]:.0f}s',flush=True)
    (OUT/'meta.json').write_text(json.dumps(allmeta,indent=2))

if __name__=='__main__':main()
