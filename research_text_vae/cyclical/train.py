from pathlib import Path
import argparse,copy,hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from run import TextVAE,load_data,reference
STEPS=7168;START=1024;GRID=64

def beta_at(t):
    return .01 if t<START else min(((t-START)%2048)/1024,1.)

def train(seed):
    out=H/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'meta.json').exists():raise RuntimeError('Complete run exists; do not overwrite.')
    torch.manual_seed(seed);xtrain,xval,vocab=load_data();net=TextVAE(vocab)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    rng=torch.Generator().manual_seed(seed+191)
    pe=torch.randn(16,16,generator=torch.Generator().manual_seed(829))
    re=torch.randn(4,len(xval),16,generator=torch.Generator().manual_seed(830))
    rows=[];refs=[];stream=hashlib.sha256();started=time.perf_counter()
    initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
    for t in range(STEPS):
        if t==START:
            torch.save(dict(net=copy.deepcopy(net.state_dict()),opt=copy.deepcopy(optimizer.state_dict()),rng=rng.get_state()),out/'warmup.pt')
        if t%GRID==0:
            refs.append(dict(step=t,**reference(net,xval,re)))
            pd.DataFrame(refs).to_csv(out/'reference.csv',index=False)
            if t%512==0:print(seed,t,'beta',round(beta_at(t),3),'MI',round(refs[-1]['MI'],3),'shuffle',round(refs[-1]['shuffle_symkl'],4),flush=True)
        tic=time.perf_counter()
        with torch.no_grad():
            ce,_=net(xval[:16],pe);probe=float(ce.sum()/(xval[:16]!=0).sum())
        probe_seconds=time.perf_counter()-tic
        ix=torch.randint(len(xtrain),(32,),generator=rng);eps=torch.randn(32,16,generator=rng)
        stream.update(ix.numpy().tobytes());stream.update(eps.numpy().tobytes())
        x=xtrain[ix];beta=beta_at(t);tic=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        ce,kl=net(x,eps);loss=ce.sum(1).mean()+beta*kl.mean()
        assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
        rows.append(dict(step=t,beta=beta,probe_nll=probe,train_nll=float(ce.detach().sum()/(x!=0).sum()),
            train_KL=float(kl.detach().mean()),train_seconds=time.perf_counter()-tic,probe_seconds=probe_seconds))
        if (t+1)%256==0:pd.DataFrame(rows).to_csv(out/'logs.csv',index=False)
    refs.append(dict(step=STEPS,**reference(net,xval,re)))
    pd.DataFrame(rows).to_csv(out/'logs.csv',index=False);pd.DataFrame(refs).to_csv(out/'reference.csv',index=False)
    torch.save(net.state_dict(),out/'final.pt')
    meta=dict(seed=seed,steps=STEPS,parameters=sum(p.numel() for p in net.parameters()),initial_sha256=initial_hash,
        stream_sha256=stream.hexdigest(),seconds=time.perf_counter()-started,threads=2,device='cpu',
        schedule='1024 beta=.01; 3x2048, first1024 linear 0..1, then1',
        protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest())
    (out/'meta.json').write_text(json.dumps(meta,indent=2));print('COMPLETE',seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):train(a.seed)
