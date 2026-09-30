"""GAN training + independent mode coverage; does not import or calculate MG."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from threadpoolctl import threadpool_limits
from model import Generator,Discriminator,Classifier,initialize,configure
from prepare import H,data

@torch.no_grad()
def metrics(g,c,z,channels=3):
    was=g.training;g.eval();ts=time.perf_counter();labels=[];confs=[];images=[]
    generation=classification=0.
    for batch in z.split(256):
        t=time.perf_counter();f=g(batch);generation+=time.perf_counter()-t
        t=time.perf_counter();p=c(f.reshape(-1,1,28,28)).softmax(1)
        confidence,label=p.max(1);classification+=time.perf_counter()-t
        labels.append(label.reshape(-1,channels).numpy());confs.append(confidence.reshape(-1,channels).numpy())
        if not images:images.append(f[:64])
    lab=np.concatenate(labels);conf=np.concatenate(confs)
    code=np.sum(lab*10**np.arange(channels-1,-1,-1)[None,:],axis=1);modes=10**channels
    def summary(n):
        ix=np.all(conf[:n]>=.9,axis=1);counts=np.bincount(code[:n][ix],minlength=modes)
        prob=counts/max(counts.sum(),1);nz=prob[prob>0]
        entropy=float(-np.sum(nz*np.log(nz))) if len(nz) else float('nan')
        return dict(coverage=int(np.sum(counts>0)),coverage5=int(np.sum(counts>=5)),
                    valid_fraction=float(ix.mean()),effective_modes=float(np.exp(entropy)) if len(nz) else 0.,
                    entropy=entropy,kl_uniform=float(np.log(modes)-entropy))
    out=summary(len(z));small=summary(min(512,len(z)))
    out.update({f'small_{k}':v for k,v in small.items()})
    out.update(generation_seconds=generation,classifier_seconds=classification,evaluation_seconds=time.perf_counter()-ts)
    g.train(was)
    return out,dict(labels=lab,confidence=conf),images[0]

def train(seed,arm,out,steps=4096,switch=2048,every=256,n_eval=10000,resume=None,channels=3):
    out.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
    a,_=data();real=a.data.float()/127.5-1
    g=Generator(channels);d=Discriminator(channels);g.apply(initialize);d.apply(initialize)
    c=Classifier();c.load_state_dict(torch.load(H/'classifier.pt',weights_only=True));c.eval()
    og=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999))
    od=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999))
    rng=torch.Generator().manual_seed(seed+10000)
    zrng=torch.Generator().manual_seed(7711);z_eval=torch.randn(n_eval,64,generator=zrng)
    rows=[];logs=[];first=0
    if resume:
        ck=torch.load(resume,weights_only=False);g.load_state_dict(ck['g']);d.load_state_dict(ck['d'])
        og.load_state_dict(ck['og']);od.load_state_dict(ck['od']);rng.set_state(ck['rng']);first=int(ck['step'])
        base=Path(resume).parent
        logs=pd.read_csv(base/'logs.csv').query('step<=@first').to_dict('records')
        rows=pd.read_csv(base/'reference.csv').query('step<=@first').to_dict('records')
    start=time.perf_counter();training_seconds=0.;evaluation_seconds=0.
    for it in range(first,steps+1):
        if (it==0 or it%every==0) and not (resume and it==first):
            m,raw,grid=metrics(g,c,z_eval,channels);evaluation_seconds+=m['evaluation_seconds']
            rows.append(dict(seed=seed,arm=arm,step=it,**m));pd.DataFrame(rows).to_csv(out/'reference.csv',index=False)
            np.savez_compressed(out/f'evaluation_{it:05d}.npz',**raw)
            save_image((grid+1)/2,out/f'images_{it:05d}.png',nrow=8)
            print(seed,arm,it,'coverage',m['coverage'],'effective',round(m['effective_modes'],1),'valid',round(m['valid_fraction'],3),flush=True)
            torch.save(dict(g=g.state_dict(),d=d.state_dict(),og=og.state_dict(),od=od.state_dict(),rng=rng.get_state(),step=it),out/f'checkpoint_{it:05d}.pt')
        if it==steps:break
        if it==switch:
            if arm=='g_fast':
                for group in og.param_groups:group['lr']=.002
            elif arm=='d_slow':
                for group in od.param_groups:group['lr']=.00002
            elif arm=='frozen':g.eval()
        tic=time.perf_counter();ix=torch.randint(len(real),(64,channels),generator=rng)
        x=real[ix];z=torch.randn(64,64,generator=rng)
        with torch.no_grad():fake=g(z)
        od.zero_grad(set_to_none=True)
        d_real=d(x);d_fake=d(fake)
        loss_d=F.softplus(-d_real).mean()+F.softplus(d_fake).mean()
        loss_d.backward();od.step()
        for v in d.parameters():v.requires_grad_(False)
        z=torch.randn(64,64,generator=rng)
        if arm=='frozen' and it>=switch:
            with torch.no_grad():loss_g=F.softplus(-d(g(z))).mean()
        else:
            og.zero_grad(set_to_none=True);loss_g=F.softplus(-d(g(z))).mean();loss_g.backward();og.step()
        for v in d.parameters():v.requires_grad_(True)
        sec=time.perf_counter()-tic;training_seconds+=sec
        logs.append(dict(step=it+1,g_loss=float(loss_g.detach()),d_loss=float(loss_d.detach()),
                         real_score=float(d_real.detach().mean()),fake_score=float(d_fake.detach().mean()),seconds=sec))
        if (it+1)%every==0:pd.DataFrame(logs).to_csv(out/'logs.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'reference.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(seed=seed,arm=arm,steps=steps,switch=switch,every=every,
        n_eval=n_eval,channels=channels,batch=64,lr=.0002,training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,
        total_seconds=time.perf_counter()-start,first_step=first,
        g_parameters=sum(p.numel() for p in g.parameters()),d_parameters=sum(p.numel() for p in d.parameters()),
        source_checkpoint=str(resume) if resume else None,actual_torch_threads=torch.get_num_threads()),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0)
    p.add_argument('--arm',choices=['base','g_fast','d_slow','frozen'],default='base')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=4096)
    p.add_argument('--switch',type=int,default=2048);p.add_argument('--every',type=int,default=256)
    p.add_argument('--n-eval',type=int,default=10000);p.add_argument('--resume',type=Path)
    p.add_argument('--channels',type=int,choices=[1,3],default=3)
    args=p.parse_args();configure()
    with threadpool_limits(limits=1):train(args.seed,args.arm,args.out,args.steps,args.switch,args.every,args.n_eval,args.resume,args.channels)
