from pathlib import Path
import argparse,copy,hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from run import load_data,TextVAE,reference

def source(seed):return H.parent/('pilot_seed0' if seed==0 else f'confirmation_seed{seed}')

def encoder_only_step(net,opt,x,eps,audit=False):
    names={name for name,_ in net.named_parameters() if name.startswith(('encoder.','mu.','lv.'))}
    before={name:p.detach().clone() for name,p in net.named_parameters()} if audit else None
    states={name:copy.deepcopy(opt.state[p]) for name,p in net.named_parameters() if name not in names} if audit else None
    for name,p in net.named_parameters():p.requires_grad_(name in names)
    try:
        opt.zero_grad(set_to_none=True);ce,kl=net(x,eps);loss=ce.sum(1).mean()+kl.mean()
        loss.backward();nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5);opt.step()
        if audit:
            changed=[]
            for name,p in net.named_parameters():
                if name not in names:
                    assert torch.equal(before[name],p),f'Frozen decoder changed: {name}'
                    for key,val in states[name].items():
                        if torch.is_tensor(val):assert torch.equal(val,opt.state[p][key])
                        else:assert val==opt.state[p][key]
                else:changed.append(not torch.equal(before[name],p))
            assert any(changed),'No encoder update occurred'
    finally:
        for p in net.parameters():p.requires_grad_(True)
    return float(loss.detach())

def freebits_loss(net,x,eps):
    mu,lv=net.encode(x);z=mu+(lv/2).exp()*eps;logits=net.decode(x,z)
    ce=F.cross_entropy(logits.flatten(0,1),x.flatten(),ignore_index=0,reduction='none').reshape_as(x)
    per_dim=.5*(mu.square()+lv.exp()-lv-1)
    penalty=per_dim.mean(0).clamp_min(.5).sum()
    return ce,per_dim.sum(1),ce.sum(1).mean()+penalty

def retention(seed,out):
    src=source(seed);before=pd.read_csv(src/'base/reference.csv').query('512<=step<=1024').median(numeric_only=True)
    regular=pd.read_csv(src/'regularized/reference.csv').query('2048<=step<=3072').median(numeric_only=True)
    protected=pd.read_csv(out/'reference.csv').query('2048<=step<=3072').median(numeric_only=True)
    details={name:dict(before=float(before[name]),regularized=float(regular[name]),protected=float(protected[name]),
        retained_fraction=float(protected[name]/before[name]),versus_regularized=float(protected[name]/regular[name]))
        for name in ['MI','shuffle_symkl']}
    passed=all(v['retained_fraction']>=.5 and v['versus_regularized']>=2 for v in details.values())
    result=dict(seed=seed,protection_valid=bool(passed),reference=details)
    (out/'retention.json').write_text(json.dumps(result,indent=2));return result

def run(seed,mode,out):
    src=source(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'meta.json').exists():raise RuntimeError('Complete branch already exists; do not overwrite it.')
    torch.manual_seed(seed);train,val,vocab=load_data();net=TextVAE(vocab);opt=torch.optim.Adam(net.parameters(),lr=.001)
    saved=torch.load(src/'branch.pt',weights_only=True);net.load_state_dict(saved['net']);opt.load_state_dict(saved['opt'])
    rng=torch.Generator();rng.set_state(saved['rng']);inner_rng=torch.Generator().manual_seed(seed+90191)
    probe_eps=torch.randn(16,16,generator=torch.Generator().manual_seed(829))
    ref_eps=torch.randn(4,len(val),16,generator=torch.Generator().manual_seed(830))
    logs=pd.read_csv(src/'base/logs.csv').query('step<1024').to_dict('records')
    refs=pd.read_csv(src/'base/reference.csv').query('step<1024').to_dict('records')
    outer_hash=hashlib.sha256();inner_count=0;start=time.perf_counter()
    for t in range(1024,3072):
        if t%128==0:
            stat=reference(net,val,ref_eps);refs.append(dict(step=t,**stat));pd.DataFrame(refs).to_csv(out/'reference.csv',index=False)
            print(seed,mode,t,'MI',round(stat['MI'],3),'shuffle',round(stat['shuffle_symkl'],4),flush=True)
        tic=time.perf_counter()
        with torch.no_grad():
            ce,_=net(val[:16],probe_eps);probe=float(ce.sum()/(val[:16]!=0).sum())
        probe_seconds=time.perf_counter()-tic
        if t==1024:
            expected=pd.read_csv(src/'regularized/logs.csv').query('step==1024').probe_nll.iloc[0]
            assert np.isclose(probe,expected,rtol=0,atol=1e-6),'Branch observation mismatch'
        # This draw sequence is exactly the unchanged original outer loop.
        ix=torch.randint(len(train),(32,),generator=rng);eps=torch.randn(32,16,generator=rng)
        outer_hash.update(ix.numpy().tobytes());outer_hash.update(eps.numpy().tobytes())
        x=train[ix];tic=time.perf_counter();extra_loss=float('nan')
        if mode=='encoder5':
            for j in range(5):
                ii=torch.randint(len(train),(32,),generator=inner_rng);ee=torch.randn(32,16,generator=inner_rng)
                extra_loss=encoder_only_step(net,opt,train[ii],ee,audit=t==1024 and j==0);inner_count+=1
        extra_seconds=time.perf_counter()-tic;tic=time.perf_counter();opt.zero_grad(set_to_none=True)
        if mode=='freebits':ce,kl,loss=freebits_loss(net,x,eps)
        else:ce,kl=net(x,eps);loss=ce.sum(1).mean()+kl.mean()
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        logs.append(dict(step=t,probe_nll=probe,train_nll=float(ce.detach().sum()/(x!=0).sum()),
            train_KL=float(kl.detach().mean()),beta=1.,train_seconds=time.perf_counter()-tic,probe_seconds=probe_seconds,
            extra_encoder_seconds=extra_seconds,extra_encoder_loss=extra_loss,extra_encoder_updates=5 if mode=='encoder5' else 0))
        if (t+1)%128==0:pd.DataFrame(logs).to_csv(out/'logs.csv',index=False)
    refs.append(dict(step=3072,**reference(net,val,ref_eps)))
    pd.DataFrame(logs).to_csv(out/'logs.csv',index=False);pd.DataFrame(refs).to_csv(out/'reference.csv',index=False)
    torch.save(net.state_dict(),out/'final.pt')
    # Independent replay audits the whole outer RNG stream.
    replay=torch.Generator();replay.set_state(saved['rng']);check=hashlib.sha256()
    for _ in range(2048):
        ii=torch.randint(len(train),(32,),generator=replay);ee=torch.randn(32,16,generator=replay)
        check.update(ii.numpy().tobytes());check.update(ee.numpy().tobytes())
    assert check.hexdigest()==outer_hash.hexdigest()
    meta=dict(seed=seed,mode=mode,steps=3072,switch=1024,seconds=time.perf_counter()-start,
        extra_encoder_updates=inner_count,outer_stream_sha256=check.hexdigest(),
        outer_stream_matches_original=True,frozen_decoder_update_audited=mode=='encoder5',
        source_checkpoint_sha256=hashlib.sha256((src/'branch.pt').read_bytes()).hexdigest())
    (out/'meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(retention(seed,out),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',choices=['encoder5','freebits'],default='encoder5')
    p.add_argument('--out',type=Path);a=p.parse_args();out=a.out or H/a.mode/f'seed{a.seed}'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):run(a.seed,a.mode,out)
