from pathlib import Path
import argparse,collections,copy,hashlib,json,time,urllib.request
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

H=Path(__file__).resolve().parent

def load_data():
    d=H/'data';d.mkdir(parents=True,exist_ok=True);raw={};hashes={}
    for split in ['train','valid']:
        path=d/f'ptb.{split}.txt'
        if not path.exists():
            urllib.request.urlretrieve(f'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.{split}.txt',path)
        content=path.read_bytes();hashes[split]=hashlib.sha256(content).hexdigest()
        raw[split]=[l.split() for l in content.decode().splitlines()]
    counts=collections.Counter(w for line in raw['train'] for w in line)
    vocab=['<pad>','<bos>','<eos>','<unk>']+[w for w,_ in counts.most_common() if w not in ['<pad>','<bos>','<eos>','<unk>']][:1996]
    lookup={w:i for i,w in enumerate(vocab)};rng=np.random.default_rng(20260929);out={};ids={}
    for split,n in [('train',6000),('valid',256)]:
        eligible=[i for i,line in enumerate(raw[split]) if 8<=len(line)<=24]
        selected=rng.choice(eligible,n,replace=False);ids[split]=selected.tolist()
        a=np.zeros((n,25),dtype=np.int64)
        for row,idx in enumerate(selected):
            line=raw[split][idx];tokens=[lookup.get(w,3) for w in line]+[2];a[row,:len(tokens)]=tokens
        out[split]=torch.from_numpy(a)
    selection=json.dumps(dict(hashes=hashes,vocab=vocab,indices=ids),indent=2)
    if not (d/'selection.json').exists() or (d/'selection.json').read_text()!=selection:
        (d/'selection.json').write_text(selection)
    return out['train'],out['valid'],len(vocab)

class TextVAE(nn.Module):
    def __init__(self,vocab):
        super().__init__();self.emb=nn.Embedding(vocab,48,padding_idx=0)
        self.encoder=nn.GRU(48,64,batch_first=True);self.mu=nn.Linear(64,16);self.lv=nn.Linear(64,16)
        self.init=nn.Linear(16,64);self.decoder=nn.GRU(64,64,batch_first=True);self.output=nn.Linear(64,vocab)
    def encode(self,x):
        h,_=self.encoder(self.emb(x));last=h[torch.arange(len(x)),(x!=0).sum(1)-1]
        return self.mu(last),self.lv(last)
    def decode(self,x,z):
        prev=torch.cat([torch.ones((len(x),1),dtype=torch.long),x[:,:-1]],1)
        inp=torch.cat([self.emb(prev),z[:,None,:].expand(-1,x.shape[1],-1)],-1)
        h,_=self.decoder(inp,torch.tanh(self.init(z))[None]);return self.output(h)
    def forward(self,x,eps):
        mu,lv=self.encode(x);z=mu+torch.exp(lv/2)*eps;logits=self.decode(x,z)
        ce=F.cross_entropy(logits.flatten(0,1),x.flatten(),ignore_index=0,reduction='none').reshape_as(x)
        kl=.5*(mu.square()+lv.exp()-lv-1).sum(1)
        return ce,kl

@torch.no_grad()
def reference(net,x,eps):
    start=time.perf_counter();mu,lv=net.encode(x)
    kl=.5*(mu.square()+lv.exp()-lv-1).sum(1).mean();kl_time=time.perf_counter()-start
    t=time.perf_counter();z=mu[None]+(lv/2).exp()[None]*eps
    # Four independent samples per sentence; uniform empirical mixture over256 sentences.
    zz=z.flatten(0,1);logq=-.5*((zz[:,None,:]-mu[None])**2/ lv.exp()[None]+lv[None]+np.log(2*np.pi)).sum(-1)
    own=-.5*(eps.square()+lv[None]+np.log(2*np.pi)).sum(-1).flatten()
    mi=(own-torch.logsumexp(logq,dim=1)+np.log(len(x))).mean();mi_time=time.perf_counter()-t+kl_time
    t=time.perf_counter();fixed_eps=eps[0]
    good=net.decode(x,mu+(lv/2).exp()*fixed_eps)
    other=net.decode(x,mu.roll(1,0)+(lv.roll(1,0)/2).exp()*fixed_eps)
    lp=good.log_softmax(-1);lq=other.log_softmax(-1);mask=x!=0
    sym=.5*((lp.exp()-lq.exp())*(lp-lq)).sum(-1)
    goodce=F.cross_entropy(good.flatten(0,1),x.flatten(),ignore_index=0,reduction='sum')/mask.sum()
    badce=F.cross_entropy(other.flatten(0,1),x.flatten(),ignore_index=0,reduction='sum')/mask.sum()
    return dict(KL=float(kl),MI=float(mi),shuffle_symkl=float(sym[mask].mean()),shuffle_nll_gap=float(badce-goodce),
        nll=float(goodce),active_units=int((mu.var(0,unbiased=False)>.01).sum()),
        kl_seconds=kl_time,mi_seconds=mi_time,ablation_seconds=time.perf_counter()-t+kl_time,
        reference_seconds=time.perf_counter()-start)

def run(seed,out,steps=3072,switch=1024):
    torch.manual_seed(seed);train,val,vocab=load_data();net=TextVAE(vocab)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    rng=torch.Generator().manual_seed(seed+191);probe_eps=torch.randn(16,16,generator=torch.Generator().manual_seed(829))
    ref_eps=torch.randn(4,len(val),16,generator=torch.Generator().manual_seed(830))
    out.mkdir(parents=True,exist_ok=True);saved=None
    for arm in ['base','regularized']:
        first=0;beta=.01;logs=[];refs=[]
        if arm=='regularized':
            net.load_state_dict(saved['net']);opt.load_state_dict(saved['opt']);rng.set_state(saved['rng'])
            first=switch;beta=1.
            logs=pd.read_csv(out/'base/logs.csv').query('step<@switch').to_dict('records')
            refs=pd.read_csv(out/'base/reference.csv').query('step<@switch').to_dict('records')
        d=out/arm;d.mkdir(exist_ok=True);start=time.perf_counter()
        for t in range(first,steps):
            if t==switch and arm=='base':
                saved=dict(net=copy.deepcopy(net.state_dict()),opt=copy.deepcopy(opt.state_dict()),rng=rng.get_state())
                torch.save(saved,out/'branch.pt')
            if t%128==0:
                stat=reference(net,val,ref_eps);refs.append(dict(step=t,**stat))
                pd.DataFrame(refs).to_csv(d/'reference.csv',index=False)
                print(seed,arm,t,'KL',round(stat['KL'],3),'MI',round(stat['MI'],3),'shuffle',round(stat['shuffle_symkl'],5),flush=True)
            a=time.perf_counter()
            with torch.no_grad():
                ce,_=net(val[:16],probe_eps);probe=float(ce.sum()/(val[:16]!=0).sum())
            probe_seconds=time.perf_counter()-a
            ix=torch.randint(len(train),(32,),generator=rng);x=train[ix];eps=torch.randn(32,16,generator=rng)
            a=time.perf_counter();opt.zero_grad(set_to_none=True);ce,kl=net(x,eps)
            loss=ce.sum(1).mean()+beta*kl.mean();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            assert torch.isfinite(loss),f'Nonfinite training objective at{t}'
            logs.append(dict(step=t,probe_nll=probe,train_nll=float(ce.detach().sum()/(x!=0).sum()),
                train_KL=float(kl.detach().mean()),beta=beta,train_seconds=time.perf_counter()-a,probe_seconds=probe_seconds))
            if (t+1)%128==0:pd.DataFrame(logs).to_csv(d/'logs.csv',index=False)
        refs.append(dict(step=steps,**reference(net,val,ref_eps)))
        pd.DataFrame(refs).to_csv(d/'reference.csv',index=False);pd.DataFrame(logs).to_csv(d/'logs.csv',index=False)
        torch.save(net.state_dict(),d/'final.pt')
        (d/'meta.json').write_text(json.dumps(dict(seed=seed,arm=arm,beta=beta,steps=steps,switch=switch,
            parameters=sum(p.numel() for p in net.parameters()),seconds=time.perf_counter()-start),indent=2))
    a=pd.read_csv(out/'base/logs.csv');b=pd.read_csv(out/'regularized/logs.csv')
    assert np.array_equal(a.probe_nll[:switch],b.probe_nll[:switch])
    assert np.isclose(a.probe_nll[switch],b.probe_nll[switch])
    print('Complete; paired prefix and branch observation match.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--out',type=Path,default=H/'pilot_seed0')
    p.add_argument('--steps',type=int,default=3072);p.add_argument('--switch',type=int,default=1024);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):run(a.seed,a.out,a.steps,a.switch)
