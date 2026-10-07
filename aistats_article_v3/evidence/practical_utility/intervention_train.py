from pathlib import Path
import sys,json,time,copy,argparse,hashlib
import numpy as np,pandas as pd,torch
from torch import nn
from torchvision.datasets import MNIST
from threadpoolctl import threadpool_limits
SOURCE=Path(__file__).resolve().parent;R=SOURCE.parent;H=SOURCE/'v2';H.mkdir(exist_ok=True);sys.path.insert(0,str(R/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
STEPS=[256,512,768,1024,1536];TOTAL=2048
CFG=EstimatorConfig(max_E=10,tau=1,k_neighbors=10,theiler=19,theiler_cap=19)
def data():
 file=SOURCE/'data.npz'
 if not file.exists():
  a=MNIST(R/'research_gan_collapse/data',train=True,download=True);b=MNIST(R/'research_gan_collapse/data',train=False,download=True);rng=np.random.default_rng(1006)
  ids=[rng.permutation(np.flatnonzero(a.targets.numpy()==c))[:250] for c in range(10)];tr=np.concatenate([v[:200] for v in ids]);va=np.concatenate([v[200:] for v in ids]);te=np.concatenate([rng.choice(np.flatnonzero(b.targets.numpy()==c),200,False) for c in range(10)])
  convert=lambda x:torch.nn.functional.avg_pool2d(x[:,None].float()/255,2).flatten(1).numpy()
  np.savez_compressed(file,x=convert(a.data[tr]),y=a.targets[tr].numpy(),vx=convert(a.data[va]),vy=a.targets[va].numpy(),tx=convert(b.data[te]),ty=b.targets[te].numpy(),train_ids=tr,val_ids=va,test_ids=te)
 d=np.load(file);assert not set(d['train_ids']).intersection(d['val_ids']);return {k:torch.from_numpy(d[k]) for k in ['x','y','vx','vy','tx','ty']}
def model(width=128):return nn.Sequential(nn.Linear(196,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,10))
def labels(y,seed,noise):
 rng=np.random.default_rng(9000+seed);out=y.clone();ids=rng.permutation(len(y))[:int(len(y)*noise)];out[ids]=(out[ids]+torch.tensor(rng.integers(1,10,len(ids))))%10;return out
@torch.no_grad()
def quality(net,d,y):
 return dict(val_acc=float((net(d['vx']).argmax(1)==d['vy']).float().mean()),test_acc=float((net(d['tx']).argmax(1)==d['ty']).float().mean()),train_clean_acc=float((net(d['x']).argmax(1)==d['y']).float().mean()),train_noisy_acc=float((net(d['x']).argmax(1)==y).float().mean()))
def features(logs):
 rows=[]
 for step in STEPS:
  for obs in ['norm','probe']:
   x=np.array([r[obs] for r in logs[step-256:step]]);z=x-x.mean();sd=x.std();t=np.linspace(-1,1,len(x));slope=np.dot(t,z)/np.dot(t,t);res=z-slope*t;v=max(sd**2,1e-30);p=abs(np.fft.rfft(res))[1:]**2;p/=max(p.sum(),1e-30)
   tic=time.perf_counter();est=estimate(x,CFG,seed=123);elapsed=time.perf_counter()-tic
   rows.append(dict(step=step,obs=obs,MG=est.MG if not est.degenerate else np.nan,MG_seconds=elapsed,degenerate=est.degenerate,level=float(x.mean()),slope=float(slope/(abs(x.mean())+1e-12)),std=float(res.std()/(abs(x.mean())+1e-12)),increments=float(np.mean(np.diff(x)**2)/(2*v)),entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p))),lag1=float(np.corrcoef(res[:-1],res[1:])[0,1])))
 return rows
def base(seed,noise):
 folder=H/f'seed{seed}_noise{noise:g}';folder.mkdir(exist_ok=True)
 if (folder/'features.csv').exists():return
 d=data();y=labels(d['y'],seed,noise);torch.manual_seed(seed);net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);rng=torch.Generator().manual_seed(seed+7000);logs=[];metrics=[]
 probe_ids=torch.randperm(len(y),generator=torch.Generator().manual_seed(123456))[:128]
 for step in range(TOTAL+1):
  if step in [0]+STEPS+[TOTAL]:
   metrics.append(dict(step=step,**quality(net,d,y)));torch.save(dict(net=net.state_dict(),opt=opt.state_dict(),rng=rng.get_state(),step=step),folder/f'checkpoint{step}.pt')
  if step==TOTAL:break
  tic=time.perf_counter()
  with torch.no_grad():probe=float(nn.functional.cross_entropy(net(d['x'][probe_ids]),y[probe_ids]));norm=float(torch.sqrt(sum(p.square().sum() for p in net.parameters())))
  monitor=time.perf_counter()-tic;ids=torch.randint(len(y),(64,),generator=rng);tic=time.perf_counter();opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(net(d['x'][ids]),y[ids]);loss.backward();opt.step();elapsed=time.perf_counter()-tic
  logs.append(dict(step=step+1,loss=float(loss.detach()),norm=norm,probe=probe,train_seconds=elapsed,probe_seconds=monitor))
 pd.DataFrame(logs).to_csv(folder/'logs.csv',index=False);pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(features(logs)).to_csv(folder/'features.csv',index=False);np.save(folder/'noisy_labels.npy',y.numpy());print('base',seed,noise,metrics[-1],flush=True)
def compress(net,opt):
 oldpars=list(net.parameters());oldstate=[copy.deepcopy(opt.state[p]) for p in oldpars]
 with torch.no_grad():
  i=torch.topk(net[0].weight.norm(dim=1)*net[2].weight.norm(dim=0),64).indices.sort().values
  j=torch.topk(net[2].weight.norm(dim=1)*net[4].weight.norm(dim=0),64).indices.sort().values
  transforms=[lambda a:a[i],lambda a:a[i],lambda a:a[j][:,i],lambda a:a[j],lambda a:a[:,j],lambda a:a]
  small=model(64)
  for dst,src,fn in zip(small.parameters(),oldpars,transforms):dst.copy_(fn(src))
 newopt=torch.optim.Adam(small.parameters(),lr=.001)
 for p,s,fn in zip(small.parameters(),oldstate,transforms):newopt.state[p]={k:(fn(v).clone() if torch.is_tensor(v) and v.ndim>0 else v) for k,v in s.items()}
 return small,newopt
def branch(seed,action,step):
 folder=H/f'seed{seed}_noise0';out=folder/f'{action}_{step}.json'
 if out.exists():return json.loads(out.read_text())
 d=data();net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);ck=torch.load(folder/f'checkpoint{step}.pt',weights_only=False);net.load_state_dict(ck['net']);opt.load_state_dict(ck['opt']);rng=torch.Generator();rng.set_state(ck['rng'])
 before=quality(net,d,d['y'])
 if action=='freeze':
  for layer in [net[0],net[2]]:
   for p in layer.parameters():p.requires_grad_(False)
 elif action=='prune':net,opt=compress(net,opt)
 after=quality(net,d,d['y']);train_seconds=0.
 for t in range(step,TOTAL):
  ids=torch.randint(len(d['y']),(64,),generator=rng);tic=time.perf_counter();opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(net(d['x'][ids]),d['y'][ids]);loss.backward();opt.step();train_seconds+=time.perf_counter()-tic
 full=3*(196*128+128*128+128*10);later=(196*128+128*128+3*128*10) if action=='freeze' else 3*(196*64+64*64+64*10)
 cost=(step*full+(TOTAL-step)*later)/(TOTAL*full)
 row=dict(seed=seed,action=action,step=step,**quality(net,d,d['y']),mac_ratio=cost,train_seconds=train_seconds,parameters=sum(p.numel() for p in net.parameters()),before=before,immediate=after)
 out.write_text(json.dumps(row,indent=2));torch.save(net.state_dict(),folder/f'{action}_{step}_final.pt');print('branch',seed,action,step,row['test_acc'],flush=True);return row
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--pilot',action='store_true');args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 with threadpool_limits(limits=1):
  for s in args.seeds:
   for noise in [0,.4,.6]:base(s,noise)
   if args.pilot:
    for action in ['freeze','prune']:
     for t in STEPS:branch(s,action,t)
