from pathlib import Path
import argparse,copy,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import MNIST
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent

def data():
    root=H.parent/'research_gan_collapse/data'
    a=MNIST(root,train=True,download=True);b=MNIST(root,train=False,download=True)
    rng=np.random.default_rng(260929)
    ids=np.concatenate([rng.choice(np.flatnonzero(a.targets.numpy()==c),204,False) for c in range(10)])
    def conv(x):return torch.nn.functional.avg_pool2d(x[:,None].float()/255,2).flatten(1)
    return conv(a.data[ids]),a.targets[ids],conv(b.data),b.targets,ids

def model():return nn.Sequential(nn.Linear(196,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,10))

def run(seed,lr,batch,momentum,out,steps=4096,switch=2048):
    out.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
    x,y,tx,ty,ids=data();np.save(out/'train_ids.npy',ids)
    net=model();opt=torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum)
    params=list(net.parameters());P=sum(p.numel() for p in params)
    observer=np.random.default_rng(1122).choice([-1.,1.],size=P)/np.sqrt(P)
    rng=torch.Generator().manual_seed(seed+1000)
    saved=None
    for arm in ['base','drop']:
        if arm=='drop':
            net.load_state_dict(saved['model']);opt.load_state_dict(saved['opt']);rng.set_state(saved['rng'])
            for group in opt.param_groups:group['lr']=lr/10
        d=out/arm;d.mkdir(exist_ok=True)
        trajectory=np.lib.format.open_memmap(d/'trajectory.npy',mode='w+',dtype=np.float32,shape=(steps,P))
        rows=[];metrics=[];first=0
        if arm=='drop':
            trajectory[:switch]=np.load(out/'base/trajectory.npy',mmap_mode='r')[:switch]
            rows=pd.read_csv(out/'base/logs.csv').iloc[:switch].to_dict('records')
            metrics=pd.read_csv(out/'base/accuracy.csv').query('step<=@switch').to_dict('records');first=switch
        total=time.perf_counter()
        for t in range(first,steps):
            if t==switch and arm=='base':
                saved=dict(model=copy.deepcopy(net.state_dict()),opt=copy.deepcopy(opt.state_dict()),rng=rng.get_state())
                torch.save(saved,out/'branch.pt')
            if t%256==0:
                tic=time.perf_counter()
                with torch.no_grad():
                    logits=net(x);test=net(tx)
                    met=dict(step=t,train_loss=float(nn.functional.cross_entropy(logits,y)),
                        train_acc=float((logits.argmax(1)==y).float().mean()),
                        test_acc=float((test.argmax(1)==ty).float().mean()),seconds=time.perf_counter()-tic)
                if not metrics or metrics[-1]['step']!=t:metrics.append(met)
                print(seed,lr,batch,arm,t,round(met['train_acc'],3),round(met['test_acc'],3),flush=True)
            tic=time.perf_counter();flat=torch.cat([p.detach().flatten() for p in params]).numpy()
            trajectory[t]=flat;record_seconds=time.perf_counter()-tic
            tic=time.perf_counter();projection=float(flat@observer);projection_seconds=time.perf_counter()-tic
            ix=torch.randint(len(x),(batch,),generator=rng) if batch else slice(None)
            tic=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(net(x[ix]),y[ix]);loss.backward()
            grad=float(torch.sqrt(sum((p.grad**2).sum() for p in params)))
            opt.step();train_seconds=time.perf_counter()-tic
            rows.append(dict(step=t+1,loss=float(loss.detach()),grad_norm=grad,projection=projection,
                train_seconds=train_seconds,record_seconds=record_seconds,projection_seconds=projection_seconds))
        with torch.no_grad():metrics.append(dict(step=steps,train_loss=float(nn.functional.cross_entropy(net(x),y)),
            train_acc=float((net(x).argmax(1)==y).float().mean()),test_acc=float((net(tx).argmax(1)==ty).float().mean()),seconds=0))
        trajectory.flush();del trajectory
        pd.DataFrame(rows).to_csv(d/'logs.csv',index=False);pd.DataFrame(metrics).to_csv(d/'accuracy.csv',index=False)
        torch.save(net.state_dict(),d/'final.pt')
        (d/'meta.json').write_text(json.dumps(dict(seed=seed,lr=lr,batch=batch,momentum=momentum,arm=arm,
            steps=steps,switch=switch,P=P,seconds=time.perf_counter()-total,torch_threads=torch.get_num_threads()),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--batch',type=int,default=0);p.add_argument('--momentum',type=float,default=0)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run(a.seed,a.lr,a.batch,a.momentum,a.out)
