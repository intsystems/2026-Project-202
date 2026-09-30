"""Train and acquire independent feature geometry. No MG-dependent choices."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from model import SmallResNet

HERE = Path(__file__).resolve().parent


def read_cifar(root):
    def read(name):
        with (root/name).open('rb') as f:
            b = pickle.load(f, encoding='bytes')
        return b[b'data'].reshape(-1, 3, 32, 32), np.asarray(b[b'labels'])
    train = [read(f'data_batch_{i}') for i in range(1, 6)]
    x = np.concatenate([a for a,b in train]); y = np.concatenate([b for a,b in train])
    tx,ty = read('test_batch')
    return x,y,tx,ty


def prepare(root):
    x,y,tx,ty = read_cifar(root)
    rng = np.random.default_rng(20260929)
    ids = np.concatenate([rng.choice(np.flatnonzero(y==c), 200, replace=False) for c in range(10)])
    tids = np.concatenate([rng.choice(np.flatnonzero(ty==c), 100, replace=False) for c in range(10)])
    mean = torch.tensor([.4914,.4822,.4465])[None,:,None,None]
    std = torch.tensor([.2470,.2435,.2616])[None,:,None,None]
    def convert(a):
        t = (torch.from_numpy(a.copy()).float()/255 - mean)/std
        return t.contiguous(memory_format=torch.channels_last)
    return convert(x[ids]), torch.from_numpy(y[ids]), convert(tx[tids]), torch.from_numpy(ty[tids]), ids,tids


def geometry(features, labels, weights):
    h = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    means = np.stack([h[y==c].mean(0) for c in range(10)])
    center = means.mean(0); m = means-center
    residual = h-means[y]
    sw = residual.T@residual/len(h)
    sb = m.T@m/10
    nc1 = float(np.trace(sw@np.linalg.pinv(sb, rcond=1e-10))/10)
    nc1_trace = float(np.trace(sw)/max(np.trace(sb),1e-30))
    norms = np.linalg.norm(m, axis=1)
    normalized = m/np.maximum(norms[:,None],1e-30)
    gram = normalized@normalized.T
    etf = (10*np.eye(10)-np.ones((10,10)))/9
    nc2 = float(np.linalg.norm(gram-etf,'fro')/np.linalg.norm(etf,'fro'))
    w = np.asarray(weights,dtype=np.float64)
    nc3 = float(np.linalg.norm(w/max(np.linalg.norm(w),1e-30)-m/max(np.linalg.norm(m),1e-30)))
    return dict(nc1=nc1, nc1_trace=nc1_trace, nc2=nc2,
                mean_norm_cv=float(norms.std()/max(norms.mean(),1e-30)), nc3=nc3), means


@torch.no_grad()
def extract(model, x, y, batch=128):
    model.eval()
    zs,hs = [],[]
    for offset in range(0,len(y),batch):
        z,h = model(x[offset:offset+batch],return_features=True)
        zs.append(z); hs.append(h)
    z = torch.cat(zs); h = torch.cat(hs)
    return z,h,float(nn.functional.cross_entropy(z,y)),float((z.argmax(1)==y).float().mean())


def reference(model,x,y,tx,ty,step,out):
    t = time.perf_counter()
    z,h,ce,acc = extract(model,x,y)
    extract_seconds = time.perf_counter()-t
    t = time.perf_counter()
    metrics,means = geometry(h.numpy(),y.numpy(),model.head.weight.detach().numpy())
    distances = ((h.numpy()[:,None,:]-means[None,:,:])**2).sum(2)
    pred_nc = distances.argmin(1)
    metrics['nc4'] = float(np.mean(pred_nc!=z.argmax(1).numpy()))
    metrics['ncc_accuracy'] = float(np.mean(pred_nc==y.numpy()))
    geometry_seconds = time.perf_counter()-t
    t = time.perf_counter()
    tz,th,tce,tacc = extract(model,tx,ty)
    test_seconds = time.perf_counter()-t
    tdist = ((th.numpy()[:,None,:]-means[None,:,:])**2).sum(2)
    metrics['test_nc4'] = float(np.mean(tdist.argmin(1)!=tz.argmax(1).numpy()))
    metrics['test_ncc_accuracy'] = float(np.mean(tdist.argmin(1)==ty.numpy()))
    metrics.update(step=step,train_ce=ce,train_accuracy=acc,test_ce=tce,test_accuracy=tacc,
                   extract_seconds=extract_seconds,geometry_seconds=geometry_seconds,
                   nc_total_seconds=extract_seconds+geometry_seconds,test_seconds=test_seconds)
    np.savez_compressed(out/'features'/f'step_{step:05d}.npz', h=h.numpy(),
                        labels=y.numpy(), weights=model.head.weight.detach().numpy())
    return metrics


def run(args, seed, arm, data):
    x,y,tx,ty,ids,tids = data
    out = args.out/f'{arm}_s{seed}'
    if (out/'metadata.json').exists():
        print('Already complete:',out,flush=True); return
    (out/'features').mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    model = SmallResNet().to(memory_format=torch.channels_last)
    if arm=='frozen':
        for p in model.features.parameters(): p.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(parameters,lr=args.lr,momentum=.9,weight_decay=.0005)
    gen = torch.Generator().manual_seed(10000+seed)
    probe_ids = torch.cat([torch.arange(200*c,200*c+10) for c in range(10)])
    test_probe_ids = torch.cat([torch.arange(100*c,100*c+10) for c in range(10)])
    px,py = x[probe_ids],y[probe_ids]
    ptx,pty = tx[test_probe_ids],ty[test_probe_ids]
    both = torch.cat([px,ptx]).contiguous(memory_format=torch.channels_last)
    meta=dict(seed=seed,arm=arm,steps=args.steps,lr=args.lr,batch=64,train_n=len(y),test_n=len(ty),
              probe_n=100,reference_every=128,parameters=sum(p.numel() for p in model.parameters()),
              trainable_parameters=sum(p.numel() for p in parameters),torch=torch.__version__,
              numpy=np.__version__,device='cpu',threads=args.threads,platform=platform.platform(),
              split_seed=20260929,augmentation=False,momentum=.9,weight_decay=.0005,
              feature_dimension=64,normalization_mean=[.4914,.4822,.4465],
              normalization_std=[.2470,.2435,.2616],constant_lr=True)
    (out/'config.json').write_text(json.dumps(meta,indent=2))
    np.savez(out/'indices.npz',train=ids,test=tids,probe_local=probe_ids.numpy(),test_probe_local=test_probe_ids.numpy())
    refs=[reference(model,x,y,tx,ty,0,out)]
    trace=[];small=[]
    train_seconds=probe_seconds=0.
    start=time.perf_counter()
    for step in range(1,args.steps+1):
        t=time.perf_counter()
        model.train()
        if arm=='frozen': model.features.eval()
        batch=torch.randint(len(y),(64,),generator=gen)
        opt.zero_grad(set_to_none=True)
        z=model(x[batch]); loss=nn.functional.cross_entropy(z,y[batch])
        loss.backward(); opt.step()
        training_time=time.perf_counter()-t;train_seconds+=training_time
        t=time.perf_counter()
        with torch.no_grad():
            model.eval()
            pz,ph=model(both,return_features=True)
            probe_loss=float(nn.functional.cross_entropy(pz[:100],py))
            test_probe_loss=float(nn.functional.cross_entropy(pz[100:],pty))
        probe_time=time.perf_counter()-t;probe_seconds+=probe_time
        trace.append(dict(step=step,train_loss=float(loss.detach()),probe_loss=probe_loss,
                          test_probe_loss=test_probe_loss,training_seconds=training_time,
                          joint_probe_seconds=probe_time,lr=args.lr))
        if step%128==0 or step==args.steps:
            t=time.perf_counter()
            sm,_=geometry(ph[:100].numpy(),py.numpy(),model.head.weight.detach().numpy())
            sm.update(step=step,seconds=time.perf_counter()-t)
            small.append(sm)
            ref=reference(model,x,y,tx,ty,step,out);refs.append(ref)
            pd.DataFrame(trace).to_csv(out/'trace.csv',index=False)
            pd.DataFrame(refs).to_csv(out/'reference.csv',index=False)
            pd.DataFrame(small).to_csv(out/'small_probe_reference.csv',index=False)
            print(f'{arm} s{seed} step {step}: train {ref["train_accuracy"]:.3f}, test {ref["test_accuracy"]:.3f}, CE {ref["train_ce"]:.4f}, NC1 {ref["nc1"]:.4f}, NC2 {ref["nc2"]:.3f}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    # Separate warmed forward timings for one 100-example scalar probe.
    timing=[]
    for repeat in range(7):
        t=time.perf_counter()
        with torch.no_grad(): _=float(nn.functional.cross_entropy(model(px),py))
        timing.append(time.perf_counter()-t)
    meta.update(wall_seconds=time.perf_counter()-start,train_seconds=train_seconds,
                joint_probe_seconds=probe_seconds,one_probe_forward_repeats=timing,
                one_probe_forward_seconds=float(np.median(timing[2:])),completed_steps=len(trace))
    torch.save(model.state_dict(),out/'final.pt')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETE',out,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'results')
    parser.add_argument('--steps',type=int,default=4096)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--arm',choices=['train','frozen'],default='train')
    parser.add_argument('--lr',type=float,default=.03)
    parser.add_argument('--threads',type=int,default=8)
    a=parser.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    a.out.mkdir(parents=True,exist_ok=True)
    files=[a.data/f'data_batch_{i}' for i in range(1,6)]+[a.data/'test_batch']
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (a.out/'data_sha256.json').write_text(json.dumps(provenance,indent=2))
    with threadpool_limits(limits=1,user_api='blas'):
        data=prepare(a.data)
        for seed in a.seeds:run(a,seed,a.arm,data)


if __name__=='__main__':main()
