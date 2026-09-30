"""Sequential warmed end-to-end benchmark on final trained checkpoints."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from model import SmallResNet
from run import prepare, extract, geometry
from analyze import score, scalar_stats


def timed(fn):
    times=[]
    for i in range(7):
        start=time.perf_counter();fn();times.append(time.perf_counter()-start)
    return times


def main(a):
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1,user_api='blas'):
        x,y,tx,ty,*_=prepare(a.data)
        ids=torch.cat([torch.arange(200*c,200*c+10) for c in range(10)])
        rows=[]
        for seed in [0,1,2]:
            d=a.root/f'train_s{seed}'
            model=SmallResNet().to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(d/'final.pt',map_location='cpu',weights_only=True))
            model.eval()
            @torch.no_grad()
            def full():
                z,h,*_=extract(model,x,y)
                metrics,means=geometry(h.numpy(),y.numpy(),model.head.weight.numpy())
                _=(((h.numpy()[:,None,:]-means[None,:,:])**2).sum(2).argmin(1)!=z.argmax(1).numpy()).mean()
            @torch.no_grad()
            def small():
                z,h,*_=extract(model,x[ids],y[ids])
                metrics,means=geometry(h.numpy(),y[ids].numpy(),model.head.weight.numpy())
                _=(((h.numpy()[:,None,:]-means[None,:,:])**2).sum(2).argmin(1)!=z.argmax(1).numpy()).mean()
            @torch.no_grad()
            def probe():
                return float(torch.nn.functional.cross_entropy(model(x[ids]),y[ids]))
            tr=pd.read_csv(d/'trace.csv');signal=tr.probe_loss.to_numpy()[-512:]
            for name,fn in [('full_nc',full),('small_nc',small),('one_probe',probe),('cheap_stats',lambda:scalar_stats(signal))]:
                times=timed(fn)
                rows.append(dict(run=d.name,method=name,median_seconds=float(np.median(times[2:])),
                                 repeats=json.dumps(times),warmup_repeats=2,threads_torch=8,threads_blas=1))
            print('Benchmark',d.name,flush=True)
        pd.DataFrame(rows).to_csv(a.root/'warmed_benchmark.csv',index=False)
        print(pd.DataFrame(rows)[['run','method','median_seconds']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).parent/'results');main(p.parse_args())
