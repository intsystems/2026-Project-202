"""Matched 8192 samples, sequential warmed CPU timings, three repetitions."""
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyze import HERE,mg,spectrum,return_residual,stats
from run import step,top_lyap

def record(a,x,length,kind):
    x=x.copy()
    for _ in range(2000):x=step(x,a)
    result=np.empty((length,len(x))) if kind=='full' else np.empty(length)
    t=time.perf_counter()
    for i in range(length):
        if kind=='full':result[i]=x
        elif kind=='scalar':result[i]=np.tanh(x[0])
        x=step(x,a)
    return time.perf_counter()-t

if __name__=='__main__':
    root=HERE/'results_chaotic';rows=[]
    with threadpool_limits(limits=1):
        for stage in [0,40000]:
            d=root/'seed_7';c=np.load(d/f'checkpoint_{stage:05d}.npz')
            a=c['j']+c['u']@c['w'].T
            states=np.load(d/f'rollout_{stage:05d}.npz')['states']
            signal=np.tanh(states[:,0])
            mg(signal[:4096]);spectrum(a,states[:1200],burn=100)
            for rep in range(3):
                logs={k:record(a,c['x'],8192,k) for k in ['none','scalar','full']}
                v=[mg(signal[:4096]),mg(signal[4096:])]
                _,_,full=spectrum(a,states)
                _,top=top_lyap(a,states)
                t=time.perf_counter();return_residual(states);ret=time.perf_counter()-t
                rows.append(dict(training_step=stage,repeat=rep,**{f'record_{k}':v for k,v in logs.items()},
                    mg=sum(x['mg_seconds'] for x in v),mg_E20_E40=sum(x['checks_seconds'] for x in v),
                    scalar_baselines=sum(x['cheap_seconds'] for x in v),
                    full_spectrum=full,top_exponent=top,full_return=ret))
                pd.DataFrame(rows).to_csv(root/'benchmark.csv',index=False)
                print(rows[-1],flush=True)
    (root/'storage.json').write_text(json.dumps(dict(samples=8192,neurons=256,dtype='float64',
        scalar_record_bytes=8192*8,full_record_bytes=8192*256*8,
        warning='record storage only, not estimator peak memory'),indent=2))
