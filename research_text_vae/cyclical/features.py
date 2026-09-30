from pathlib import Path
import argparse,json,sys,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent.parent/'code'))
from actdim.estimator.mle import estimate
from actdim.estimator.config import EstimatorConfig
CFG=EstimatorConfig(window=512,max_E=20,tau=1,k_neighbors=20,theiler=39,theiler_cap=39)

def spectral_entropy(x):
    power=np.abs(np.fft.rfft(x-x.mean()))[1:]**2
    power/=max(power.sum(),1e-30)
    return float(-np.sum(power*np.log(np.maximum(power,1e-30)))/np.log(len(power)))

def features(seed):
    out=H/f'seed{seed}';meta=json.loads((out/'meta.json').read_text());log=pd.read_csv(out/'logs.csv')
    assert meta['steps']==len(log)==7168
    rows=[];x=log.probe_nll.to_numpy()
    for end in range(512,len(x)+1,64):
        z=x[end-512:end];tic=time.perf_counter();est=estimate(z,CFG,seed=123);elapsed=time.perf_counter()-tic
        diag=estimate(z,CFG.replace(max_E=40),seed=123)
        rows.append(dict(end=end,MG=est.MG,MG40=diag.MG,ident=diag.MG/est.MG,degenerate=est.degenerate,
            MG_seconds=elapsed,std=float(z.std()),entropy=spectral_entropy(z),
            KL=float(log.train_KL.iloc[end-64:end].mean()),beta=max(.01,float(log.beta.iloc[end-1]))))
    frame=pd.DataFrame(rows);frame.to_csv(out/'features.csv',index=False)
    print('FEATURES',seed,len(frame),'degenerate',int(frame.degenerate.sum()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);a=p.parse_args()
    estimate(np.sin(np.arange(512)/11),CFG,seed=123)
    with threadpool_limits(limits=1):features(a.seed)
