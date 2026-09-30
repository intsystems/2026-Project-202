"""Warmed, interleaved local measurements; no parallel-run timings used."""
from pathlib import Path
import json,platform,sys,time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from run import TextVAE,load_data,reference
from features import estimate,CFG,spectral_entropy
from monitor import policy

def main():
    _,val,vocab=load_data();net=TextVAE(vocab)
    net.load_state_dict(torch.load(H/'seed101/final.pt',weights_only=True))
    pe=torch.randn(16,16,generator=torch.Generator().manual_seed(829))
    re=torch.randn(4,len(val),16,generator=torch.Generator().manual_seed(830))
    logs=pd.read_csv(H/'seed101/logs.csv');feat=pd.read_csv(H/'seed101/features.csv')
    windows=[logs.probe_nll.iloc[end-512:end].to_numpy() for end in range(1024,7169,256)]
    @torch.no_grad()
    def probe():
        ce,_=net(val[:16],pe);return float(ce.sum()/(val[:16]!=0).sum())
    def kl_aggregate():
        a=logs.train_KL.to_numpy()
        return [float(a[end-64:end].mean()) for end in range(512,7169,64)]
    def beta_aggregate():
        a=logs.beta.to_numpy()
        return [max(.01,float(a[end-1])) for end in range(512,7169,64)]
    functions={
        'probe':lambda i:probe(),
        'reference':lambda i:reference(net,val,re),
        'MG':lambda i:estimate(windows[i%len(windows)],CFG,seed=123),
        'std':lambda i:float(windows[i%len(windows)].std()),
        'entropy':lambda i:spectral_entropy(windows[i%len(windows)]),
        'KL_all105':lambda i:kl_aggregate(),
        'beta_all105':lambda i:beta_aggregate(),
        'policy_all96':lambda i:policy(feat,'MG',.2,12),
    }
    # Warm imports, kernels and neighbour library before timing.
    for _ in range(3):
        for name,fn in functions.items():
            with threadpool_limits(limits=1 if name in ['MG','std','entropy'] else 2):fn(0)
    rng=np.random.default_rng(20260929);rows=[]
    for repeat in range(40):
        for name in rng.permutation(list(functions)):
            inner=20 if name in ['std','entropy','KL_all105','beta_all105','policy_all96'] else (5 if name=='probe' else 1)
            with threadpool_limits(limits=1 if name in ['MG','std','entropy'] else 2):
                started=time.perf_counter()
                for i in range(inner):functions[name](repeat+i)
                elapsed=(time.perf_counter()-started)/inner
            rows.append(dict(repeat=repeat,operation=name,seconds=elapsed,inner=inner))
    df=pd.DataFrame(rows);df.to_csv(H/'timings.csv',index=False)
    timing={name:dict(median=float(g.seconds.median()),q25=float(g.seconds.quantile(.25)),q75=float(g.seconds.quantile(.75)),n=40)
        for name,g in df.groupby('operation')}
    result=dict(seed=101,checkpoint='final.pt',device='cpu',torch_threads=2,estimator_threads=1,
        platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,operations=timing,
        caveat='One process, warmed interleaved tasks; another user job may share the host. These are local component-cost estimates, not paired end-to-end online runs.')
    (H/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(timing,indent=2))

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):main()
