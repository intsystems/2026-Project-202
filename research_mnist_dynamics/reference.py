from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent

def residual(x):
    x=np.array(x,dtype=np.float64);x-=x.mean(0)
    t=np.arange(len(x),dtype=float);t-=t.mean();x-=np.outer(t,t@x/(t@t))
    return x

def pr(x,detrend=True):
    x=residual(x) if detrend else np.asarray(x,dtype=float)-np.mean(x,axis=0)
    g=x@x.T;trace=float(np.trace(g));den=float(np.sum(g*g))
    return (trace*trace/den if den>0 else np.nan),trace/len(x)

def analyze(root):
    for d in root.rglob('base'):
        if not (d/'trajectory.npy').exists():continue
        for name in ['base','drop']:
            arm=d.parent/name;traj=np.load(arm/'trajectory.npy',mmap_mode='r');P=traj.shape[1]
            idx=np.random.default_rng(718).choice(P,128,False);rows=[]
            for end in range(512,len(traj)+1,256):
                x=traj[end-512:end];tic=time.perf_counter();v,var=pr(x);seconds=time.perf_counter()-tic
                tic=time.perf_counter();small,sv=pr(x[:,idx]);ss=time.perf_counter()-tic
                raw,_=pr(x,False);update,_=pr(np.diff(x.astype(float),axis=0))
                rows.append(dict(end=end,pr=v,variance=var,raw_pr=raw,update_pr=update,
                    small_pr=small,ref_seconds=seconds,small_seconds=ss))
                if end in [2048,4096]:
                    xx=residual(x);s=np.maximum(np.linalg.eigvalsh(xx@xx.T),0)[::-1]
                    np.save(arm/f'spectrum_{end}.npy',s)
            df=pd.DataFrame(rows);df.to_csv(arm/'reference.csv',index=False)
            pre=df[df.end.between(1024,2048)].pr.median();post=df[df.end>=3072].pr.median()
            acc=pd.read_csv(arm/'accuracy.csv').iloc[-1]
            print(arm,'PR',pre,post,'ratio',post/pre,'test',acc.test_acc,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):analyze(a.root)
