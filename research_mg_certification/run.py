from pathlib import Path
import sys,json,time,argparse,hashlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'research_force_motion'))
from run import init,step,target,task,DT
from analyze import spectrum,return_residual
sys.path.insert(0,str(R/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG=EstimatorConfig(max_E=20,tau=4,k_neighbors=20,theiler=156,theiler_cap=156)
STEPS=[0,200,500,1000,2000,4000,8000]

def evaluate(j,u,w,x,it):
    tic=time.perf_counter();a=j+u@w.T;x=x.copy()
    for _ in range(1024):x=step(x,a)
    states=np.empty((2048,len(x)))
    for t in range(2048):states[t]=x;x=step(x,a)
    row={'rollout_seconds':time.perf_counter()-tic}
    z=np.tanh(states)@w;obs=states[:,0];ts=(it+1024+np.arange(2048))*DT
    funcs={'MG':lambda:estimate(obs,CFG,seed=123),'entropy':lambda:entropy(obs),
           'increments':lambda:np.mean(np.diff(obs)**2)/(2*np.var(obs)),
           'recurrence':lambda:return_residual(obs)[0],
           'output_error':lambda:task(z,ts)['aligned_nrmse']}
    for name,fn in funcs.items():
        tic=time.perf_counter();v=fn();row[name+'_seconds']=time.perf_counter()-tic
        if name=='MG':row.update(MG=float(v.MG) if not v.degenerate else np.nan,degenerate=bool(v.degenerate))
        else:row[name]=float(v)
    tic=time.perf_counter();ls,half,sec=spectrum(a,states,burn=512)
    ref,period,var=return_residual(states)
    error=task(z,ts)['aligned_nrmse']
    row.update(reference_seconds=time.perf_counter()-tic,lambda1=float(ls[0]),lambda2=float(ls[1]),full_return=ref,
               qualified=bool(error<.1 and ref<.1 and abs(ls[0])<.005 and ls[1]<-.005))
    return row,states,ls
def entropy(x):
    p=np.abs(np.fft.rfft(x-x.mean()))[1:]**2;p/=max(p.sum(),1e-30)
    return -np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p))
def run(seed):
    out=H/f'seed{seed}';out.mkdir(exist_ok=True)
    if (out/'records.csv').exists():return
    j,u,x=init(seed,96);w=np.zeros((96,2));p=np.eye(96);rows=[];errs=[];elapsed=0.
    for it in range(STEPS[-1]+1):
        if it in STEPS:
            row,states,ls=evaluate(j,u,w,x,it)
            row.update(seed=seed,step=it,training_seconds=elapsed,training_error=float(np.mean(errs[-100:])) if errs else np.nan,training_error_seconds=0.)
            rows.append(row);np.savez_compressed(out/f'checkpoint{it}.npz',j=j,u=u,w=w,x=x,states=states,spectrum=ls)
            print(seed,it,'err',round(row['output_error'],3),'MG',round(row['MG'],3),'pass',row['qualified'],flush=True)
        if it==STEPS[-1]:break
        tic=time.perf_counter();x=(1-DT)*x+DT*(j@np.tanh(x)+u@(w.T@np.tanh(x)));r=np.tanh(x)
        err=r@w-target((it+1)*DT)
        if (it+1)%2==0:
            pr=p@r;den=1+r@pr;w-=np.outer(pr/den,err);p-=np.outer(pr,pr)/den
        errs.append(float(err@err/2));elapsed+=time.perf_counter()-tic
    pd.DataFrame(rows).to_csv(out/'records.csv',index=False)
    (out/'meta.json').write_text(json.dumps(dict(seed=seed,n=96,steps=STEPS,protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True);a=ap.parse_args()
    with threadpool_limits(limits=1):
        for seed in a.seeds:run(seed)
