"""FORCE autonomous figure-eight learning; pilot judged without MG."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
DT=.1
PERIOD=10*np.pi

def target(t):
    phase=2*np.pi*np.asarray(t)/PERIOD
    return np.stack([np.sin(phase),.5*np.sin(2*phase)],axis=-1)

def init(seed,n,gain=1.5):
    rng=np.random.default_rng(seed)
    j=rng.normal(size=(n,n))*gain/np.sqrt(n)
    u=rng.uniform(-1,1,size=(n,2))
    x=rng.normal(size=n)*.5
    return j,u,x

def step(x,a):
    return (1-DT)*x+DT*(a@np.tanh(x))

def task(z,t):
    phase=2*np.pi*t/PERIOD
    # Grid phase fit, full-record error; no frequency/time-warp fitting.
    phases=np.linspace(-np.pi,np.pi,721)[:-1]
    errors=[]
    for p in phases:
        f=np.stack([np.sin(phase+p),.5*np.sin(2*(phase+p))],axis=-1)
        errors.append(np.mean((z-f)**2))
    k=int(np.argmin(errors))
    raw=np.mean((z-target(t))**2)
    return dict(raw_nrmse=float(np.sqrt(raw/.3125)),
                aligned_nrmse=float(np.sqrt(errors[k]/.3125)),phase=float(phases[k]))

def recurrence(x):
    stride=4
    y=x[::stride]
    period=PERIOD/DT/stride
    lags=np.arange(int(.8*period),int(1.2*period)+1)
    denom=np.mean(np.sum((y-y.mean(0))**2,axis=1))
    vals=[np.mean(np.sum((y[lag:]-y[:-lag])**2,axis=1))/max(denom,1e-30) for lag in lags]
    best=int(np.argmin(vals))
    return dict(return_error=float(np.sqrt(vals[best])),period_steps=int(lags[best]*stride),
                state_variance=float(denom))

def rollout(a,w,x,tstart,length=8192,burn=2000):
    x=x.copy()
    for i in range(burn):x=step(x,a)
    states=np.empty((length,len(x)));z=np.empty((length,2))
    started=time.perf_counter()
    for i in range(length):
        states[i]=x;z[i]=np.tanh(x)@w;x=step(x,a)
    seconds=time.perf_counter()-started
    ts=tstart+(burn+np.arange(length))*DT
    return states,z,ts,seconds

def top_lyap(a,states,burn=1000):
    v=np.random.default_rng(832).normal(size=len(a));v/=np.linalg.norm(v)
    logs=[];started=time.perf_counter()
    for i,x in enumerate(states):
        d=1-np.tanh(x)**2
        v=(1-DT)*v+DT*(a@(d*v))
        norm=np.linalg.norm(v);v/=norm
        if i>=burn:logs.append(np.log(norm))
    return float(np.mean(logs)/DT),time.perf_counter()-started

def train(seed,n,out,checkpoints,gain=1.5):
    d=out/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
    j,u,x=init(seed,n,gain);w=np.zeros((n,2));p=np.eye(n)
    checkpointset=set(checkpoints);logs=[];rows=[]
    started=time.perf_counter()
    for it in range(max(checkpoints)+1):
        if it in checkpointset:
            np.savez(d/f'checkpoint_{it:05d}.npz',j=j,u=u,w=w,x=x,training_step=it)
            a=j+u@w.T
            states,z,t,seconds=rollout(a,w,x,it*DT)
            lam,lsec=top_lyap(a,states)
            row=dict(seed=seed,n=n,training_step=it,**task(z,t),**recurrence(states),
                     lambda_top=lam,top_seconds=lsec,record_seconds=seconds)
            rows.append(row)
            np.savez_compressed(d/f'rollout_{it:05d}.npz',states=states,z=z,t=t)
            pd.DataFrame(rows).to_csv(d/'reference_pilot.csv',index=False)
            print(row,flush=True)
        if it==max(checkpoints):break
        x=(1-DT)*x+DT*(j@np.tanh(x)+u@(w.T@np.tanh(x)))
        r=np.tanh(x);err=r@w-target((it+1)*DT)
        if (it+1)%2==0:
            pr=p@r;denom=1+r@pr
            k=pr/denom
            w-=np.outer(k,err)
            p-=np.outer(pr,pr)/denom
        if (it+1)%100==0:logs.append(dict(step=it+1,mse=float(err@err/2),weight_norm=float(np.linalg.norm(w))))
    pd.DataFrame(logs).to_csv(d/'training.csv',index=False)
    (d/'metadata.json').write_text(json.dumps(dict(seed=seed,n=n,gain=gain,dt=DT,period=PERIOD,
        training_steps=max(checkpoints),checkpoints=checkpoints,total_seconds=time.perf_counter()-started,
        platform=platform.platform(),numpy=np.__version__,threads=1),indent=2))

def validate():
    j,u,x=init(42,12);w=np.random.default_rng(41).normal(size=(12,2))*.1;a=j+u@w.T
    v=np.random.default_rng(43).normal(size=12);eps=1e-6
    numeric=(step(x+eps*v,a)-step(x-eps*v,a))/(2*eps)
    analytic=(1-DT)*v+DT*(a@((1-np.tanh(x)**2)*v))
    assert np.linalg.norm(numeric-analytic)/np.linalg.norm(analytic)<1e-8
    print('Jacobian finite-difference validation passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=128)
    p.add_argument('--seeds',type=int,nargs='+',default=[0]);p.add_argument('--out',type=Path,default=HERE/'pilot')
    p.add_argument('--gain',type=float,default=1.5)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[0,2000,8000,20000,40000])
    a=p.parse_args()
    with threadpool_limits(limits=1):
        validate()
        for seed in a.seeds:train(seed,a.n,a.out,a.checkpoints,a.gain)
