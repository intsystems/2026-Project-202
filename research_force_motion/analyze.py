"""Independent full-state diagnostics and frozen scalar MG on held-out runs."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'code'))
from actdim.estimator.mle import estimate
from actdim.estimator.config import EstimatorConfig
from run import DT,PERIOD,step,rollout,task,top_lyap

def spectrum(a,states,qr_every=10,burn=1000):
    n=len(a)
    q,_=np.linalg.qr(np.random.default_rng(832).normal(size=(n,n)))
    sums=np.zeros(n);half=np.zeros(n);count=0;half_count=0
    started=time.perf_counter()
    for i,x in enumerate(states):
        d=1-np.tanh(x)**2
        q=(1-DT)*q+DT*a@(d[:,None]*q)
        if (i+1)%qr_every==0:
            q,r=np.linalg.qr(q)
            if i+1>burn:
                sums+=np.log(np.maximum(np.abs(np.diag(r)),1e-300));count+=qr_every
                if i+1<=len(states)//2:
                    half=sums.copy();half_count=count
    values=np.sort(sums/(count*DT))[::-1]
    first_half=np.sort(half/(half_count*DT))[::-1]
    return values,first_half,time.perf_counter()-started

def return_residual(states,period=PERIOD/DT):
    # Fractional shift interpolated from full state, no period fitted using MG.
    x=np.asarray(states,dtype=float)
    if x.ndim==1:x=x[:,None]
    maxlag=int(np.ceil(period*1.2))+1
    origin=x[:-maxlag]
    denom=float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    def objective(lag):
        j=int(lag);f=lag-j
        shifted=(1-f)*x[j:j+len(origin)]+f*x[j+1:j+1+len(origin)]
        return float(np.mean(np.sum((shifted-origin)**2,axis=1))/max(denom,1e-30))
    fit=minimize_scalar(objective,bounds=(.8*period,1.2*period),method='bounded',options={'xatol':.01})
    return float(np.sqrt(fit.fun)),float(fit.x),denom

def stats(x):
    y=x-x.mean();power=np.abs(np.fft.rfft(y))[1:]**2;power/=max(power.sum(),1e-300)
    entropy=float(-np.sum(power*np.log(np.maximum(power,1e-300)))/np.log(len(power)))
    residual,period,variance=return_residual(x)
    return dict(std=float(x.std()),entropy=entropy,scalar_return=residual,scalar_period=period)

def mg(x,w=4096,tau=4):
    cfg=EstimatorConfig(max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,
                        theiler_cap=39*tau,window=w,spectral_bins=())
    t=time.perf_counter();a=estimate(x,cfg,seed=123);ta=time.perf_counter()
    b=estimate(x,cfg.replace(max_E=40),seed=123);tb=time.perf_counter()
    s=stats(x);tc=time.perf_counter()
    return dict(MG=a.MG,MG40=b.MG,ident=b.MG/a.MG if a.MG>0 else np.nan,
                degenerate=a.degenerate or b.degenerate,mg_seconds=ta-t,
                checks_seconds=tb-t,cheap_seconds=tc-tb,**s)

def reference(root):
    rows=[]
    for d in sorted(root.glob('seed_*')):
        if not (d/'metadata.json').exists():continue
        seed=int(d.name.split('_')[-1])
        pilot=pd.read_csv(d/'reference_pilot.csv').set_index('training_step')
        for training in [0,8000,40000]:
            r=np.load(d/f'rollout_{training:05d}.npz')
            ck=np.load(d/f'checkpoint_{training:05d}.npz');a=ck['j']+ck['u']@ck['w'].T
            cache=d/f'lyapunov_{training:05d}.npz'
            if cache.exists():
                c=np.load(cache);ls,half,sec=c['spectrum'],c['half'],float(c['seconds'])
            else:
                ls,half,sec=spectrum(a,r['states'])
                np.savez(cache,spectrum=ls,half=half,seconds=sec)
            t=time.perf_counter();ret,period,var=return_residual(r['states']);rsec=time.perf_counter()-t
            row=dict(seed=seed,training_step=training,lambda1=ls[0],lambda2=ls[1],lambda3=ls[2],
                     positive_005=int(np.sum(ls>.005)),positive_0001=int(np.sum(ls>.0001)),
                     neutral_005=int(np.sum(np.abs(ls)<=.005)),
                     lambda1_half=half[0],lambda2_half=half[1],
                     full_seconds=sec,return_error=ret,period_steps=period,state_variance=var,
                     return_seconds=rsec,aligned_nrmse=pilot.loc[training,'aligned_nrmse'],
                     raw_nrmse=pilot.loc[training,'raw_nrmse'],top_seconds=pilot.loc[training,'top_seconds'],
                     top_single=pilot.loc[training,'lambda_top'],record_seconds=pilot.loc[training,'record_seconds'])
            row['endpoint_pass']=bool(row['aligned_nrmse']<.1 and ret<.1 and abs(ls[0])<.005 and ls[1]<-.005)
            rows.append(row)
            pd.DataFrame(rows).to_csv(root/'independent_reference.csv',index=False)
            print('Spectrum',seed,training,'top',ls[:3],'return',ret,'seconds',sec,flush=True)
    return pd.DataFrame(rows)

def scalar_analysis(root):
    # Warm the estimator outside recorded benchmark calls.
    mg(np.random.default_rng(921).normal(size=4096))
    rows=[];sensitivity=[]
    for d in sorted(root.glob('seed_*')):
        if not (d/'metadata.json').exists():continue
        seed=int(d.name.split('_')[-1])
        for training in [0,2000,8000,20000,40000]:
            r=np.load(d/f'rollout_{training:05d}.npz')
            for neuron in [0,1,2]:
                signal=np.tanh(r['states'][:,neuron])
                for start in [0,4096]:
                    x=signal[start:start+4096]
                    v=mg(x)
                    rows.append(dict(seed=seed,training_step=training,neuron=neuron,start=start,**v))
            if training in [0,40000]:
                for w,tau in [(2048,4),(8192,4),(4096,2),(4096,8)]:
                    x=np.tanh(r['states'][-w:,0]);v=mg(x,w,tau)
                    sensitivity.append(dict(seed=seed,training_step=training,window=w,tau=tau,**v))
            pd.DataFrame(rows).to_csv(root/'mg.csv',index=False)
            pd.DataFrame(sensitivity).to_csv(root/'sensitivity.csv',index=False)
            print('MG',seed,training,flush=True)
    df=pd.DataFrame(rows)
    df.groupby(['seed','training_step','neuron'],as_index=False).median(numeric_only=True).to_csv(root/'mg_summary.csv',index=False)

def controls(root):
    rows=[];amp=[]
    for d in sorted(root.glob('seed_*')):
        if not (d/'metadata.json').exists():continue
        seed=int(d.name.split('_')[-1])
        ck=np.load(d/'checkpoint_40000.npz');initial=np.load(d/'checkpoint_00000.npz')
        a=ck['j']+ck['u']@ck['w'].T
        # Robustness to moderate perturbations of the training terminal state.
        for pseed in [11,12,13]:
            rng=np.random.default_rng(pseed)
            x=ck['x']+.5*rng.normal(size=len(a))
            states,z,t,sec=rollout(a,ck['w'],x,40000*DT,length=4096,burn=2000)
            error=task(z,t);ret,period,var=return_residual(states)
            v=mg(np.tanh(states[:,0]))
            rows.append(dict(seed=seed,perturbation=pseed,**error,return_error=ret,
                             period_steps=period,MG=v['MG'],ident=v['ident']))
        before=np.load(d/'rollout_00000.npz')['states'][:4096]
        x=np.tanh(before[:,0]);v=mg(x);scaled=mg(.1*x)
        amp.append(dict(seed=seed,MG=v['MG'],MG_scaled=scaled['MG'],delta=scaled['MG']-v['MG']))
        assert np.isclose(v['MG'],scaled['MG'],rtol=1e-9,atol=1e-9)
        print('Perturbations',seed,flush=True)
    pd.DataFrame(rows).to_csv(root/'perturbations.csv',index=False)
    pd.DataFrame(amp).to_csv(root/'scale_control.csv',index=False)
    # A genuinely trained readout with disconnected feedback must leave the hidden
    # autonomous field, and therefore any hidden-neuron observation, unchanged.
    control_dir=sorted(root.glob('seed_*'),key=lambda d:int(d.name.split('_')[-1]))[0]
    initial=np.load(control_dir/'checkpoint_00000.npz')
    j=initial['j'];x=initial['x'].copy();n=len(x);w=np.zeros((n,2));p=np.eye(n)
    from run import target
    x_control=x.copy()
    for it in range(8000):
        x=step(x,j);x_control=step(x_control,j)
        r=np.tanh(x)
        if (it+1)%2==0:
            pr=p@r;denom=1+r@pr;err=r@w-target((it+1)*DT)
            w-=np.outer(pr/denom,err);p-=np.outer(pr,pr)/denom
    assert np.array_equal(x,x_control)
    states,z,t,_=rollout(j,w,initial['x'],0,length=4096,burn=2000)
    ref=np.load(control_dir/'rollout_00000.npz')['states'][:4096]
    assert np.array_equal(states,ref)
    (root/'control_audit.json').write_text(json.dumps(dict(disconnected_readout_trained=True,
        learned_readout_norm=float(np.linalg.norm(w)),identical_hidden_trajectory=True,
        maximum_difference=float(np.max(np.abs(states-ref))),scale_invariance=True),indent=2))

def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref=pd.read_csv(root/'independent_reference.csv');mgdf=pd.read_csv(root/'mg_summary.csv')
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for seed,r in ref.groupby('seed'):
        color=f'C{(seed-1)%10}'
        axs[0].plot(r.training_step,r.aligned_nrmse,'o-',color=color,label=f'seed {seed}')
        axs[1].plot(r.training_step,r.return_error,'o-',color=color)
        d=mgdf[(mgdf.seed==seed)&(mgdf.neuron==0)]
        axs[2].plot(d.training_step,d.MG,'o-',color=color)
    axs[0].set_ylabel('Task NRMSE (phase aligned)');axs[0].set_yscale('log');axs[0].legend(ncol=5)
    axs[1].set_ylabel('Full-state return error');axs[1].set_yscale('log')
    axs[2].set_ylabel('MG, fixed neuron 0');axs[2].set_xlabel('Training steps (weights frozen for each rollout)')
    fig.savefig(root/'learning.png',dpi=180);fig.savefig(root/'learning.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    selected=3 if (root/'seed_3').exists() else int(ref.seed.min())
    d=root/f'seed_{selected}'
    for training,label in [(0,'Before'),(8000,'8,000 steps'),(40000,'40,000 steps')]:
        l=np.load(d/f'lyapunov_{training:05d}.npz')['spectrum']
        axs[0,0].plot(np.arange(1,len(l)+1),l,label=label)
        axs[0,1].plot(np.arange(1,21),l[:20],'.-',label=label)
        s=np.load(d/f'rollout_{training:05d}.npz')
        axs[1,0].plot(np.arange(1000)*DT,np.tanh(s['states'][:1000,0]),lw=.8,label=label)
    axs[0,0].legend(fontsize=8);axs[0,0].set_xlabel('Exponent index');axs[0,0].set_ylabel('Full Lyapunov spectrum')
    axs[0,1].axhline(0,color='k',lw=.7);axs[0,1].set_ylabel('Top 20 exponents');axs[0,1].set_xlabel('Exponent index')
    axs[1,0].set_xlabel('Autonomous time');axs[1,0].set_ylabel('Observed neuron')
    for neuron in [0,1,2]:
        m=mgdf[(mgdf.seed==selected)&(mgdf.neuron==neuron)]
        axs[1,1].plot(m.training_step,m.MG,'o-',label=f'neuron {neuron}')
    axs[1,1].legend();axs[1,1].set_ylabel('MG');axs[1,1].set_xlabel('Training steps')
    fig.suptitle(f'Seed {selected}: same autonomous trajectories for MG and full-state reference')
    fig.savefig(root/'same_seed.png',dpi=180);fig.savefig(root/'same_seed.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE/'results')
    p.add_argument('--phase',choices=['reference','mg','controls','plots','all'],default='all');a=p.parse_args()
    with threadpool_limits(limits=1):
        if a.phase in ['reference','all']:reference(a.root)
        if a.phase in ['mg','all']:scalar_analysis(a.root)
        if a.phase in ['controls','all']:controls(a.root)
        if a.phase in ['plots','all']:plots(a.root)
