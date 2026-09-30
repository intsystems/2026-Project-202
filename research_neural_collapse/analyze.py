"""Frozen MG configuration, independent NC audit, scalar baselines and timing."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'code'))
from actdim.estimator.mle import estimate
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.diagnostics import trend_crossings
from run import geometry


def scalar_stats(x):
    t=np.arange(len(x),dtype=float);t-=t.mean()
    centered=x-x.mean()
    slope=float(t@centered/(t@t))
    residual=centered-slope*t
    power=np.abs(np.fft.rfft(residual))[1:]**2
    power=power/max(power.sum(),1e-30)
    entropy=float(-np.sum(power*np.log(np.maximum(power,1e-30)))/np.log(max(len(power),2)))
    acf=float(np.corrcoef(residual[:-1],residual[1:])[0,1]) if residual.std()>1e-30 else np.nan
    return dict(mean=float(x.mean()),slope=slope,std=float(x.std()),
                detrended_std=float(residual.std()),acf1=acf,spectral_entropy=entropy,
                crossings=trend_crossings(x))


def score(x,w,tau):
    cfg=EstimatorConfig(max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,
                        theiler_cap=39*tau,window=w,stride=128,spectral_bins=())
    reps=[];vals=None
    repeats=3 if (w,tau)==(512,1) else 1
    for _ in range(repeats):
        t=time.perf_counter(); a=estimate(x,cfg,seed=123);t1=time.perf_counter()
        b=estimate(x,cfg.replace(max_E=40),seed=123);t2=time.perf_counter()
        stats=scalar_stats(x);t3=time.perf_counter()
        reps.append(dict(mg_seconds=t1-t,checks_seconds=t3-t,cheap_seconds=t3-t2))
        vals=dict(MG=a.MG,MG40=b.MG,ident=b.MG/a.MG if a.MG>0 else np.nan,
                  degenerate=a.degenerate or b.degenerate,**stats)
    vals.update({k:float(np.median([r[k] for r in reps])) for k in reps[0]})
    for i,r in enumerate(reps):vals.update({f'{k}_rep{i}':v for k,v in r.items()})
    return vals


def audit(root):
    # Exact simplex with zero within-class scatter is the zero of all NC1/NC2/NC3.
    eye=np.eye(10);m=eye-eye.mean(0);h=np.repeat(m,5,axis=0);y=np.repeat(np.arange(10),5)
    a,_=geometry(h,y,m)
    assert abs(a['nc1'])<1e-12 and abs(a['nc2'])<1e-12 and abs(a['nc3'])<1e-12,a
    rng=np.random.default_rng(23)
    noisy=h+.1*rng.normal(size=h.shape)
    b,_=geometry(noisy,y,m)
    q,_=np.linalg.qr(rng.normal(size=(10,10)))
    c,_=geometry(7*noisy@q,y,7*m@q)
    assert all(abs(b[k]-c[k])<1e-8 for k in ['nc1','nc1_trace','nc2','nc3']), (b,c)
    rec=[]
    for meta_path in root.glob('*/metadata.json'):
        d=meta_path.parent
        initial=np.load(d/'features/step_00000.npz')
        last=np.load(sorted((d/'features').glob('*.npz'))[-1])
        ref=pd.read_csv(d/'reference.csv');trace=pd.read_csv(d/'trace.csv')
        computed,_=geometry(last['h'],last['labels'],last['weights'])
        for key in ['nc1','nc1_trace','nc2','nc3']:
            assert np.isclose(computed[key],ref.iloc[-1][key],rtol=1e-8,atol=1e-9)
        change=float(np.max(np.abs(initial['h']-last['h'])))
        if 'frozen' in d.name:assert change==0,change
        assert len(trace)==json.loads(meta_path.read_text())['steps']
        rec.append(dict(run=d.name,rows=len(trace),full_state_max_change=change,
                        geometry_recomputed=True))
    result=dict(simplex_zero_test=a,scale_rotation_invariance=True,runs=rec)
    (root/'audit.json').write_text(json.dumps(result,indent=2))
    return result


def corr(x,y):
    good=np.isfinite(x)&np.isfinite(y)
    x=np.asarray(x)[good];y=np.asarray(y)[good]
    if len(x)<4 or np.std(x)<1e-14 or np.std(y)<1e-14:return np.nan
    return float(spearmanr(x,y).statistic)


def analyze(root):
    audit(root)
    score(np.random.default_rng(0).normal(size=512),512,1)
    allrows=[];refs=[];runs=[];small_refs=[]
    for p in sorted(root.glob('*/metadata.json')):
        name=p.parent.name;meta=json.loads(p.read_text());runs.append(dict(run=name,**meta))
        tr=pd.read_csv(p.parent/'trace.csv')
        ref=pd.read_csv(p.parent/'reference.csv');ref['run']=name;refs.append(ref)
        small=pd.read_csv(p.parent/'small_probe_reference.csv');small['run']=name;small_refs.append(small)
        for w,tau in [(512,1),(256,1),(1024,1),(512,4)]:
            for signal in ['train_loss','probe_loss','test_probe_loss']:
                values=tr[signal].to_numpy()
                for step in ref.step:
                    if step<w:continue
                    x=values[step-w:step]
                    allrows.append(dict(run=name,arm=meta['arm'],seed=meta['seed'],signal=signal,
                                        window=w,tau=tau,step=step,**score(x,w,tau)))
        print('Analyzed',name,flush=True)
    mg=pd.DataFrame(allrows);reference=pd.concat(refs,ignore_index=True)
    mg.to_csv(root/'mg_windows.csv',index=False)
    reference.to_csv(root/'all_reference.csv',index=False)
    pd.DataFrame(runs).to_csv(root/'runs.csv',index=False)
    sm=pd.concat(small_refs,ignore_index=True);sm.to_csv(root/'all_small_probe.csv',index=False)
    primary=mg[(mg.window==512)&(mg.tau==1)].merge(reference,on=['run','step'])
    primary.to_csv(root/'matched.csv',index=False)
    effects=[];timings=[];correlations=[];terminals=[];sensitivity=[]
    for meta in runs:
        name=meta['run'];ref=reference[reference.run==name]
        zero=ref[ref.train_accuracy>=1-1e-8]
        zero_step=int(zero.step.min()) if len(zero) else None
        first=ref.iloc[0];last=ref.iloc[-1]
        terminal=ref[ref.step>=zero_step] if zero_step else ref.iloc[:0]
        zrow=zero.iloc[0] if len(zero) else None
        terminals.append(dict(run=name,arm=meta['arm'],seed=meta['seed'],zero_step=zero_step,
            final_train_accuracy=last.train_accuracy,final_test_accuracy=last.test_accuracy,
            initial_nc1=first.nc1,final_nc1=last.nc1,
            zero_nc1=zrow.nc1 if zrow is not None else np.nan,
            zero_nc2=zrow.nc2 if zrow is not None else np.nan,
            final_nc2=last.nc2,final_nc3=last.nc3,final_nc4=last.nc4,
            initial_nc2=first.nc2,initial_nc3=first.nc3,initial_nc4=first.nc4,
            terminal_nc1_ratio=last.nc1/zrow.nc1 if zrow is not None else np.nan))
        for signal,d in primary[primary.run==name].groupby('signal'):
            pre=d[(d.step>=512)&(d.step<=1024)]
            post=d[d.step>meta['steps']-1024]
            eff=dict(run=name,arm=meta['arm'],seed=meta['seed'],signal=signal,
                     mg_early=pre.MG.median(),mg_late=post.MG.median(),mg_ratio=post.MG.median()/pre.MG.median(),
                     nc1_early=pre.nc1.median(),nc1_late=post.nc1.median(),
                     nc2_early=pre.nc2.median(),nc2_late=post.nc2.median(),
                     ident_late=post.ident.median(),degenerate_fraction=d.degenerate.mean())
            effects.append(eff)
            # Reconstruction cost for every optimizer step, even if windows are sparse.
            acq=0. if signal=='train_loss' else meta['steps']*meta['one_probe_forward_seconds']
            m=d.mg_seconds.sum();v=d.checks_seconds.sum();n=d.nc_total_seconds.sum()
            timings.append(dict(run=name,signal=signal,checkpoints=len(d),mg_seconds=m,
                mg_checks_seconds=v,probe_acquisition_seconds=acq,mg_end_to_end_seconds=v+acq,
                reference_seconds=n,reference_over_mg=n/m,reference_over_checks=n/v,
                reference_over_end_to_end=n/(v+acq),cheap_seconds=d.cheap_seconds.sum(),
                mg_per_window_ms=1000*d.mg_seconds.median(),
                reference_per_checkpoint_ms=1000*d.nc_total_seconds.median()))
            subsets=[('all',d)]
            if zero_step:subsets.append(('terminal_full_windows',d[d.step-512+1>=zero_step]))
            for phase,sub in subsets:
                for metric in ['MG','mean','slope','std','detrended_std','acf1','spectral_entropy','crossings']:
                    correlations.append(dict(run=name,arm=meta['arm'],signal=signal,phase=phase,metric=metric,
                        checkpoints=len(sub),rho_nc1=corr(sub[metric],sub.nc1),rho_nc2=corr(sub[metric],sub.nc2),
                        rho_differences_nc1=corr(np.diff(sub[metric]),np.diff(sub.nc1))))
        for (signal,w,tau),d in mg[mg.run==name].groupby(['signal','window','tau']):
            # Use same step intervals, but W1024 has only one early checkpoint.
            pre=d[(d.step>=512)&(d.step<=1024)];post=d[d.step>meta['steps']-1024]
            sensitivity.append(dict(run=name,arm=meta['arm'],signal=signal,window=w,tau=tau,
                pre_windows=len(pre),mg_early=pre.MG.median(),mg_late=post.MG.median(),
                mg_ratio=post.MG.median()/pre.MG.median(),ident_late=post.ident.median()))
    for filename,rows in [('effects',effects),('timing',timings),('correlations',correlations),
                          ('terminal',terminals),('sensitivity',sensitivity)]:
        pd.DataFrame(rows).to_csv(root/f'{filename}.csv',index=False)
    small_quality=[]
    for name,d in sm.groupby('run'):
        merged=d.merge(reference[reference.run==name],on=['run','step'],suffixes=('_small','_full'))
        for metric in ['nc1','nc1_trace','nc2','nc3']:
            small_quality.append(dict(run=name,metric=metric,
                rho=corr(merged[metric+'_small'],merged[metric+'_full']),
                median_small=merged[metric+'_small'].median(),median_full=merged[metric+'_full'].median()))
    pd.DataFrame(small_quality).to_csv(root/'small_probe_quality.csv',index=False)
    plots(root,primary,reference)
    print(pd.DataFrame(effects).to_string(index=False),flush=True)


def plots(root,matched,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    fig,axs=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
    colors=['#1976a3','#d17b18','#5a8e40']
    for seed,color in enumerate(colors):
        name=f'train_s{seed}';r=reference[reference.run==name]
        if r.empty:continue
        axs[0].plot(r.step,100*r.train_accuracy,color=color,label=f'train, seed {seed}')
        axs[0].plot(r.step,100*r.test_accuracy,color=color,ls='--',alpha=.6)
        axs[1].plot(r.step,r.nc1,color=color)
        axs[2].plot(r.step,r.nc2,color=color)
        for signal,style in [('probe_loss','-'),('train_loss','--')]:
            m=matched[(matched.run==name)&(matched.signal==signal)]
            axs[3].plot(m.step,m.MG,color=color,ls=style,label=f'{signal}, seed {seed}')
    axs[0].set_ylabel('Accuracy, %');axs[0].legend(ncol=3,fontsize=8)
    axs[0].set_title('CIFAR-10 subset: solid train accuracy; dashed test accuracy')
    axs[1].set_ylabel('NC1 (lower is simpler)');axs[1].set_yscale('log')
    axs[2].set_ylabel('NC2 ETF distance')
    axs[3].set_ylabel('MG');axs[3].set_xlabel('Optimizer step');axs[3].legend(ncol=2,fontsize=8)
    fig.savefig(root/'training_and_mg.png',dpi=170);fig.savefig(root/'training_and_mg.pdf');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
    for name,style in [('train_s0','-'),('frozen_s0','--')]:
        r=reference[reference.run==name]
        if r.empty:continue
        axs[0].plot(r.step,r.nc1,style,label=name)
        axs[1].plot(r.step,r.nc2,style,label=name)
        m=matched[(matched.run==name)&(matched.signal=='probe_loss')]
        axs[2].plot(m.step,m.MG,style,label=name)
    axs[0].set_yscale('log');axs[0].set_ylabel('NC1');axs[0].legend()
    axs[1].set_ylabel('NC2');axs[2].set_ylabel('MG: fixed train probe');axs[2].set_xlabel('Optimizer step')
    fig.savefig(root/'frozen_control.png',dpi=170);fig.savefig(root/'frozen_control.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE/'results');a=p.parse_args()
    with threadpool_limits(limits=1):analyze(a.root)
