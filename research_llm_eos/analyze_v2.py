"""Recompute MG, validity checks, matched timestamps, timing and figures."""
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
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
from actdim.estimator.diagnostics import trend_crossings

def scalar_stats(x):
    t=np.arange(len(x)); t=t-t.mean()
    residual=x-x.mean()-t*(t@(x-x.mean()))/(t@t)
    ps=np.abs(np.fft.rfft(residual))[1:]**2
    ps=ps/max(ps.sum(),1e-30)
    entropy=-np.sum(ps*np.log(np.maximum(ps,1e-30)))/np.log(max(len(ps),2))
    acorr=np.corrcoef(residual[:-1],residual[1:])[0,1] if residual.std()>1e-14 else np.nan
    return dict(std=x.std(),residual_std=residual.std(),acorr1=acorr,
                spectral_entropy=entropy,crossings=trend_crossings(x))

def score(x,W,tau):
    cfg=EstimatorConfig(max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,
                        theiler_cap=39*tau,window=W,stride=128,spectral_bins=())
    t=time.perf_counter(); a=estimate(x,cfg); mgsec=time.perf_counter()-t
    t=time.perf_counter(); b=estimate(x,cfg.replace(max_E=40)); extra=time.perf_counter()-t
    t=time.perf_counter(); cheap=scalar_stats(x); cheapsec=time.perf_counter()-t
    return dict(MG=a.MG,MG40=b.MG,ident=b.MG/a.MG if a.MG>0 else np.nan,
                degenerate=a.degenerate or b.degenerate,points=a.n_points,
                mg_seconds=mgsec,validity_seconds=mgsec+extra+cheapsec,
                cheap_seconds=cheapsec,**cheap)

def analyze(root):
    rows=[]; metas=[]; diagnostics=[]
    # Warm import/tree routines outside measured calls.
    score(np.random.default_rng(0).normal(size=512),512,1)
    for path in sorted(root.glob('*/metadata.json')):
        meta=json.loads(path.read_text()); name=path.parent.name
        tr=pd.read_csv(path.parent/'trace.csv')
        dg=pd.read_csv(path.parent/'diagnostics.csv')
        if not len(dg):continue
        dg['run']=name; diagnostics.append(dg)
        metas.append(dict(run=name,**meta))
        for W,tau in [(512,1),(256,1),(1024,1),(512,4)]:
            for signal in ['train_loss','probe_loss','grad_norm']:
                for step in dg['step']:
                    if step+1<W:continue
                    x=tr[signal].to_numpy()[step-W+1:step+1]
                    d=score(x,W,tau)
                    if (W,tau)==(512,1):
                        # Warmed paired repeats. Retain each runtime, use medians in tables.
                        reps=[d]+[score(x,W,tau) for _ in range(2)]
                        for key in ['mg_seconds','validity_seconds','cheap_seconds']:
                            for i,r in enumerate(reps):d[f'{key}_rep{i}']=r[key]
                            d[key]=float(np.median([r[key] for r in reps]))
                    rows.append(dict(run=name,arm=meta['arm'],mode=meta['mode'],lr=meta['lr'],
                                     seed=meta['seed'],step=step,window=W,tau=tau,signal=signal,**d))
        print('Analyzed',name,flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'mg_windows.csv',index=False)
    pd.DataFrame(metas).to_csv(root/'runs.csv',index=False)
    diag=pd.concat(diagnostics,ignore_index=True)
    diag.to_csv(root/'all_diagnostics.csv',index=False)
    primary=frame[(frame.window==512)&(frame.tau==1)].merge(diag,on=['run','step'],suffixes=('','_checkpoint'))
    primary.to_csv(root/'matched.csv',index=False)
    timing=[]; effects=[]; correlations=[]
    for meta in metas:
        name=meta['run']; d=primary[primary.run==name]
        tr=pd.read_csv(root/name/'trace.csv'); n=meta['completed_steps']
        for signal,sub in d.groupby('signal'):
            acquisition=meta['probe_seconds'] if signal=='probe_loss' else meta['norm_seconds'] if signal=='grad_norm' else 0
            # Full log acquisition is required even if only matching a subset of checkpoint windows.
            h=sub.hessian_seconds.sum(); m=sub.mg_seconds.sum(); v=sub.validity_seconds.sum()
            timing.append(dict(run=name,signal=signal,n_checkpoints=len(sub),hessian_seconds=h,
                mg_seconds=m,mg_validity_seconds=v,acquisition_seconds=acquisition,
                mg_end_to_end_seconds=v+acquisition,hessian_over_mg=h/m,
                hessian_over_mg_end_to_end=h/(v+acquisition),activation_seconds=sub.activation_seconds.sum(),
                gradient_proxy_seconds=sub.gradient_seconds.sum(),cheap_seconds=sub.cheap_seconds.sum()))
            pre=sub[(sub.step>=512-1)&(sub.step<n//2)]
            post=sub[sub.step>=n//2+512-1]
            effects.append(dict(run=name,arm=meta['arm'],mode=meta['mode'],seed=meta['seed'],lr=meta['lr'],signal=signal,
                pre_MG=pre.MG.median(),post_MG=post.MG.median(),
                post_pre_ratio=post.MG.median()/pre.MG.median(),
                post_ident=post.ident.median(),degenerate_fraction=sub.degenerate.mean(),
                pre_std=pre['std'].median(),post_std=post['std'].median(),
                pre_hessian=pre.lambda_max.median(),post_hessian=post.lambda_max.median(),
                pre_activation=pre.activation_pr.median(),post_activation=post.activation_pr.median()))
            for metric in ['MG','std','residual_std','acorr1','spectral_entropy']:
                good=sub[[metric,'lambda_max']].dropna()
                rho=spearmanr(good[metric],good.lambda_max).statistic if len(good)>3 else np.nan
                correlations.append(dict(run=name,signal=signal,metric=metric,spearman_hessian=rho))
    pd.DataFrame(timing).to_csv(root/'timing.csv',index=False)
    pd.DataFrame(effects).to_csv(root/'effects.csv',index=False)
    pd.DataFrame(correlations).to_csv(root/'correlations.csv',index=False)
    plots(root,primary,metas)

def plots(root,primary,metas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    for arm in primary.arm.unique():
        runs=[m for m in metas if m['arm']==arm and m['seed']==1]
        if not runs:runs=[m for m in metas if m['arm']==arm and m['seed']==0]
        fig,ax=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
        for meta in runs:
            name=meta['run']; tr=pd.read_csv(root/name/'trace.csv')
            dg=pd.read_csv(root/name/'diagnostics.csv')
            label=f"{meta['mode']}, lr={meta['lr']:g}"
            ax[0].plot(tr.step,tr.train_loss,label=label,alpha=.85,lw=.8)
            signal='train_loss' if arm=='gd' else 'probe_loss'
            mg=primary[(primary.run==name)&(primary.signal==signal)]
            ax[1].plot(mg.step,mg.MG,'o-',label=label,ms=3)
            ax[2].plot(dg.step,dg.gd_ratio if arm=='gd' else dg.lambda_max,'o-',label=label,ms=3)
            ax[3].plot(mg.step,mg.ident,'o-',label=label,ms=3)
            if meta.get('switch'):
                for a in ax:a.axvline(meta['switch'],color='gray',ls=':',alpha=.6)
        ax[0].set_ylabel('Training CE'); ax[0].set_yscale('log')
        ax[1].set_ylabel('MG (E=20)')
        ax[2].set_ylabel('GD stability ratio' if arm=='gd' else 'Hessian top eigenvalue')
        ax[3].set_ylabel('MG(40) / MG(20)'); ax[3].set_xlabel('Optimizer step (window right edge)')
        if arm=='gd':ax[2].axhline(1,color='k',ls='--',lw=.8)
        ax[3].axhline(1,color='k',ls='--',lw=.8)
        ax[0].legend(ncol=2,fontsize=8)
        fig.suptitle(f'{arm.upper()}: matched trajectories, seed {runs[0]["seed"]}')
        fig.savefig(root/f'{arm}_same_seed.png',dpi=170)
        fig.savefig(root/f'{arm}_same_seed.pdf')
        plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=HERE/'v2')
    a=parser.parse_args()
    with threadpool_limits(limits=4):analyze(a.root)
