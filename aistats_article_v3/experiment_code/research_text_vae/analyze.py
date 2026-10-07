from pathlib import Path
import argparse,json,sys,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'code'))
from actdim.estimator.mle import estimate
from actdim.estimator.config import EstimatorConfig

def measure(x,w,tau=1):
    cfg=EstimatorConfig(window=w,max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,theiler_cap=39*tau)
    start=time.perf_counter();a=estimate(x,cfg,seed=123);b=estimate(x,cfg.replace(max_E=40),seed=123)
    y=x-x.mean();ps=np.abs(np.fft.rfft(y))[1:]**2;ps/=max(ps.sum(),1e-30)
    return dict(MG=a.MG,MG40=b.MG,ident=b.MG/a.MG,degenerate=a.degenerate or b.degenerate,
        std=float(x.std()),entropy=float(-np.sum(ps*np.log(np.maximum(ps,1e-30)))/np.log(len(ps))),seconds=time.perf_counter()-start)

def change(df,key,timecol='end',lower=512):
    before=df[df[timecol].between(lower,1024)][key].median()
    after=df[df[timecol].between(2048,3072)][key].median()
    return dict(before=float(before),after=float(after),ratio=float(after/before))

def analyze(root):
    rows=[];summaries={};refs={};logs={}
    for arm in ['base','regularized']:
        d=root/arm;log=pd.read_csv(d/'logs.csv');ref=pd.read_csv(d/'reference.csv')
        assert len(log)==3072 and int(ref.step.max())==3072,'Incomplete run'
        refs[arm]=ref;logs[arm]=log;x=log.probe_nll.to_numpy()
        for w,tau in [(512,1),(256,1),(1024,1),(512,4)]:
            for end in range(w,len(x)+1,128):
                rows.append(dict(arm=arm,window=w,tau=tau,end=end,**measure(x[end-w:end],w,tau)))
        summaries[arm]={name:change(ref,name,'step') for name in ['KL','MI','shuffle_symkl','shuffle_nll_gap','nll']}
    df=pd.DataFrame(rows);df.to_csv(root/'windows.csv',index=False);sens=[]
    for (w,tau),g in df.groupby(['window','tau']):
        # W1024 has just one pre-intervention window; disclose shorter pre evidence.
        armstats={arm:{k:change(g[g.arm==arm],k,lower=max(512,w)) for k in ['MG','std','entropy']} for arm in summaries}
        sens.append(dict(window=int(w),tau=int(tau),arms=armstats,
            paired={k:armstats['regularized'][k]['ratio']/armstats['base'][k]['ratio'] for k in ['MG','std','entropy']}))
    paired={k:summaries['regularized'][k]['ratio']/summaries['base'][k]['ratio'] for k in summaries['base']}
    event=bool(paired['MI']<.5 and paired['shuffle_symkl']<.5 and summaries['base']['MI']['before']>.1 and summaries['base']['shuffle_symkl']['before']>1e-4)
    result=dict(reference=summaries,reference_paired=paired,independent_event=event,scalar=sens,
        settings='Fixed pilot; no choice by MG outcome. W1024 has only one pre window.')
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(10,6),constrained_layout=True);cols={'base':'#777777','regularized':'#0868ac'}
    for arm in summaries:
        ref=refs[arm];log=logs[arm];c=cols[arm];g=df[(df.arm==arm)&(df.window==512)&(df.tau==1)]
        axs[0,0].plot(ref.step,ref.MI,label=arm,color=c)
        axs[0,1].plot(ref.step,ref.shuffle_symkl,label=arm,color=c)
        axs[1,0].plot(log.step,log.probe_nll,label=arm,color=c,lw=.6)
        axs[1,1].plot(g.end,g.MG,label=arm,color=c)
    for ax in axs.flat:
        ax.axvline(1024,color='#b35806',ls='--',lw=1);ax.set_xlabel('Optimizer step');ax.legend(fontsize=8)
    axs[0,0].set_title('Independent: empirical sentence-code MI (nats)')
    axs[0,1].set_title('Independent: prediction response to code shuffling')
    axs[1,0].set_title('MG input: fixed-probe reconstruction NLL/token')
    axs[1,1].set_title('Primary MG, W=512 (window end)')
    fig.savefig(root/'overview.pdf');fig.savefig(root/'overview.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=H/'pilot_seed0');a=p.parse_args()
    measure(np.sin(np.arange(512)/11)+np.cos(np.arange(512)/7),512)
    with threadpool_limits(limits=1):analyze(a.root)
