"""Generate paired summaries and figures from completed, unfiltered runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;R=H/'confirmation'

def ratios(df, columns, groups):
    early=df[df.end.between(1024,2048)].groupby(groups)[columns].median()
    late=df[df.end.between(3072,4096)].groupby(groups)[columns].median()
    return late/early

if __name__=='__main__':
    p=pd.read_csv(R/'probe_windows.csv');p=p[(p.signal=='test_probe')&(p.preprocess=='raw')]
    refs=[]
    for f in sorted(R.glob('seed_*/*/reference.csv')):
        d=pd.read_csv(f);d['run']=f.parent.relative_to(R).as_posix();refs.append(d)
    ref=pd.concat(refs,ignore_index=True)
    a=ratios(ref,['pr','small_pr','variance','raw_pr','update_pr'],['run'])
    b=ratios(p,['MG','std','entropy','crossings'],['run'])
    a=a.join(b);a.to_csv(R/'summary_ratios.csv')
    pairs=[]
    for seed in range(1,6):
        base=a.loc[f'seed_{seed}/base'];drop=a.loc[f'seed_{seed}/drop']
        row={'seed':seed}
        for name in a.columns:
            row[name+'_base']=float(base[name]);row[name+'_drop']=float(drop[name]);row[name+'_paired']=float(drop[name]/base[name])
        pairs.append(row)
    paired=pd.DataFrame(pairs);paired.to_csv(R/'paired_summary.csv',index=False)
    sensitivity=pd.read_csv(R/'probe_sensitivity.csv')
    sx=ratios(sensitivity,['MG'],['window','tau','run']).reset_index();summaries=[]
    for (w,tau),g in sx.groupby(['window','tau']):
        vals=[]
        for s in range(1,6):
            q=g.set_index('run').MG;vals.append(q[f'seed_{s}/drop']/q[f'seed_{s}/base'])
        summaries.append(dict(window=int(w),tau=int(tau),paired=vals,signs=int(np.sum(np.array(vals)<1)),median=float(np.median(vals))))
    bench=pd.read_csv(R/'benchmark.csv').median(numeric_only=True)
    # Whole 4096-step monitoring path, 15 overlapping windows: acquire each sample once.
    costs={
      'full_PR':15*bench.full_seconds+8*bench.full_record512_seconds,
      'small128_PR':15*bench.small_seconds+8*bench.small_record512_seconds,
      'MG_only':15*bench.mg_seconds+4096*bench.probe_forward_seconds,
      'MG_with_E40_check':15*bench.checks_seconds+4096*bench.probe_forward_seconds,
      'cheap_scalar_metrics':15*bench.cheap_seconds+4096*bench.probe_forward_seconds,
    }
    c=pd.read_csv(R/'probe_controls.csv');cr=c.groupby('end').surrogate_ratio.agg(['min','median','max'])
    acc=pd.read_csv(R/'unseen_accuracy.csv')
    result=dict(paired_median={k:float(paired[k].median()) for k in paired if k.endswith('_paired')},
        paired_signs={k:int((paired[k]<1).sum()) for k in paired if k.endswith('_paired')},
        sensitivity=summaries,timing_medians=bench.to_dict(),monitoring4096_seconds=costs,
        accuracy_min=float(acc.unseen_test_accuracy.min()),accuracy_max=float(acc.unseen_test_accuracy.max()),
        primary_degenerate_count=int(p.degenerate.sum()),ident_range=[float(p.ident.min()),float(p.ident.max())],
        surrogate_comparison=cr.to_dict())
    (R/'summary.json').write_text(json.dumps(result,indent=2))
    print(paired[['seed','pr_base','pr_drop','MG_base','MG_drop','MG_paired']].to_string(index=False))
    print(json.dumps(result,indent=2))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'base':'#777777','drop':'#0868ac'}
    fig,axs=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for arm in ['base','drop']:
        tag=f'seed_1/{arm}';g=ref[ref.run==tag];m=p[p.run==tag]
        axs[0,0].plot(g.end,g.pr,label=arm,color=colors[arm])
        axs[0,1].plot(m.end,m.MG,label=arm,color=colors[arm])
        e=np.load(R/tag/'spectrum_4096.npy');axs[1,0].plot(np.arange(1,21),e[:20]/e.sum(),label=f'{arm}, late',color=colors[arm])
    e=np.load(R/'seed_1/base/spectrum_2048.npy');axs[1,0].plot(np.arange(1,21),e[:20]/e.sum(),'--',label='shared, before',color='#b35806')
    for ax in axs[0]:
        ax.axvline(2048,ls='--',color='#b35806',lw=1);ax.set_xlabel('Training step (window end)');ax.legend()
    axs[0,0].set_title('Seed 1: full-weight covariance PR');axs[0,1].set_title('Seed 1: MG of fixed-probe loss')
    axs[1,0].set(title='Seed 1: normalized covariance spectrum',xlabel='Eigenvalue index',ylabel='Fraction of total variance',yscale='log');axs[1,0].legend(fontsize=8)
    for k,offset,label,col in [('pr_paired',-.12,'Full-weight PR','#238b45'),('MG_paired',.12,'MG','#0868ac')]:
        axs[1,1].scatter(paired.seed+offset,paired[k],label=label,color=col)
    axs[1,1].axhline(1,color='gray',ls='--');axs[1,1].set(title='All fresh seeds: drop/base change ratios',xlabel='Seed',ylabel='Paired ratio (<1: stronger decrease)',xticks=range(1,6));axs[1,1].legend(fontsize=8)
    fig.savefig(H/'results.png',dpi=180);fig.savefig(H/'results.pdf');plt.close(fig)
