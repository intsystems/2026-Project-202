from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;OLD=H.parent/'research_walker_repair'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})

def old_results():
    mg=pd.read_csv(H/'diagnostic/MG_summary.csv');period=pd.read_csv(H/'diagnostic/periods.csv');rows=[]
    probes=pd.DataFrame(json.loads((H/'diagnostic_probes/summary.json').read_text()))
    for seed in range(211,216):
        row=dict(seed=seed)
        # Inspect original schema explicitly rather than guessing field names.
        for mode in ['fixed','cycles','resampled']:
            a=mg[(mg.seed==seed)&(mg.step==0)&(mg['mode']==mode)].set_index('reset')
            b=mg[(mg.seed==seed)&(mg.step>0)&(mg['mode']==mode)].set_index('reset')
            ratio=(b.MG/a.MG)[b.valid&a.valid].dropna();row[mode]=float(ratio.median());row[mode+'_n']=len(ratio)
            pa=period[(period.seed==seed)&(period.step==0)].set_index('reset')
            pb=period[(period.seed==seed)&(period.step>0)].set_index('reset')
            stable=(pa.stable&pb.stable).reindex(ratio.index,fill_value=False)
            row[mode+'_stable_n']=int(stable.sum());row[mode+'_stable_ratio']=float(ratio[stable].median()) if stable.any() else None
        p=probes[probes.seed==seed].set_index('step')
        row['amplification_ratio']=float(p.loc[1048576,'median_amplification']/p.loc[0,'median_amplification'])
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(H/'diagnostic_summary.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(10,3.1))
    for k,title in [('fixed','MG: фиксированное окно'),('cycles','MG: 14 циклов'),('resampled','MG: 14 циклов, интерполяция'),('amplification_ratio','Возмущения: одинаковые 4,8 с')]:ax[0].plot(frame.seed,frame[k],'o-',label=title)
    ax[0].axhline(1,color='gray',ls='--');ax[0].set(xlabel='Сид дообучения',ylabel='Финал / исходная политика',xticks=frame.seed);ax[0].legend(fontsize=7)
    for step,title in [(0,'Исходная политика'),(1048576,'Финал, сид211')]:
        data=np.load(OLD/'seed211'/f'step{step:07d}'/'reset51001'/'trajectory.npz')
        x=np.concatenate([data['qpos'][:,1:],data['qvel']],axis=1)/np.r_[np.ones(8),np.full(9,5.)]
        var=np.mean(np.sum((x-x.mean(0))**2,axis=1));lag=np.arange(20,251)
        err=[np.mean(np.sum((x[p:]-x[:-p])**2,axis=1))/(2*var) for p in lag]
        ax[1].plot(lag,err,label=title)
    ax[1].set(xlabel='Лаг (шаги)',ylabel='Ошибка возврата полного состояния');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(H/'diagnostic_figure.pdf');fig.savefig(H/'diagnostic_figure.png');plt.close(fig)
    return rows

def new_results():
    coef=json.loads((H/'selection.json').read_text())['selected_coef']
    if coef is None:return []
    results=[];trace=[]
    for seed in range(221,226):
        labels=[f'seed{seed}_lambda0',f'seed{seed}_lambda{coef:g}'];a,b=[pd.read_csv(H/l/'test.csv').set_index('reset') for l in labels]
        common=a.eligible&b.eligible;ix=a.index[common]
        row=dict(seed=seed,common=len(ix),control_healthy=int((a.complete&(a.mean_speed>=.5)).sum()),smooth_healthy=int((b.complete&(b.mean_speed>=.5)).sum()),reward_ratio=float(b.padded_reward.mean()/a.padded_reward.mean()),speed_ratio=float((b.loc[ix,'mean_speed']/a.loc[ix,'mean_speed']).median()))
        for k in ['J1','J2','recurrence','section_dispersion']:
            ratio=b.loc[ix,k]/a.loc[ix,k];row[k+'_ratio']=float(ratio.median())
            for reset,val in ratio.items():trace.append(dict(seed=seed,reset=int(reset),metric=k,ratio=float(val)))
        for key in ['entropy','autocorr','std','period_cv']:
            ratios=[]
            for reset in ix:
                vals=[json.loads((H/l/'step1048576'/f'reset{reset}'/'metrics.json').read_text())['cheap'][key] for l in labels]
                if key=='autocorr':vals=[1-v for v in vals]
                if vals[0] is not None and vals[0]>0 and vals[1] is not None:ratios.append(vals[1]/vals[0])
            row['cheap_'+key+'_ratio']=float(np.median(ratios)) if ratios else None
        row['independent_event']=bool(len(ix)>=8 and row['recurrence_ratio']<=.75 and row['section_dispersion_ratio']<=.75 and a.loc[ix,'recurrence'].median()>=.02 and a.loc[ix,'section_dispersion'].median()>=.01 and row['reward_ratio']>=.9 and row['smooth_healthy']>=row['control_healthy']-1)
        df=pd.read_csv(H/f'MG_seed{seed}.csv')
        for mode in ['fixed','cycles','left']:
            x=df[(df.label==labels[0])&(df['mode']==mode)].set_index('reset');y=df[(df.label==labels[1])&(df['mode']==mode)].set_index('reset')
            ratio=(y.MG/x.MG)[x.all_valid&y.all_valid].dropna();row['MG_'+mode]=float(ratio.median());row['MG_'+mode+'_n']=len(ratio)
            row['MG_'+mode+'_ident_max']=float(max(x.ident_max.max(),y.ident_max.max()))
            for reset,val in ratio.items():trace.append(dict(seed=seed,reset=int(reset),metric='MG_'+mode,ratio=float(val)))
        row['MG_assessable']=bool(row['MG_fixed_n']>=8)
        pair=json.loads((H/f'pair{seed}.json').read_text());reset=pair['common_resets'][0] if pair['common_resets'] else None
        if reset:
            pa,pb=[json.loads((H/l/'step1048576'/f'reset{reset}'/'fixed600_2.json').read_text()) for l in labels]
            row['amplification_ratio']=pb['median_amplification']/pa['median_amplification'];row['probe_reset']=reset
            row['probe_falls_control']=pa['falls'];row['probe_falls_smooth']=pb['falls']
        results.append(row)
    frame=pd.DataFrame(results);frame.to_csv(H/'all_seeds.csv',index=False);pd.DataFrame(trace).to_csv(H/'paired_traces.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    groups=[(['J1_ratio','J2_ratio','reward_ratio'],['Изменение команд J1','Вторая разность J2','Исходная награда']),(['recurrence_ratio','section_dispersion_ratio','amplification_ratio'],['Ошибка возврата R','Разброс сечения D','Усиление возмущений A']),(['MG_fixed','MG_cycles','MG_left'],['MG: фиксированное окно','MG: окно по циклам','MG: левое колено'])]
    for ax,(keys,names) in zip(axes,groups):
        for k,name in zip(keys,names):ax.plot(frame.seed,frame[k],'o-',label=name)
        ax.axhline(1,color='gray',ls='--');ax.set(xlabel='Сид дообучения',ylabel='Сглаживание / контроль',xticks=frame.seed);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(H/'results.pdf');fig.savefig(H/'results.png');plt.close(fig)
    return results

if __name__=='__main__':
    old=old_results();new=new_results();(H/'summary.json').write_text(json.dumps(dict(diagnostic=old,confirmation=new),indent=2,allow_nan=False))
