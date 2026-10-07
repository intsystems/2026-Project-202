from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from measure import experiments
H=Path(__file__).resolve().parent

def run():
    rows=[];paired=[]
    for seed,coef in experiments():
        labels=[f'seed{seed}_lambda0',f'seed{seed}_lambda{coef:g}'];a,b=[pd.read_csv(H/l/'test.csv').set_index('reset') for l in labels]
        pair=json.loads((H/f'pair{seed}_lambda{coef:g}.json').read_text());idx=pair['common_resets'];d=pd.read_csv(H/f'MG_seed{seed}_lambda{coef:g}.csv')
        r=dict(seed=seed,coef=coef,n=len(idx),control_walk=int((a.complete&(a.mean_speed>=.5)).sum()),imitation_walk=int((b.complete&(b.mean_speed>=.5)).sum()),reward_ratio=float(b.padded_reward.mean()/a.padded_reward.mean()))
        for key in ['recurrence','D_section','section_dispersion','orbit_distance2','J1','J2','mean_speed']:
            ratios=b.loc[idx,key]/a.loc[idx,key];r[key]=float(ratios.median()) if len(ratios) else None
            for reset,v in ratios.items():paired.append(dict(seed=seed,coef=coef,reset=int(reset),metric=key,ratio=float(v)))
        r['independent_event']=bool(len(idx)>=8 and r['recurrence']<=.75 and r['D_section']<=.75 and a.loc[idx,'recurrence'].median()>=.02 and a.loc[idx,'D_section'].median()>=.01 and r['reward_ratio']>=.9 and r['imitation_walk']>=r['control_walk']-1)
        for mode in ['fixed','cycles','left','tau4','tau16']:
            x=d[(d.label==labels[0])&(d['mode']==mode)].set_index('reset');y=d[(d.label==labels[1])&(d['mode']==mode)].set_index('reset')
            valid_idx=sorted(set(idx)&set(x[x.all_valid].index)&set(y[y.all_valid].index))
            ratio=y.loc[valid_idx,'MG']/x.loc[valid_idx,'MG'];r['MG_'+mode]=float(ratio.median()) if len(ratio) else None;r['MG_'+mode+'_n']=len(ratio)
            r['ident_'+mode]=float(max(x.ident_max.max(),y.ident_max.max()))
            for reset,v in ratio.items():paired.append(dict(seed=seed,coef=coef,reset=int(reset),metric='MG_'+mode,ratio=float(v)))
        for key in ['entropy','autocorr','std','period_cv']:
            vals=[]
            for reset in idx:
                x,y=[json.loads((H/l/'step1048576'/f'reset{reset}'/'metrics.json').read_text())['cheap'][key] for l in labels]
                if key=='autocorr':x,y=1-x,1-y
                if x is not None and y is not None and x>0:vals.append(y/x)
            r['cheap_'+key]=float(np.median(vals)) if vals else None
        if idx:
            p,q=[json.loads((H/l/'step1048576'/f'reset{idx[0]}'/'fixed600_2.json').read_text()) for l in labels]
            r['amplification']=q['median_amplification']/p['median_amplification'];r['probe_reset']=idx[0];r['probe_falls_control']=p['falls'];r['probe_falls_imitation']=q['falls']
        rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(H/'all_results.csv',index=False);pd.DataFrame(paired).to_csv(H/'paired_traces.csv',index=False)
    (H/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(11,3.4));pos=np.arange(len(df));ticks=[str(s) if len(df)>2 else f'lambda={c:g}' for s,c in zip(df.seed,df.coef)]
    groups=[(['orbit_distance2','reward_ratio'],['Ошибка имитации','Исходная награда']),(['recurrence','D_section','section_dispersion','amplification'],['Возврат R','Фиксированное сечение D','Максимумы колена','Возмущения']),(['MG_fixed','MG_cycles','MG_left','MG_tau4','MG_tau16'],['MG: фиксированное окно','MG: окно по циклам','MG: левое колено','MG: задержка4','MG: задержка16'])]
    for a,(keys,names) in zip(ax,groups):
        for k,n in zip(keys,names):a.plot(pos,df[k],'o-',label=n)
        a.axhline(1,c='gray',ls='--');a.set(xticks=pos,xticklabels=ticks,ylabel='Имитация / контроль');a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(H/'results.pdf');fig.savefig(H/'results.png',dpi=150);plt.close(fig)
    print(df.to_string(index=False))

if __name__=='__main__':run()
