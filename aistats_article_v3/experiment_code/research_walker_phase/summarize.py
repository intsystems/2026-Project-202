import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from measure import experiments
from motion import H

def run():
    rows=[];trace=[]
    for seed in experiments():
        labels=[f'seed{seed}_lambda0',f'seed{seed}_lambda3'];a,b=[pd.read_csv(H/l/'test.csv').set_index('reset') for l in labels];idx=json.loads((H/f'pair{seed}.json').read_text())['common_resets'];d=pd.read_csv(H/f'MG_seed{seed}.csv')
        r=dict(seed=seed,n=len(idx),control_walk=int((a.complete&(a.mean_speed>=.5)).sum()),tracking_walk=int((b.complete&(b.mean_speed>=.5)).sum()),reward_ratio=float(b.padded_reward.mean()/a.padded_reward.mean()))
        for key in ['recurrence','D_strobe','section_dispersion','tracking_error2','J1','J2','mean_speed']:
            vals=b.loc[idx,key]/a.loc[idx,key];r[key]=float(vals.median()) if len(vals) else None
            for reset,v in vals.items():trace.append(dict(seed=seed,reset=reset,metric=key,ratio=float(v)))
        r['event']=bool(len(idx)>=8 and r['recurrence']<=.75 and r['D_strobe']<=.75 and a.loc[idx,'recurrence'].median()>=.02 and a.loc[idx,'D_strobe'].median()>=.001 and r['reward_ratio']>=.9 and r['tracking_walk']>=r['control_walk']-1)
        for mode in ['fixed','cycles','left','tau4','tau16']:
            x=d[(d.label==labels[0])&(d['mode']==mode)].set_index('reset');y=d[(d.label==labels[1])&(d['mode']==mode)].set_index('reset');ix=sorted(set(idx)&set(x[x.valid].index)&set(y[y.valid].index))
            vals=y.loc[ix,'MG']/x.loc[ix,'MG'];r['MG_'+mode]=float(vals.median()) if len(vals) else None;r['MG_'+mode+'_n']=len(vals);r['ident_'+mode]=float(max(x.ident_max.max(),y.ident_max.max()))
            for reset,v in vals.items():trace.append(dict(seed=seed,reset=reset,metric='MG_'+mode,ratio=float(v)))
        if idx:
            p,q=[json.loads((H/l/'step1048576'/f'reset{idx[0]}'/'probe_eps0.001_a2.json').read_text()) for l in labels]
            r.update(amplification=q['median_amplification']/p['median_amplification'],probe_reset=idx[0],falls_control=p['falls'],falls_tracking=q['falls'])
            if seed==experiments()[0]:
                for label in labels:
                    cp=H/label/'step1048576'/f'reset{idx[0]}';big=np.load(cp/'probe_eps0.001_a2.npz')['jacobians'][0];small=np.load(cp/'probe_eps0.0001_a1.npz')['jacobians'][0]
                    err=float(np.linalg.norm(big-small)/max(np.linalg.norm(small),1e-30));r['jacobian_relative_error_'+label.split('_')[-1]]=err
        for key in ['entropy','autocorr']:
            vals=[]
            for reset in idx:
                x,y=[json.loads((H/l/'step1048576'/f'reset{reset}'/'metrics.json').read_text())['cheap'][key] for l in labels]
                if key=='autocorr':x,y=1-x,1-y
                vals.append(y/x)
            r['cheap_'+key]=float(np.median(vals)) if vals else None
        rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(H/'all_results.csv',index=False);pd.DataFrame(trace).to_csv(H/'paired_traces.csv',index=False);(H/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    if len(df)==1:
        r=df.iloc[0];keys=['reward_ratio','tracking_error2','recurrence','D_strobe','amplification'];axes[0].bar(range(len(keys)),[r[k] for k in keys]);axes[0].set_xticks(range(len(keys)),['Награда','Ошибка\nслежения','Возврат R','Сечение D','Возмущения'])
        keys=['MG_fixed','MG_cycles','MG_left','MG_tau4','MG_tau16'];axes[1].bar(range(len(keys)),[r[k] for k in keys]);axes[1].set_xticks(range(len(keys)),['Фикс.','14 циклов','Левое\nколено','Лаг4','Лаг16'])
    else:
        for k,name in [('recurrence','R'),('D_strobe','D'),('amplification','A')]:axes[0].plot(df.seed,df[k],'o-',label=name)
        for k,name in [('MG_fixed','фикс.'),('MG_cycles','циклы'),('MG_left','левое')]:axes[1].plot(df.seed,df[k],'o-',label=name)
        for ax in axes:ax.legend()
    for ax in axes:ax.axhline(1,c='gray',ls='--');ax.set_ylabel('Слежение / контроль')
    fig.tight_layout();fig.savefig(H/'results.pdf');fig.savefig(H/'results.png',dpi=150);plt.close(fig)
    print(df.to_string(index=False))

if __name__=='__main__':run()
