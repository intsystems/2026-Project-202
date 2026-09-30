from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from analyze import measure,change
from run_protection import source,retention

def analyze(seed,mode):
    out=H/mode/f'seed{seed}';src=source(seed)
    assert json.loads((out/'meta.json').read_text())['steps']==3072
    log=pd.read_csv(out/'logs.csv');assert np.array_equal(log.step,np.arange(3072))
    original=pd.read_csv(src/'windows.csv');rows=[]
    for window,tau in [(512,1),(256,1),(1024,1),(512,4)]:
        for end in range(window,3073,128):
            rows.append(dict(arm='protected',window=window,tau=tau,end=end,
                **measure(log.probe_nll.to_numpy()[end-window:end],window,tau)))
    protected=pd.DataFrame(rows);all_windows=pd.concat([original,protected],ignore_index=True)
    all_windows.to_csv(out/'three_arm_windows.csv',index=False);sens=[]
    for window,tau in [(512,1),(256,1),(1024,1),(512,4)]:
        sub=all_windows[(all_windows.window==window)&(all_windows.tau==tau)];arms={}
        for arm in ['base','regularized','protected']:
            a=sub[sub.arm==arm];arms[arm]={k:change(a,k,lower=max(512,window)) for k in ['MG','std','entropy']}
        assert np.isclose(arms['base']['MG']['before'],arms['protected']['MG']['before'])
        qreg=arms['regularized']['MG']['after']/arms['base']['MG']['after']
        qprot=arms['protected']['MG']['after']/arms['base']['MG']['after']
        ratio=qprot/qreg
        sens.append(dict(window=window,tau=tau,arms=arms,q_regularized=qreg,q_protected=qprot,
            protected_vs_regularized=ratio,closer_to_base=bool(abs(qprot-1)<abs(qreg-1))))
    ref={};frames={}
    for arm,d in [('base',src/'base'),('regularized',src/'regularized'),('protected',out)]:
        r=pd.read_csv(d/'reference.csv');frames[arm]=r
        ref[arm]={k:change(r,k,'step') for k in ['KL','MI','shuffle_symkl','shuffle_nll_gap','nll']}
    result=dict(seed=seed,mode=mode,retention=retention(seed,out),reference=ref,scalar=sens,
        primary_protected_degenerate_windows=int(protected.query('window==512 and tau==1').degenerate.sum()))
    (out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(seed=seed,
        protection_valid=result['retention']['protection_valid'],q_regularized=sens[0]['q_regularized'],
        q_protected=sens[0]['q_protected'],R=sens[0]['protected_vs_regularized']),indent=2),flush=True)
    fig,axs=plt.subplots(2,2,figsize=(10,5.7),constrained_layout=True)
    for arm,col in [('base','#777777'),('regularized','#0868ac'),('protected','#d95f02')]:
        r=frames[arm];d=out if arm=='protected' else src/arm;l=pd.read_csv(d/'logs.csv')
        w=all_windows[(all_windows.arm==arm)&(all_windows.window==512)&(all_windows.tau==1)]
        axs[0,0].plot(r.step,r.MI,color=col,label=arm);axs[0,1].plot(r.step,r.shuffle_symkl,color=col,label=arm)
        axs[1,0].plot(l.step,l.probe_nll,color=col,label=arm,lw=.6);axs[1,1].plot(w.end,w.MG,color=col,label=arm)
    titles=['Sentence-code MI (nats)','Prediction response to code shuffling','Observed reconstruction NLL/token','MG, W=512, delay=1']
    for ax,title in zip(axs.flat,titles):
        ax.set_title(title);ax.set_xlabel('Outer training step');ax.axvline(1024,color='black',ls=':',lw=1);ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'comparison.pdf');fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['encoder5','freebits'],required=True);a=p.parse_args()
    measure(np.sin(np.arange(512)/11)+np.cos(np.arange(512)/7),512)
    with threadpool_limits(limits=1):analyze(a.seed,a.mode)
