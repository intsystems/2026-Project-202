from pathlib import Path
import sys,subprocess,json,concurrent.futures,argparse,hashlib
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent

def stage(label,script,args):
    root=H/label;root.mkdir(exist_ok=True)
    with (root/(script.replace('.py','')+'_'+'_'.join(args).replace('--','')+'.log')).open('w') as log:
        r=subprocess.run([sys.executable,str(H/script),*args],cwd=H.parent,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'{label}: {script}: {r.returncode}')

def one(seed,coef,test=False):
    label=f'seed{seed}_lambda{coef:g}'
    if not (H/label/'train.json').exists():stage(label,'train.py',['--seed',str(seed),'--coef',str(coef)])
    stage(label,'evaluate.py',['--label',label,'--split','validation'])
    if test:stage(label,'evaluate.py',['--label',label,'--split','test'])
    print('DONE',label,flush=True);return label

def pilot_gate(coef):
    a=pd.read_csv(H/'seed220_lambda0'/'validation.csv');b=pd.read_csv(H/f'seed220_lambda{coef:g}'/'validation.csv')
    x=a[a.step==1048576].set_index('reset');y=b[b.step==1048576].set_index('reset')
    common=x.complete&y.complete&(x.mean_speed>=.5)&(y.mean_speed>=.5)
    healthy=[int((g.complete&(g.mean_speed>=.5)).sum()) for _,g in b.groupby('step')]
    reward=float(y.padded_reward.mean()/x.padded_reward.mean());smooth=float((y.loc[common,'J1']/x.loc[common,'J1']).median())
    result=dict(coef=coef,healthy_last3=healthy,control_healthy_last3=[int((g.complete&(g.mean_speed>=.5)).sum()) for _,g in a.groupby('step')],reward_ratio=reward,J1_ratio=smooth,
        passes_health=bool(min(healthy)>=4 and reward>=.9),passes_smooth=bool(smooth<=.8))
    (H/f'pilot_gate_lambda{coef:g}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result

def pilot():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda c:one(220,c),[0,1]))
    g=pilot_gate(1);chosen=None
    if g['passes_health']:
        if g['passes_smooth']:chosen=1
        else:
            one(220,10);g=pilot_gate(10)
            if g['passes_health'] and g['passes_smooth']:chosen=10
    (H/'selection.json').write_text(json.dumps(dict(selected_coef=chosen,pilot=g,confirmation_seeds=list(range(221,226)),protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()),indent=2))

def confirm():
    sel=json.loads((H/'selection.json').read_text());coef=sel['selected_coef'];assert coef is not None,'Pilot failed'
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(one,s,c,True) for s in range(221,226) for c in [0,coef]]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result());(H/'confirmation_status.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','confirm']);a=p.parse_args()
    if a.mode=='pilot':pilot()
    else:confirm()
