from pathlib import Path
import sys,subprocess,json,concurrent.futures,argparse,hashlib
import pandas as pd
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

def gate(coef):
    a=pd.read_csv(H/'seed240_lambda0/validation.csv');b=pd.read_csv(H/f'seed240_lambda{coef:g}/validation.csv')
    x=a[a.step==1048576].set_index('reset');y=b[b.step==1048576].set_index('reset')
    eligible=x.eligible&y.eligible&x.section_eligible.fillna(False)&y.section_eligible.fillna(False)
    ratios={k:float((y.loc[eligible,k]/x.loc[eligible,k]).median()) for k in ['recurrence','D_section','section_dispersion','orbit_distance2']}
    reward=float(y.padded_reward.mean()/x.padded_reward.mean());healthy=int((y.complete&(y.mean_speed>=.5)).sum())
    passed=healthy>=4 and reward>=.9 and eligible.sum()>=4 and ratios['recurrence']<=.8 and ratios['D_section']<=.8 and x.loc[eligible,'recurrence'].median()>=.02 and x.loc[eligible,'D_section'].median()>=.01
    return dict(coef=coef,passed=bool(passed),healthy=healthy,common=int(eligible.sum()),reward_ratio=reward,ratios=ratios,last3_healthy=[int((g.complete&(g.mean_speed>=.5)).sum()) for _,g in b.groupby('step')])

def pilot():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda c:one(240,c),[0,1,3]))
    checks=[gate(c) for c in [1,3]];chosen=next((g['coef'] for g in checks if g['passed']),None)
    result=dict(selected_coef=chosen,pilots=checks,confirmation_seeds=list(range(241,246)),protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest())
    (H/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def confirm():
    sel=json.loads((H/'selection.json').read_text());coef=sel['selected_coef'];assert coef is not None
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(one,s,c,True) for s in range(241,246) for c in [0,coef]]
        for job in concurrent.futures.as_completed(jobs):results.append(job.result());(H/'confirmation_status.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','confirm']);a=p.parse_args()
    if a.mode=='pilot':pilot()
    else:confirm()
