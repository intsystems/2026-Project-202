from pathlib import Path
import sys,subprocess,json,concurrent.futures,argparse,hashlib
import pandas as pd
H=Path(__file__).resolve().parent

def stage(label,script,args):
    root=H/label;root.mkdir(exist_ok=True)
    with (root/(script.replace('.py','')+'_'+'_'.join(args).replace('--','')+'.log')).open('w') as log:r=subprocess.run([sys.executable,str(H/script),*args],cwd=H.parent,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'{label}: {script}: {r.returncode}')

def one(seed,coef,test=False):
    label=f'seed{seed}_lambda{coef:g}'
    if not (H/label/'train.json').exists():stage(label,'train.py',['--seed',str(seed),'--coef',str(coef)])
    stage(label,'evaluate.py',['--label',label,'--split','validation'])
    if test:stage(label,'evaluate.py',['--label',label,'--split','test'])
    print('DONE',label,flush=True);return label

def pilot():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda c:one(260,c),[0,3]))
    a,b=[pd.read_csv(H/f'seed260_lambda{c}/validation.csv') for c in [0,3]];a=a[a.step==1048576].set_index('reset');b=b[b.step==1048576].set_index('reset')
    ok=a.eligible&b.eligible&a.section_eligible&b.section_eligible;ratios={k:float((b.loc[ok,k]/a.loc[ok,k]).median()) for k in ['recurrence','D_strobe','tracking_error2']}
    reward=float(b.padded_reward.mean()/a.padded_reward.mean());healthy=int((b.complete&(b.mean_speed>=.5)).sum())
    passed=healthy>=4 and ok.sum()>=4 and reward>=.9 and ratios['recurrence']<=.8 and ratios['D_strobe']<=.8 and a.loc[ok,'recurrence'].median()>=.02 and a.loc[ok,'D_strobe'].median()>=.001
    r=dict(selected_coef=3 if passed else None,pilot=dict(healthy=healthy,common=int(ok.sum()),reward_ratio=reward,ratios=ratios),protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest())
    (H/'selection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

def confirm():
    assert json.loads((H/'selection.json').read_text())['selected_coef']==3
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(one,s,c,True) for s in range(261,266) for c in [0,3]]
        for job in concurrent.futures.as_completed(jobs):results.append(job.result());(H/'confirmation_status.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','confirm']);a=p.parse_args()
    if a.mode=='pilot':pilot()
    else:confirm()
