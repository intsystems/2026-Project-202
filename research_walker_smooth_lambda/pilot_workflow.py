from pathlib import Path
import concurrent.futures, subprocess, sys, json, hashlib
import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
COEFS=[0,0.25,1,4]
PILOT=230

def label(coef): return f'seed{PILOT}_lambda{coef:g}'

def train(coef):
    lab=label(coef); log=H/(lab+'.train.log')
    if not (H/lab/'train.json').exists():
        with log.open('w') as f:
            r=subprocess.run([sys.executable,str(H/'train.py'),'--seed',str(PILOT),'--coef',str(coef)],cwd=H.parent,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode: raise RuntimeError(f'train failed: {lab}')
    return lab

def evaluate(lab):
    log=H/(lab+'.validation.log')
    with log.open('w') as f:
        r=subprocess.run([sys.executable,str(H/'evaluate.py'),'--label',lab,'--split','validation'],cwd=H.parent,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'validation failed: {lab}')
    return lab

def summarize(labels):
    frames={lab:pd.read_csv(H/lab/'validation.csv') for lab in labels}
    final={lab:d[d.step==1048576].set_index('reset') for lab,d in frames.items()}
    base=final[label(0)]
    rows=[]
    for coef in COEFS:
        lab=label(coef);d=final[lab];common=base.eligible & d.eligible
        def med(col):
            x=d.loc[common,col]/base.loc[common,col]
            return float(x.median()) if len(x) else None
        rows.append(dict(label=lab,coef=coef,healthy=int((d.complete&(d.mean_speed>=.5)).sum()),
            eligible=int(d.eligible.sum()),common=int(common.sum()),
            reward_ratio=float(d.padded_reward.mean()/base.padded_reward.mean()),
            J1_ratio=med('J1'),J2_ratio=med('J2'),R_ratio=med('recurrence'),
            D_ratio=med('section_dispersion'),
            baseline_R=float(base.loc[common,'recurrence'].median()) if common.any() else None,
            baseline_D=float(base.loc[common,'section_dispersion'].median()) if common.any() else None))
    out=pd.DataFrame(rows);out.to_csv(H/'pilot_summary.csv',index=False)
    result=dict(protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest(),rows=rows)
    (H/'pilot_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(out.to_string(index=False))

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        labels=list(pool.map(train,COEFS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate,labels))
    summarize(labels)

if __name__=='__main__': main()
