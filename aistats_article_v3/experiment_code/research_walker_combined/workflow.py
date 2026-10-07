from pathlib import Path
import concurrent.futures, subprocess, sys, json, hashlib
import pandas as pd

H=Path(__file__).resolve().parent
ARMS=[(0,0),(0,1),(3,0),(3,1)]

def run_train(arm):
    track,smooth=arm;label=f'seed280_track{track:g}_smooth{smooth:g}'
    log=H/(label+'.train.log')
    if not (H/label/'train.json').exists():
        with log.open('w') as f:
            r=subprocess.run([sys.executable,str(H/'train.py'),'--seed','280','--coef',str(track),'--smooth',str(smooth)],cwd=H.parent,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode: raise RuntimeError(f'{label} train failed')
    return label

def run_eval(label):
    log=H/(label+'.validation.log')
    with log.open('w') as f:
        r=subprocess.run([sys.executable,str(H/'evaluate.py'),'--label',label,'--split','validation'],cwd=H.parent,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'{label} validation failed')
    return label

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        labels=list(pool.map(run_train,ARMS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_eval,labels))
    rows=[]
    for label in labels:
        d=pd.read_csv(H/label/'validation.csv');d=d[d.step==1048576]
        track=float(label.split('_track')[1].split('_')[0]);smooth=float(label.split('_smooth')[1])
        rows.append(dict(label=label,track=track,smooth=smooth,
            healthy=int((d.complete&(d.mean_speed>=.5)).sum()),
            eligible=int(d.eligible.sum()),
            reward=float(d.padded_reward.mean()),
            R=float(d.loc[d.eligible,'recurrence'].median()) if d.eligible.any() else None,
            D=float(d.loc[d.eligible,'D_strobe'].median()) if d.eligible.any() else None,
            C=float(d.loc[d.eligible,'C_cycle'].median()) if d.eligible.any() else None,
            J1=float(d.loc[d.eligible,'J1'].median()) if d.eligible.any() else None,
            J2=float(d.loc[d.eligible,'J2'].median()) if d.eligible.any() else None))
    result=dict(protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest(),rows=rows)
    (H/'pilot_summary.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(rows).to_csv(H/'pilot_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
