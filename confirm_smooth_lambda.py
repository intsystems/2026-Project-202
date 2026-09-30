from pathlib import Path
import concurrent.futures, subprocess, sys, json

H=Path(__file__).resolve().parent/'research_walker_smooth_lambda'
COEFS=[0,0.25,1,4]
SEEDS=[231,232,233,234,235]

def label(seed,coef): return f'seed{seed}_lambda{coef:g}'

def train(job):
    seed,coef=job;lab=label(seed,coef);log=H/(lab+'.train.log')
    if not (H/lab/'train.json').exists():
        with log.open('w') as f:
            r=subprocess.run([sys.executable,str(H/'train.py'),'--seed',str(seed),'--coef',str(coef)],cwd=H.parent,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode: raise RuntimeError(f'train failed: {lab}')
    return lab

def evaluate(lab):
    log=H/(lab+'.test.log')
    with log.open('w') as f:
        r=subprocess.run([sys.executable,str(H/'evaluate.py'),'--label',lab,'--split','test'],cwd=H.parent,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'test failed: {lab}')
    return lab

def main():
    jobs=[(s,c) for s in SEEDS for c in COEFS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(train,jobs))
    labels=[label(s,c) for s,c in jobs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate,labels))
    (H/'confirmation_status.json').write_text(json.dumps(dict(seeds=SEEDS,coefs=COEFS,completed=labels),indent=2))
    print('CONFIRMATION COMPLETE',len(labels))

if __name__=='__main__': main()
