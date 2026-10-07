import concurrent.futures,json,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent

def one(seed,conservative):
    # The _B suffix identifies the configuration, not a different random seed.
    label=f'seed{seed}'+('_B' if conservative else '');out=H/label;out.mkdir(exist_ok=True)
    stages=[('train.py',['--seed',str(seed)]+(['--conservative'] if conservative else [])),
        ('evaluate.py',['--label',label]),('evaluate.py',['--label',label,'--gate']),
        ('evaluate.py',['--label',label,'--split','test','--steps','0','1048576']),
        ('evaluate.py',['--label',label,'--pair']),('measure.py',['--label',label])]
    for i,(script,args) in enumerate(stages):
        print('START',label,script,args,flush=True)
        with (out/f'process_{i}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(H/script),*args],cwd=H.parent,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:return dict(label=label,status='error',stage=script,returncode=r.returncode)
    pair=json.loads((out/'pair.json').read_text());gate=json.loads((out/'gate.json').read_text())
    result=dict(label=label,status='complete',usable=pair['usable'],gate=gate['passed']);print('DONE',json.dumps(result),flush=True);return result

if __name__=='__main__':
    selection=json.loads((H/'selection.json').read_text());results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        jobs=[pool.submit(one,seed,selection['conservative']) for seed in selection['confirmation_seeds']]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result());(H/'confirmation_status.json').write_text(json.dumps(results,indent=2))
    assert all(r['status']=='complete' for r in results),'Inspect per-seed process logs'
