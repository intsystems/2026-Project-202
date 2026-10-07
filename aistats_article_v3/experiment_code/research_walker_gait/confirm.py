from pathlib import Path
import concurrent.futures,json,subprocess,sys,time
H=Path(__file__).resolve().parent

def one(seed,horizon):
    out=H/f'seed{seed}';out.mkdir(exist_ok=True);start=time.perf_counter()
    commands=[]
    if not (out/f'train_{horizon}.json').exists():commands.append(('train',['train.py','--seed',str(seed),'--steps',str(horizon)]))
    commands += [('motion',['motion.py','--seed',str(seed)]),('pair',['motion.py','--seed',str(seed),'--pair',str(horizon)]),
        ('perturb',['perturb.py','--seed',str(seed)]),('MG',['features.py','--seed',str(seed)])]
    for stage,args in commands:
        print('START',seed,stage,flush=True)
        with (out/(stage+'_process.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,str(H/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,cwd=H.parent)
        if r.returncode:return dict(seed=seed,status='failed',stage=stage,code=r.returncode)
    pair=json.loads((out/'pair.json').read_text())
    result=dict(seed=seed,status='complete',seconds=time.perf_counter()-start,usable=pair['usable'],early=pair['early'])
    print('DONE',json.dumps(result),flush=True);return result

if __name__=='__main__':
    selection=json.loads((H/'selection.json').read_text());assert selection['selected_without_MG']
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=selection.get('confirmation_workers',2)) as pool:
        jobs=[pool.submit(one,s,selection['horizon']) for s in selection['confirmation_seeds']]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result());(H/'confirmation_status.json').write_text(json.dumps(results,indent=2))
    if any(r['status']=='failed' for r in results):raise SystemExit('A stage failed; inspect logs.')
