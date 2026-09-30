from pathlib import Path
import concurrent.futures,json,subprocess,sys,time
H=Path(__file__).resolve().parent

def one(seed):
    out=H/f'seed{seed}';out.mkdir(exist_ok=True);start=time.perf_counter()
    for script,marker in [('train.py','meta.json'),('features.py','features.csv'),('monitor.py','evaluation.json')]:
        if (out/marker).exists():continue
        print('START',seed,script,flush=True)
        with (out/(script+'.log')).open('w',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,str(H/script),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=H.parent.parent)
        if r.returncode:return dict(seed=seed,status='failed',script=script,code=r.returncode)
    d=json.loads((out/'evaluation.json').read_text())
    result=dict(seed=seed,status='complete',seconds=time.perf_counter()-start,events=d['dense']['events'])
    print('DONE',json.dumps(result),flush=True);return result

if __name__=='__main__':
    selection=json.loads((H/'selection.json').read_text());assert selection['confirmation_seeds']==list(range(101,110))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,s) for s in selection['confirmation_seeds']]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result());(H/'confirmation_status.json').write_text(json.dumps(results,indent=2))
    if any(r['status']=='failed' for r in results):raise SystemExit('A stage failed; inspect seed logs.')
