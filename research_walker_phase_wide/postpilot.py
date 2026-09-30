import json,concurrent.futures,subprocess,sys
from workflow import H,stage

def run():
    selected=json.loads((H/'selection.json').read_text())['selected_coef']
    if selected is None:
        def ev(c):stage(f'seed270_lambda{c}','evaluate.py',['--label',f'seed270_lambda{c}','--split','test'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(ev,[0,3]))
        seeds=[270]
    else:
        with (H/'confirmation_process.log').open('w') as f:r=subprocess.run([sys.executable,str(H/'workflow.py'),'confirm'],stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0;seeds=list(range(271,276))
    def meas(s):stage(f'seed{s}_lambda3','measure.py',['--seed',str(s)])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(meas,seeds))
    print('POSTPILOT COMPLETE',flush=True)

if __name__=='__main__':run()
