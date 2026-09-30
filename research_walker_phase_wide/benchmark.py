import json,time,shutil
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,rollout,state_reference,cheap
from evaluate import strobe
from features import measure
from probes import probe
from measure import experiments

def run():
    seed=experiments()[0];pair=json.loads((H/f'pair{seed}.json').read_text());reset=pair['common_resets'][0];rows=[]
    for label in [f'seed{seed}_lambda0',f'seed{seed}_lambda3']:
        source=H/label/'step1048576';target=H/'benchmark'/label;target.mkdir(parents=True,exist_ok=True)
        for name in ['policy.zip','normalize.pkl']:shutil.copy2(source/name,target/name)
        actor=Actor(target)
        for rep in range(3):
            cp=target/f'repeat{rep}';cp.mkdir(exist_ok=True);m=rollout(actor,cp,reset);d=np.load(cp/f'reset{reset}'/'trajectory.npz');x=d['qpos'][:,4]
            np.testing.assert_array_equal(d['qpos'],np.load(source/f'reset{reset}'/'trajectory.npz')['qpos'])
            rows.append(dict(label=label,repeat=rep,method='acquisition',seconds=m['acquisition_seconds']))
            if rep==0:measure(x[:2048],2048,8)
            tic=time.perf_counter();mg=[measure(x[end-2048:end],2048,8) for end in [2048,3072,4096]]
            rows.append(dict(label=label,repeat=rep,method='MG20+40',seconds=time.perf_counter()-tic));rows.append(dict(label=label,repeat=rep,method='MG20',seconds=sum(v['MG_seconds'] for v in mg)))
            tic=time.perf_counter()
            for end in [2048,3072,4096]:
                state_reference(d['qpos'][end-2048:end],d['qvel'][end-2048:end]);strobe({k:d[k][end-2048:end] for k in ['qpos','qvel','phase']})
            rows.append(dict(label=label,repeat=rep,method='full_state',seconds=time.perf_counter()-tic));tic=time.perf_counter()
            for end in [2048,3072,4096]:cheap(x[end-2048:end])
            rows.append(dict(label=label,repeat=rep,method='cheap_scalar',seconds=time.perf_counter()-tic))
        root=target/f'reset{reset}';root.mkdir(exist_ok=True)
        for name in ['trajectory.npz','metrics.json']:shutil.copy2(source/f'reset{reset}'/name,root/name)
        result=probe(target,reset,force=True);rows.append(dict(label=label,repeat=0,method='probes68',seconds=result['seconds']))
    frame=pd.DataFrame(rows);frame.to_csv(H/'timings.csv',index=False);print(frame.groupby('method').seconds.median().to_string())

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run()
