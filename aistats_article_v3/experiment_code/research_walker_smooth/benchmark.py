from pathlib import Path
import json,time,shutil
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from motion import Actor,rollout,state_reference,cheap
from diagnostic import measure
from perturb_fixed import probe
H=Path(__file__).resolve().parent

def run():
    c=json.loads((H/'selection.json').read_text())['selected_coef'];pair=json.loads((H/'pair221.json').read_text());reset=pair['common_resets'][0];rows=[]
    for label in [f'seed221_lambda0',f'seed221_lambda{c:g}']:
        source=H/label/'step1048576';target=H/'benchmark'/label;target.mkdir(parents=True,exist_ok=True)
        for name in ['policy.zip','normalize.pkl']:shutil.copy2(source/name,target/name)
        actor=Actor(target)
        # Separate identical trace copies keep acquisition uncached.
        for rep in range(3):
            cp=target/f'repeat{rep}';cp.mkdir(exist_ok=True)
            m=rollout(actor,cp,reset);data=np.load(cp/f'reset{reset}'/'trajectory.npz');x=data['qpos'][:,4]
            np.testing.assert_array_equal(data['qpos'],np.load(source/f'reset{reset}'/'trajectory.npz')['qpos'])
            rows.append(dict(label=label,repeat=rep,method='acquisition_4096',seconds=m['acquisition_seconds']))
            # Warm both dimensions before repeated measurement.
            if rep==0:measure(x[:2048],2048,8)
            windows=[(end-2048,end) for end in [2048,3072,4096]]
            tic=time.perf_counter();mg=[measure(x[lo:hi],2048,8) for lo,hi in windows];both=time.perf_counter()-tic
            rows.append(dict(label=label,repeat=rep,method='MG_E20_3windows',seconds=sum(v['MG_seconds'] for v in mg)))
            rows.append(dict(label=label,repeat=rep,method='MG_E20_E40_3windows',seconds=both))
            tic=time.perf_counter()
            for lo,hi in windows:state_reference(data['qpos'][lo:hi],data['qvel'][lo:hi])
            rows.append(dict(label=label,repeat=rep,method='full_state_RD_3windows',seconds=time.perf_counter()-tic))
            tic=time.perf_counter()
            for lo,hi in windows:cheap(x[lo:hi])
            rows.append(dict(label=label,repeat=rep,method='cheap_scalar_3windows',seconds=time.perf_counter()-tic))
        # Original checkpoint layout for probes, one frozen-horizon cost per arm.
        p=target/f'reset{reset}';p.mkdir(exist_ok=True)
        for name in ['trajectory.npz','metrics.json']:shutil.copy2(source/f'reset{reset}'/name,p/name)
        result=probe(target,reset,[0,1024],force=True)
        rows.append(dict(label=label,repeat=0,method='perturb_68_fixed600',seconds=result['seconds']))
    df=pd.DataFrame(rows);df.to_csv(H/'timings.csv',index=False)
    print(df.groupby(['label','method']).seconds.median().to_string())

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run()
