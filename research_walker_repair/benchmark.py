import json,time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,make_env,state_reference,cheap,BURN,N
from features import estimate,EstimatorConfig
from perturb import probe

def acquire(actor,reset):
    env=make_env();obs,_=env.reset(seed=reset);start=time.perf_counter();qp=[];qv=[]
    for t in range(BURN+N):
        if t>=BURN:qp.append(env.data.qpos.copy());qv.append(env.data.qvel.copy())
        obs,_,terminated,truncated,_=env.step(actor(obs))
        assert not terminated and not truncated,'Previously eligible rollout now failed'
    elapsed=time.perf_counter()-start;env.close();return elapsed,np.array(qp),np.array(qv)

def main():
    selected=json.loads((H/'selection.json').read_text());labels=[f'seed{s}'+('_B' if selected['conservative'] else '') for s in selected['confirmation_seeds']]
    pairs=[json.loads((H/label/'pair.json').read_text()) for label in labels]
    usable=[p for p in pairs if p['usable']]
    if not usable:
        (H/'benchmark.json').write_text(json.dumps(dict(available=False,reason='No usable confirmation pair'),indent=2));return
    pair=usable[0];label=pair['label'];seed=label;reset=pair['common_resets'][0];rows=[];results=[];tau=selected['tau']
    config=EstimatorConfig(window=2048,max_E=20,tau=tau,k_neighbors=20,theiler=39*tau,theiler_cap=39*tau)
    for stage in ['early','late']:
        checkpoint=H/label/f"step{pair[stage]:07d}";root=checkpoint/f'reset{reset}'
        data=np.load(root/'trajectory.npz');qpos=data['qpos'];qvel=data['qvel'];x=qpos[:,4];actor=Actor(checkpoint)
        def mg():return [estimate(x[end-2048:end],config,seed=123).MG for end in range(2048,4097,512)]
        functions={'MG_five_windows':mg,'full_state_regularities':lambda:state_reference(qpos,qvel),'cheap_scalar_all':lambda:cheap(x)}
        for f in functions.values():f()
        for rep in range(9):
            for name in np.random.default_rng(rep+832).permutation(list(functions)):
                started=time.perf_counter();functions[name]();elapsed=time.perf_counter()-started
                rows.append(dict(seed=seed,stage=stage,operation=name,repeat=rep,seconds=elapsed))
        for rep in range(3):
            elapsed,qp,qv=acquire(actor,reset)
            np.testing.assert_array_equal(qp,qpos);np.testing.assert_array_equal(qv,qvel)
            rows.append(dict(seed=seed,stage=stage,operation='common_motion_acquisition',repeat=rep,seconds=elapsed))
        perturb=probe(checkpoint,reset,selected['anchors'],force=True,tag='benchmark_perturb')
        rows.append(dict(seed=seed,stage=stage,operation='closed_loop_perturbations',repeat=0,seconds=perturb['seconds']))
        results.append(dict(stage=stage,period=perturb['period'],probes=perturb['probes'],horizon=perturb['horizon']))
    df=pd.DataFrame(rows);df.to_csv(H/'timings.csv',index=False)
    summary=[]
    for (stage,op),part in df.groupby(['stage','operation']):
        summary.append(dict(stage=stage,operation=op,median=float(part.seconds.median()),min=float(part.seconds.min()),max=float(part.seconds.max()),repeats=len(part)))
    result=dict(available=True,seed=seed,reset=reset,operations=summary,perturbation_work=results,threads=1,
        caveat='Serial warmed CPU component timings. Same common nominal acquisition; perturbations measure a different property. One full perturbation repetition per trace, nine cheap-operation repetitions, three trajectory repetitions.')
    (H/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):main()
