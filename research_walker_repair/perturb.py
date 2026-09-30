from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import mujoco
import torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,make_env,reduced,SCALE,STATE,BURN,ANCHORS
EPS=1e-3

def phase_dist(points,reference,index,radius):
    points=np.atleast_2d(points)
    lo=max(0,index-radius);hi=min(len(reference)-1,index+radius)
    a=reference[lo:hi];v=reference[lo+1:hi+1]-a
    diff=points[:,None,:]-a[None]
    projection=np.clip(np.sum(diff*v[None],axis=-1)/np.maximum(np.sum(v*v,axis=-1),1e-30)[None],0,1)
    return np.sqrt(np.min(np.sum((diff-projection[:,:,None]*v[None])**2,axis=-1),axis=1))

def restore(env,state):
    # Integration state contains warm-start variables. mj_step recomputes derived
    # kinematics; avoid an extra forward solve before the first restored step.
    mujoco.mj_setState(env.model,env.data,state,STATE)

def zero_audit(actor,data,anchor,period,batch_size=1):
    env=make_env();env.reset(seed=0);i=list(data['anchor_ids']).index(anchor);restore(env,data['integration_states'][i])
    error=0.
    for t in range(4*period+1):
        expected=reduced(data['qpos'][anchor+t],data['qvel'][anchor+t])
        actual=reduced(env.data.qpos,env.data.qvel)
        error=max(error,float(np.max(np.abs(actual-expected))))
        if t<4*period:
            observation=env._get_obs()
            action=actor(observation) if batch_size==1 else actor(np.repeat(observation[None],batch_size,axis=0))[0]
            env.step(action)
    env.close();assert error<(1e-8 if batch_size==1 else .1*EPS),f'Zero perturbation replay drift {error}, batch={batch_size}'
    return error

def probe(checkpoint,reset,anchors,force=False,tag='perturb'):
    out=checkpoint/f'reset{reset}';name=tag+'_'+str(len(anchors))
    if (out/(name+'.json')).exists() and not force:
        cached=json.loads((out/(name+'.json')).read_text())
        assert cached.get('inference_mode')=='single_observation_as_nominal','Archive exploratory batched result and rerun with --force.'
        return cached
    metrics=json.loads((out/'metrics.json').read_text());assert metrics['eligible']
    data=np.load(out/'trajectory.npz');actor=Actor(checkpoint);period=metrics['period'];horizon=4*period
    reference=reduced(np.concatenate([data['burn_qpos'],data['qpos']]),np.concatenate([data['burn_qvel'],data['qvel']]))
    zero_error=zero_audit(actor,data,anchors[0],period)
    specs=[(j,sign) for j in range(17) for sign in [-1,1]];envs=[make_env() for _ in specs]
    for env in envs:env.reset(seed=0)
    rows=[];curves=[];started=time.perf_counter();cpu_started=time.process_time();feedback_changes=[]
    for anchor in anchors:
        state=data['integration_states'][list(data['anchor_ids']).index(anchor)]
        original=reference[BURN+anchor];nominal_obs=None
        for env,(j,sign) in zip(envs,specs):
            restore(env,state)
            if nominal_obs is None:nominal_obs=env._get_obs().copy()
            if j<8:env.data.qpos[j+1]+=sign*EPS*SCALE[j]
            else:env.data.qvel[j-8]+=sign*EPS*SCALE[j]
        x=np.array([reduced(e.data.qpos,e.data.qvel) for e in envs])
        np.testing.assert_allclose(np.linalg.norm(x-original,axis=1),EPS,atol=1e-12)
        dist=np.full((len(specs),horizon+1),np.nan);dist[:,0]=phase_dist(x,reference,BURN+anchor,period//2)
        alive=np.ones(len(specs),dtype=bool);fall_step=np.full(len(specs),-1)
        initial_actions=np.array([actor(e._get_obs()) for e in envs])
        feedback_changes.append(float(np.max(np.abs(initial_actions-actor(nominal_obs)))))
        for t in range(1,horizon+1):
            active=np.flatnonzero(alive)
            if not len(active):break
            actions=np.array([actor(envs[i]._get_obs()) for i in active])
            states=[]
            for i,action in zip(active,actions):
                _,_,terminated,truncated,_=envs[i].step(action)
                states.append(reduced(envs[i].data.qpos,envs[i].data.qvel))
                if terminated or truncated:alive[i]=False;fall_step[i]=t
            dist[active,t]=phase_dist(np.array(states),reference,BURN+anchor+t,period//2)
        for i,(j,sign) in enumerate(specs):
            tangent=bool(dist[i,0]<.1*EPS);fell=bool(fall_step[i]>=0)
            amplification=float(np.median(dist[i,-period:])/dist[i,0]) if not tangent and not fell else None
            rows.append(dict(anchor=anchor,coordinate=j,sign=sign,initial_distance=float(dist[i,0]),
                nearly_tangent=tangent,fell=fell,fall_step=int(fall_step[i]),amplification=amplification))
        curves.append(dist)
    elapsed=time.perf_counter()-started;cpu_elapsed=time.process_time()-cpu_started
    for e in envs:e.close()
    frame=pd.DataFrame(rows);valid=frame[frame.amplification.notna()]
    result=dict(reset=reset,anchors=anchors,period=period,horizon=horizon,epsilon=EPS,probes=len(rows),
        seconds=elapsed,cpu_seconds=cpu_elapsed,zero_replay_max_error=zero_error,inference_mode='single_observation_as_nominal',initial_action_response_max=max(feedback_changes),
        falls=int(frame.fell.sum()),nearly_tangent=int(frame.nearly_tangent.sum()),assessable=len(valid),
        median_amplification=float(valid.amplification.median()) if len(valid) else None,
        fraction_amplifying_above2=float((valid.amplification>2).mean()) if len(valid) else None)
    frame.to_csv(out/(name+'.csv'),index=False)
    np.savez_compressed(out/(name+'_curves.npz'),distances=np.array(curves),anchors=np.array(anchors),period=period)
    (out/(name+'.json')).write_text(json.dumps(result,indent=2));print('PERTURB',checkpoint.name,reset,json.dumps(result),flush=True)
    return result

def selected(seed):
    selected=json.loads((H/'selection.json').read_text());pair=json.loads((H/f'seed{seed}/pair.json').read_text())
    if not pair['usable']:print('No usable pair',seed);return
    for step in [pair['early'],pair['late']]:
        for reset in pair['common_resets']:probe(H/f'seed{seed}'/f'step{step:07d}',reset,selected['anchors'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=200);p.add_argument('--step',type=int);p.add_argument('--reset',type=int,default=31001);p.add_argument('--anchors',type=int,nargs='+',default=ANCHORS);p.add_argument('--force',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        if a.step is not None:probe(H/f'seed{a.seed}'/f'step{a.step:07d}',a.reset,a.anchors,force=a.force)
        else:selected(a.seed)
