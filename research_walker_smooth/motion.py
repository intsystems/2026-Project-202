from pathlib import Path
import argparse,json,pickle,time
import numpy as np
import pandas as pd
import gymnasium as gym
import mujoco
import torch
from scipy.signal import find_peaks
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
RESETS=[31001,31002,31003];BURN=512;N=4096;ANCHORS=[0,512,1024,1536]
SCALE=np.r_[np.ones(8),np.full(9,5.)]
STATE=mujoco.mjtState.mjSTATE_INTEGRATION

class Actor:
    def __init__(self,checkpoint):
        self.model=PPO.load(checkpoint/'policy.zip',device='cpu')
        self.model.policy.set_training_mode(False)
        with (checkpoint/'normalize.pkl').open('rb') as f:self.norm=pickle.load(f)
        self.norm.training=False;self.norm.norm_reward=False
    @torch.no_grad()
    def __call__(self,obs):
        normalized=self.norm.normalize_obs(np.asarray(obs,dtype=np.float64))
        action,_=self.model.predict(normalized,deterministic=True)
        return action

def make_env():return gym.make('Walker2d-v5').unwrapped

def reduced(qpos,qvel):return np.concatenate([qpos[...,1:],qvel],axis=-1)/SCALE

def cheap(signal):
    x=np.asarray(signal);y=x-x.mean();variance=float(np.mean(y*y))
    power=np.abs(np.fft.rfft(y))[1:]**2;power/=max(power.sum(),1e-30)
    entropy=float(-np.sum(power*np.log(np.maximum(power,1e-30)))/np.log(len(power)))
    peaks,_=find_peaks(x,distance=20,prominence=.15)
    intervals=np.diff(peaks)
    errors=[float(np.mean((x[p:]-x[:-p])**2)/(2*max(variance,1e-30))) for p in range(20,251)]
    return dict(std=float(np.std(x)),entropy=entropy,autocorr=float(1-min(errors)),
        peak_count=len(peaks),period_cv=float(np.std(intervals)/np.mean(intervals)) if len(intervals) else None)

def state_reference(qpos,qvel):
    x=reduced(qpos,qvel);variance=float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    errors=np.array([np.mean(np.sum((x[p:]-x[:-p])**2,axis=1))/(2*max(variance,1e-30)) for p in range(20,251)])
    period=20+int(np.argmin(errors));knee=qpos[:,4];peaks,_=find_peaks(knee,distance=20,prominence=.15)
    sections=[]
    for j in peaks:
        denominator=knee[j-1]-2*knee[j]+knee[j+1]
        delta=float(np.clip(.5*(knee[j-1]-knee[j+1])/denominator,-.5,.5)) if denominator else 0.
        position=j+delta;lo=int(np.floor(position));fraction=position-lo
        sections.append((1-fraction)*x[lo]+fraction*x[lo+1])
    sec=np.array(sections)
    dispersion=float(np.mean(np.sum((sec-sec.mean(0))**2,axis=1))/max(variance,1e-30)) if len(sec) else None
    return dict(recurrence=float(errors.min()),section_dispersion=dispersion,period=period,
        period_boundary=period in [20,250],state_variance=variance,peak_count=len(peaks)),errors

def rollout(actor,checkpoint,reset):
    out=checkpoint/f'reset{reset}';out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    env=make_env();obs,_=env.reset(seed=reset);qpos=[];qvel=[];burn_qpos=[];burn_qvel=[];rewards=[];velocities=[];actions=[];saved=[];anchor_ids=[]
    tic=time.perf_counter();terminated=False
    for step in range(BURN+N):
        if step<BURN:burn_qpos.append(env.data.qpos.copy());burn_qvel.append(env.data.qvel.copy())
        if step>=BURN:
            qpos.append(env.data.qpos.copy());qvel.append(env.data.qvel.copy())
            if step-BURN in ANCHORS:
                state=np.empty(mujoco.mj_stateSize(env.model,STATE));mujoco.mj_getState(env.model,env.data,state,STATE)
                saved.append(state);anchor_ids.append(step-BURN)
        action=actor(obs)
        if step>=BURN:actions.append(action.copy())
        obs,reward,terminated,truncated,info=env.step(action)
        if step>=BURN:rewards.append(reward);velocities.append(info['x_velocity'])
        if terminated or truncated:break
    elapsed=time.perf_counter()-tic
    qp=np.array(qpos).reshape(-1,9);qv=np.array(qvel).reshape(-1,9)
    complete=bool(len(qp)==N and not terminated)
    std=float(qp[:,4].std()) if len(qp) else None
    peaks=len(find_peaks(qp[:,4],distance=20,prominence=.15)[0]) if len(qp)>2 else 0
    speed=float(np.mean(velocities)) if velocities else None
    eligible=bool(complete and speed>=.5 and std>=.05 and peaks>=8)
    result=dict(reset=reset,complete=complete,terminated=bool(terminated),steps=step+1,n_analysis=len(qp),
        eligible=eligible,mean_speed=speed,knee_std=std,peaks=peaks,
        mean_reward=float(np.mean(rewards)) if rewards else None,acquisition_seconds=elapsed,dt=env.dt)
    result['padded_reward']=float(np.sum(rewards)/N)
    result['J1']=float(np.mean(np.diff(actions,axis=0)**2)) if len(actions)>1 else None
    result['J2']=float(np.mean(np.diff(actions,n=2,axis=0)**2)) if len(actions)>2 else None
    if eligible:
        started=time.perf_counter();reference,errors=state_reference(qp,qv);result['state_seconds']=time.perf_counter()-started
        started=time.perf_counter();result['cheap']=cheap(qp[:,4]);result['cheap_seconds']=time.perf_counter()-started
        result.update(reference);pd.DataFrame(dict(lag=np.arange(20,251),error=errors)).to_csv(out/'recurrence.csv',index=False)
    np.savez_compressed(out/'trajectory.npz',qpos=qp,qvel=qv,reward=np.array(rewards),velocity=np.array(velocities),
        actions=np.array(actions),anchor_ids=np.array(anchor_ids),integration_states=np.array(saved),burn_qpos=np.array(burn_qpos),burn_qvel=np.array(burn_qvel))
    (out/'metrics.json').write_text(json.dumps(result,indent=2));env.close();return result

def evaluate(seed,steps=None):
    out=H/f'seed{seed}';rows=[]
    paths=sorted(out.glob('step*/policy.zip'))
    for policy in paths:
        checkpoint=policy.parent;step=int(checkpoint.name[4:])
        if steps is not None and step not in steps:continue
        actor=None
        for reset in RESETS:
            if not (checkpoint/f'reset{reset}/metrics.json').exists() and actor is None:actor=Actor(checkpoint)
            result=rollout(actor,checkpoint,reset)
            rows.append(dict(seed=seed,step=step,**{k:v for k,v in result.items() if k!='cheap'}))
        print('EVAL',seed,step,'eligible',sum(r['eligible'] for r in rows if r['step']==step),flush=True)
    all_rows=[]
    for path in sorted(out.glob('step*/reset*/metrics.json')):
        r=json.loads(path.read_text());all_rows.append(dict(seed=seed,step=int(path.parent.parent.name[4:]),**{k:v for k,v in r.items() if k!='cheap'}))
    pd.DataFrame(all_rows).to_csv(out/'evaluation.csv',index=False)

def choose_pair(seed,horizon):
    out=H/f'seed{seed}';df=pd.read_csv(out/'evaluation.csv');counts=df.groupby('step').eligible.sum()
    eligible=counts[counts>=2].index.tolist();earlier=[int(t) for t in eligible if t<horizon]
    final_valid=horizon in eligible
    early=earlier[0] if earlier else None
    common=[]
    if early is not None and final_valid:
        a=set(df[(df.step==early)&df.eligible].reset);b=set(df[(df.step==horizon)&df.eligible].reset);common=sorted(a&b)
    pair=dict(seed=seed,horizon=horizon,early=early,late=horizon,final_eligible=final_valid,
        eligible_checkpoints=[int(x) for x in eligible],common_resets=[int(x) for x in common],usable=bool(len(common)>=2))
    (out/'pair.json').write_text(json.dumps(pair,indent=2));return pair

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,nargs='*');p.add_argument('--pair',type=int);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        if a.steps is not None or not a.pair:evaluate(a.seed,a.steps)
        if a.pair:print(json.dumps(choose_pair(a.seed,a.pair),indent=2))
