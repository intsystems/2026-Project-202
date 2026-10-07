from pathlib import Path
import argparse,json,time,sys,hashlib,platform
import numpy as np,pandas as pd,torch,gymnasium as gym
from stable_baselines3 import PPO
import pickle
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG=EstimatorConfig(max_E=20,tau=8,k_neighbors=20,theiler=312,theiler_cap=312)
COEFS=[0,.25,1,4];BURN=256;N=2048

def features(actions):
    x=np.linalg.norm(actions,axis=1);v=np.var(x)
    result={};tim={}
    functions={
      'MG':lambda:estimate(x,CFG,seed=123),
      'entropy':lambda:entropy(x),
      'increments':lambda:np.mean(np.diff(x)**2)/(2*v),
      'recurrence':lambda:min(np.mean((x[l:]-x[:-l])**2)/(2*v) for l in range(20,251)),
      'J1':lambda:np.mean(np.diff(actions,axis=0)**2)}
    for name,fn in functions.items():
        start=time.perf_counter();value=fn();tim[name+'_seconds']=time.perf_counter()-start
        if name=='MG':
            result.update(MG=float(value.MG) if not value.degenerate else np.nan,degenerate=bool(value.degenerate))
        else:result[name]=float(value)
    return dict(**result,**tim)

def entropy(x):
    power=np.abs(np.fft.rfft(x-x.mean()))[1:]**2
    if power.sum()==0:return np.nan
    p=power/power.sum();p=p[p>0]
    return float(-np.sum(p*np.log(p))/np.log(len(power)))

def rollout_batch(policy,norm,configs):
    envs=[gym.make('Walker2d-v5').unwrapped for _ in configs]
    obs=np.array([e.reset(seed=c['reset'])[0] for e,c in zip(envs,configs)])
    active=np.ones(len(envs),bool);rewards=np.zeros(len(envs));steps=np.zeros(len(envs),int)
    actions=[[] for _ in envs];queues=[[] for _ in envs];last=np.zeros((len(envs),6))
    started=time.perf_counter()
    for t in range(BURN+N):
        ids=np.flatnonzero(active)
        if not len(ids):break
        with torch.no_grad():command,_=policy.predict(norm.normalize_obs(obs[ids]),deterministic=True)
        for idx,a in zip(ids,command):
            delay=configs[idx]['delay']
            if t==BURN:queues[idx]=[last[idx].copy() for _ in range(delay)]
            applied=a
            if t>=BURN and delay:
                queues[idx].append(a.copy());applied=queues[idx].pop(0)
            obs[idx],reward,done,trunc,info=envs[idx].step(applied)
            last[idx]=applied.copy();steps[idx]=t+1
            if t>=BURN:
                rewards[idx]+=reward;actions[idx].append(a.copy())
            if done or trunc:active[idx]=False
    seconds=time.perf_counter()-started
    for e in envs:e.close()
    rows=[]
    for i,c in enumerate(configs):
        a=np.array(actions[i]).reshape(-1,6)
        row=dict(**c,complete=bool(active[i]),reward=float(rewards[i]/N),steps=int(steps[i]))
        if c['split']=='nominal':
            if active[i]:row.update(features(a))
            np.savez_compressed(H/f"nominal_s{c['seed']}_l{c['coef']:g}_r{c['reset']}.npz",actions=a)
        rows.append(row)
    return rows,seconds

def run(seed):
    out=H/f'seed{seed}.json'
    if out.exists():return
    allrows=[];cost=[]
    for coef in COEFS:
        root=R/'research_walker_smooth_lambda'/f'seed{seed}_lambda{coef:g}'/'step1048576'
        model=PPO.load(root/'policy.zip',device='cpu');model.policy.set_training_mode(False)
        with (root/'normalize.pkl').open('rb') as f:norm=pickle.load(f)
        norm.training=False;norm.norm_reward=False
        configs=[dict(seed=seed,coef=coef,split='nominal',delay=0,reset=r) for r in range(81001,81004)]
        rows,sec=rollout_batch(model,norm,configs);allrows+=rows
        cost.append(dict(coef=coef,split='nominal',seconds=sec,episodes=len(configs)))
        configs=[dict(seed=seed,coef=coef,split='target',delay=d,reset=r) for d in [0,1,2,3] for r in range(82001,82011)]
        rows,sec=rollout_batch(model,norm,configs);allrows+=rows
        cost.append(dict(coef=coef,split='target',seconds=sec,episodes=len(configs)))
        print(seed,coef,'nominal',np.mean([x['reward'] for x in allrows if x['coef']==coef and x['split']=='nominal']),
              'target',np.mean([x['reward'] for x in rows if x['delay']>0]),flush=True)
    out.write_text(json.dumps(dict(seed=seed,rows=allrows,cost=cost,protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        for seed in a.seeds:run(seed)
