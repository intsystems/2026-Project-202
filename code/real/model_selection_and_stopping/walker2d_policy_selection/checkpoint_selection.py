"""Offline checkpoint selection under hidden actuator smoothing/noise."""
import argparse
from pathlib import Path
import pickle, time, sys
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
CODE = Path(__file__).resolve().parents[3]
CFG={"E":20,"tau":8,"k":20,"theiler":312,"window":2048}
STEPS=[262144,524288,786432,1048576];COEFS=[0,.25,1,4];BURN=256;N=2048
sys.path.insert(0, str(CODE))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
MGCFG=EstimatorConfig(max_E=20,tau=8,k_neighbors=20,theiler=312,theiler_cap=312)
def entropy(x):
 p=np.abs(np.fft.rfft(x-x.mean()))[1:]**2
 if p.sum()==0:return np.nan
 p=p/p.sum();return float(-np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p)))
def features(a):
 x=np.linalg.norm(a,axis=1);v=np.var(x)
 rec=float(min(np.mean((x[l:]-x[:-l])**2)/(2*v) for l in range(20,251)))
 r=estimate(x,MGCFG,seed=123)
 return dict(MG=float(r.MG) if not r.degenerate else np.nan,degenerate=bool(r.degenerate),entropy=entropy(x),increments=float(np.mean(np.diff(x)**2)/(2*v)),recurrence=rec,J1=float(np.mean(np.diff(a,axis=0)**2)))
def eval_policy(model,norm,resets,alpha,noise,features_on=False):
 import gymnasium as gym
 import torch

 rows=[];envs=[gym.make('Walker2d-v5').unwrapped for _ in resets];obs=np.array([e.reset(seed=r)[0] for e,r in zip(envs,resets)])
 active=np.ones(len(envs),bool);rew=np.zeros(len(envs));acts=[[] for _ in resets];last=np.zeros((len(resets),6))
 started=time.perf_counter()
 perturbations=[np.array([np.random.default_rng(int(r+t*1009)).normal(0,noise,size=6) for t in range(BURN+N)]) if noise else None for r in resets]
 for t in range(BURN+N):
  ids=np.flatnonzero(active)
  if not len(ids):break
  with torch.no_grad():a,_=model.predict(norm.normalize_obs(obs[ids]),deterministic=True)
  for i,u in zip(ids,a):
   applied=(1-alpha)*u+alpha*last[i]
   if noise:applied=applied+perturbations[i][t]
   obs[i],r,done,trunc,info=envs[i].step(np.clip(applied,-1,1));last[i]=applied
   if t>=BURN:rew[i]+=r;acts[i].append(u.copy())
   if done or trunc:active[i]=False
 elapsed=time.perf_counter()-started
 for e in envs:e.close()
 for r,v,a in zip(resets,rew,acts):
  row=dict(reset=int(r),reward=float(v/N),complete=bool(active[list(resets).index(r)]),alpha=alpha,noise=noise)
  row['rollout_batch_seconds']=elapsed
  row['feature_seconds']=0.
  if features_on and row['complete']:
   tic=time.perf_counter();row.update(features(np.asarray(a)));row['feature_seconds']=time.perf_counter()-tic
  rows.append(row)
 return rows
def collect(seed, checkpoint_root, output):
 from stable_baselines3 import PPO

 rows=[]
 for coef in COEFS:
  for step in STEPS:
   root=checkpoint_root/f'seed{seed}_lambda{coef:g}'/f'step{step:07d}'
   model=PPO.load(root/'policy.zip',device='cpu');model.policy.set_training_mode(False)
   with (root/'normalize.pkl').open('rb') as f:norm=pickle.load(f)
   norm.training=False;norm.norm_reward=False
   nom=eval_policy(model,norm,range(83001,83004),0,0,True)
   for n in nom:rows.append(dict(seed=seed,coef=coef,step=step,split='nominal',**n))
   for alpha in [.05,.1,.2]:
    for noise in [0,.02]:
     tar=eval_policy(model,norm,range(84001,84006),alpha,noise,False)
     for n in tar:rows.append(dict(seed=seed,coef=coef,step=step,split='target',**n))
   print('checkpoint',seed,coef,step,flush=True)
 df=pd.DataFrame(rows);df.to_csv(output/f'checkpoint_selection_s{seed}.csv',index=False)
 # Each selector chooses on nominal rows, then is scored on every hidden target condition.
 nom=df[df.split=='nominal'];target=df[df.split=='target']
 methods={'reward':('reward',False),'MG':('MG',True),'entropy':('entropy',True),'increments':('increments',True),'recurrence':('recurrence',True),'J1':('J1',True)}
 out=[]
 for method,(metric,low) in methods.items():
  q=nom[nom.complete].groupby(['coef','step'])
  allq=nom.groupby(['coef','step'])
  elig=allq.reward.mean(); count=allq.complete.sum()
  elig=elig[(elig>=.9*elig.max())&(count>=2)]
  if len(elig)==0:elig=allq.reward.mean().nlargest(1)
  if method=='reward':
   best=elig.idxmax()
  else:
   feature=q[metric].median().dropna()
   feature=feature[feature.index.isin(elig.index)]
   if len(feature)==0: best=elig.idxmax()
   else: best=feature.idxmin()
  chosen=target[(target.coef==best[0])&(target.step==best[1])]
  for (alpha,noise),g in chosen.groupby(['alpha','noise']):out.append(dict(seed=seed,method=method,coef=best[0],step=best[1],alpha=alpha,noise=noise,reward=g.reward.mean(),complete=g.complete.mean()))
 pd.DataFrame(out).to_csv(output/f'checkpoint_selection_summary_s{seed}.csv',index=False)
 print(pd.DataFrame(out).to_string(index=False))
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CODE / "runs" / "walker2d_policy_selection")
    args = parser.parse_args()
    for coef in COEFS:
        for step in STEPS:
            checkpoint = args.checkpoint_root / f"seed{args.seed}_lambda{coef:g}" / f"step{step:07d}"
            for name in ("policy.zip", "normalize.pkl"):
                if not (checkpoint / name).is_file():
                    parser.error(f"Missing checkpoint file {checkpoint / name}.")

    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        collect(args.seed, args.checkpoint_root, args.output)


if __name__ == "__main__":
    main()
