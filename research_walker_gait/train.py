from pathlib import Path
import argparse,hashlib,json,platform,time
import numpy as np
import torch
import gymnasium as gym
import mujoco
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
INTERVAL=131072

class Save(BaseCallback):
    def __init__(self,out):
        super().__init__();self.out=out;self.started=time.perf_counter();self.previous=0
    def _on_step(self):
        if self.num_timesteps%INTERVAL==0 and self.num_timesteps!=self.previous:
            self.previous=self.num_timesteps;save(self.model,self.training_env,self.out,self.num_timesteps)
            rewards=[ep['r'] for ep in self.model.ep_info_buffer]
            row=dict(step=self.num_timesteps,seconds=time.perf_counter()-self.started,
                recent_return=float(np.mean(rewards)) if rewards else None)
            with (self.out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('CHECKPOINT',json.dumps(row),flush=True)
        return True

def save(model,env,out,step):
    d=out/f'step{step:07d}';d.mkdir(parents=True,exist_ok=True)
    model.save(d/'policy.zip');env.save(d/'normalize.pkl')

def train(seed,steps,resume):
    out=H/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/f'train_{steps}.json').exists():raise RuntimeError('Completed horizon already exists.')
    raw=make_vec_env('Walker2d-v5',n_envs=8,seed=seed,monitor_dir=str(out/'monitor'))
    if resume:
        d=out/f'step{resume:07d}';env=VecNormalize.load(d/'normalize.pkl',raw);env.training=True
        model=PPO.load(d/'policy.zip',env=env,device='cpu');model.set_random_seed(seed+resume)
    else:
        env=VecNormalize(raw,norm_obs=True,norm_reward=True,clip_obs=10.,clip_reward=10.,gamma=.99)
        model=PPO('MlpPolicy',env,learning_rate=.0003,n_steps=256,batch_size=64,n_epochs=10,
            gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64])),seed=seed,device='cpu',verbose=0)
        save(model,env,out,0)
    start=time.perf_counter()
    model.learn(total_timesteps=steps-resume,callback=Save(out),reset_num_timesteps=not bool(resume))
    # Checkpoints in callbacks precede a PPO update at a rollout boundary. Overwrite
    # the FINAL checkpoint after the last update so it represents the trained horizon.
    save(model,env,out,int(model.num_timesteps))
    meta=dict(seed=seed,steps=int(model.num_timesteps),resume=resume,seconds=time.perf_counter()-start,
        versions=dict(gymnasium=gym.__version__,mujoco=mujoco.__version__,stable_baselines3=sb3.__version__,torch=torch.__version__),
        python=platform.python_version(),threads=1,device='cpu',
        protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest(),
        checkpoint_note='Intermediate callback saves before corresponding PPO update; final saved after update.')
    (out/f'train_{steps}.json').write_text(json.dumps(meta,indent=2));env.close();print('COMPLETE',seed,steps,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=2097152);p.add_argument('--resume',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):train(a.seed,a.steps,a.resume)
