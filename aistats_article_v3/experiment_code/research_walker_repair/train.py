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
INTERVAL=131072;STEPS=1048576

def save(model,env,out,step):
    d=out/f'step{step:07d}';d.mkdir(parents=True,exist_ok=True)
    model.save(d/'policy.zip');env.save(d/'normalize.pkl')

class Save(BaseCallback):
    def __init__(self,out):
        super().__init__();self.out=out;self.started=time.perf_counter();self.previous=0
    def _on_step(self):return True
    def record(self):
        save(self.model,self.training_env,self.out,self.num_timesteps)
        values=self.model.logger.name_to_value
        row=dict(step=self.num_timesteps,seconds=time.perf_counter()-self.started,
            recent_return=float(np.mean([ep['r'] for ep in self.model.ep_info_buffer])),
            log_std_mean=float(self.model.policy.log_std.mean().detach()),
            **{k.replace('train/',''):float(v) for k,v in values.items() if k.startswith('train/') and np.isscalar(v)})
        with (self.out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.previous=self.num_timesteps;print('CHECKPOINT',json.dumps(row),flush=True)
    def _on_rollout_start(self):
        if self.num_timesteps and self.num_timesteps%INTERVAL==0 and self.num_timesteps!=self.previous:self.record()
    def _on_training_end(self):self.record()

def train(seed,conservative=False):
    label=f'seed{seed}'+('_B' if conservative else '');out=H/label;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'train.json').exists(),'Completed run exists'
    anchor=H/'anchor'
    raw=make_vec_env('Walker2d-v5',n_envs=8,seed=seed,env_kwargs=dict(max_episode_steps=5000),monitor_dir=str(out/'monitor'))
    env=VecNormalize.load(anchor/'normalize.pkl',raw);env.training=False;env.norm_reward=True
    lr=3e-6 if conservative else 3e-5;clip=.05 if conservative else .1;kl=.005 if conservative else .01
    schedule=lambda progress:lr*(.1+.9*progress)
    model=PPO.load(anchor/'policy.zip',env=env,device='cpu',n_steps=512,batch_size=256,n_epochs=5,
        learning_rate=schedule,clip_range=clip,target_kl=kl)
    model.set_random_seed(seed);model.num_timesteps=0;model.ep_info_buffer.clear()
    before=np.concatenate([p.detach().numpy().ravel() for p in model.policy.parameters()]);save(model,env,out,0)
    started=time.perf_counter();model.learn(total_timesteps=STEPS,callback=Save(out),reset_num_timesteps=True)
    after=np.concatenate([p.detach().numpy().ravel() for p in model.policy.parameters()])
    metadata=dict(seed=seed,conservative=conservative,additional_steps=STEPS,original_steps=655360,
        seconds=time.perf_counter()-started,learning_rate_initial=lr,learning_rate_final=lr*.1,
        clip=clip,target_kl=kl,normalization_frozen=True,training_episode_limit=5000,
        parameter_delta_norm=float(np.linalg.norm(after-before)),parameter_relative_delta=float(np.linalg.norm(after-before)/np.linalg.norm(before)),
        protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest(),
        versions=dict(gymnasium=gym.__version__,mujoco=mujoco.__version__,stable_baselines3=sb3.__version__,torch=torch.__version__),python=platform.python_version())
    (out/'train.json').write_text(json.dumps(metadata,indent=2));env.close();print('COMPLETE',json.dumps(metadata),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=210);p.add_argument('--conservative',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):train(a.seed,a.conservative)
