from pathlib import Path
import argparse,json,time,hashlib,shutil
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from threadpoolctl import threadpool_limits
from smooth_ppo import SmoothPPO
H=Path(__file__).resolve().parent;ANCHOR=H.parent/'research_walker_repair'/'anchor';FINAL=1048576

def save(model,env,out,step):
    p=out/f'step{step:07d}';p.mkdir(exist_ok=True);model.save(p/'policy.zip');env.save(p/'normalize.pkl')

class Save(BaseCallback):
    def __init__(self,out):super().__init__();self.out=out;self.previous=0;self.started=time.perf_counter()
    def _on_step(self):return True
    def record(self):
        save(self.model,self.training_env,self.out,self.num_timesteps)
        r=dict(step=self.num_timesteps,seconds=time.perf_counter()-self.started,
            recent_return=float(np.mean([e['r'] for e in self.model.ep_info_buffer])),
            **{k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)})
        with (self.out/'progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        self.previous=self.num_timesteps;print('CHECKPOINT',json.dumps(r),flush=True)
    def _on_rollout_start(self):
        if self.num_timesteps and self.num_timesteps%131072==0 and self.previous!=self.num_timesteps:self.record()
    def _on_training_end(self):self.record()

def train(seed,coef,steps=FINAL,label=None):
    label=label or f'seed{seed}_lambda{coef:g}';out=H/label;out.mkdir(exist_ok=True)
    assert not (out/'train.json').exists(),'Completed run exists'
    raw=make_vec_env('Walker2d-v5',n_envs=8,seed=seed,env_kwargs=dict(max_episode_steps=5000),monitor_dir=str(out/'monitor'))
    env=VecNormalize.load(ANCHOR/'normalize.pkl',raw);env.training=False;env.norm_reward=True
    model=SmoothPPO.load(ANCHOR/'policy.zip',env=env,device='cpu',n_steps=512,batch_size=256,n_epochs=5,
        learning_rate=lambda p:3e-5*(.1+.9*p),clip_range=.1,target_kl=.01)
    model.smooth_coef=coef;model.pair_rng=np.random.default_rng(seed+100000)
    model.set_random_seed(seed);model.num_timesteps=0;model.ep_info_buffer.clear()
    before=np.concatenate([p.detach().numpy().ravel() for p in model.policy.parameters()]);save(model,env,out,0)
    started=time.perf_counter();model.learn(total_timesteps=steps,callback=Save(out),reset_num_timesteps=True)
    after=np.concatenate([p.detach().numpy().ravel() for p in model.policy.parameters()])
    result=dict(seed=seed,coef=coef,steps=steps,seconds=time.perf_counter()-started,parameter_delta=float(np.linalg.norm(after-before)),
        protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest())
    (out/'train.json').write_text(json.dumps(result,indent=2));env.close();print('COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--coef',type=float,required=True);p.add_argument('--steps',type=int,default=FINAL);p.add_argument('--label');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):train(a.seed,a.coef,a.steps,a.label)
