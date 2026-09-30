from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,torch,gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from threadpoolctl import threadpool_limits
from environment import PhaseWalker
from motion import reduced
H=Path(__file__).resolve().parent;ANCHOR=H/'anchor';FINAL=1048576

def save(model,env,out,step):
    cp=out/f'step{step:07d}';cp.mkdir(exist_ok=True);model.save(cp/'policy.zip');env.save(cp/'normalize.pkl')

class Save(BaseCallback):
    def __init__(self,out):super().__init__();self.out=out;self.previous=0;self.started=time.perf_counter();self.penalties=[]
    def _on_step(self):
        self.penalties.extend(i['orbit_penalty'] for i in self.locals['infos']);return True
    def record(self):
        save(self.model,self.training_env,self.out,self.num_timesteps)
        r=dict(step=self.num_timesteps,seconds=time.perf_counter()-self.started,mean_orbit_penalty=float(np.mean(self.penalties)),recent_training_return=float(np.mean([e['r'] for e in self.model.ep_info_buffer])),**{k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)})
        self.penalties=[]
        with (self.out/'progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        self.previous=self.num_timesteps;print('CHECKPOINT',json.dumps(r),flush=True)
    def _on_rollout_start(self):
        if self.num_timesteps and self.num_timesteps%131072==0 and self.previous!=self.num_timesteps:self.record()
    def _on_training_end(self):self.record()

def train(seed,coef):
    out=H/f'seed{seed}_lambda{coef:g}';out.mkdir(exist_ok=True);assert not (out/'train.json').exists()
    raw=make_vec_env(lambda: gym.wrappers.TimeLimit(PhaseWalker(coef=coef),max_episode_steps=5000),n_envs=8,seed=seed,monitor_dir=str(out/'monitor'))
    env=VecNormalize.load(ANCHOR/'normalize.pkl',raw);env.training=False;env.norm_reward=True
    model=PPO.load(ANCHOR/'policy.zip',env=env,device='cpu',n_steps=512,batch_size=256,n_epochs=5,learning_rate=lambda p:3e-5*(.1+.9*p),clip_range=.1,target_kl=.01)
    model.set_random_seed(seed);model.num_timesteps=0
    if model.ep_info_buffer is not None:model.ep_info_buffer.clear()
    save(model,env,out,0)
    before=np.concatenate([p.detach().numpy().ravel() for p in model.policy.parameters()]);started=time.perf_counter()
    model.learn(total_timesteps=FINAL,callback=Save(out),reset_num_timesteps=True)
    after=np.concatenate([p.detach().numpy().ravel() for p in model.policy.parameters()])
    result=dict(seed=seed,coef=coef,steps=FINAL,seconds=time.perf_counter()-started,parameter_delta=float(np.linalg.norm(after-before)),protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((H/'reference.npz').read_bytes()).hexdigest())
    (out/'train.json').write_text(json.dumps(result,indent=2));env.close();print('COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--coef',type=float,required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):train(a.seed,a.coef)
