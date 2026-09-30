from pathlib import Path
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
H=Path(__file__).resolve().parent
SCALE=np.r_[np.ones(8),np.full(9,5.)];RHO=1/(1+np.sqrt(2)/1000)

def reduced(qp,qv):return np.concatenate([qp[...,1:],qv],axis=-1)/SCALE

class PhaseWalker(gym.Wrapper):
    def __init__(self,coef=0):
        super().__init__(gym.make('Walker2d-v5').unwrapped);self.coef=coef
        d=np.load(H/'reference.npz');self.points=d['points'];self.sigma2=float(d['sigma2']);self.P=len(self.points);self.phase=0.
        self.observation_space=Box(-np.inf,np.inf,(19,),dtype=np.float64)
    def target(self,phase=None):
        phase=self.phase if phase is None else phase;u=phase%self.P;j=int(u);frac=u-j
        x=(1-frac)*self.points[j]+frac*self.points[(j+1)%self.P];x=x.copy();x[8:]*=RHO;return x
    def observation(self):
        return np.r_[self.unwrapped._get_obs(),np.sin(2*np.pi*self.phase/self.P),np.cos(2*np.pi*self.phase/self.P)]
    def reset(self,seed=None,options=None):
        _,info=self.env.reset(seed=seed,options=options);rng=self.unwrapped.np_random;self.phase=float(rng.uniform(0,self.P));x=self.target()*SCALE
        qp=np.r_[0.,x[:8]];qv=x[8:];qp[1:]+=rng.uniform(-.002,.002,8);qv=qv+rng.uniform(-.01,.01,9)
        self.unwrapped.set_state(qp,qv);return self.observation(),info
    def step(self,action):
        _,reward,term,trunc,info=self.env.step(action);self.phase=(self.phase+RHO)%self.P
        x=reduced(self.unwrapped.data.qpos,self.unwrapped.data.qvel);e2=float(np.sum((x-self.target())**2));penalty=float(1-np.exp(-e2/(2*self.sigma2)))
        info.update(original_reward=float(reward),tracking_error2=e2,orbit_penalty=penalty,phase=self.phase)
        return self.observation(),reward-self.coef*penalty,term,trunc,info
