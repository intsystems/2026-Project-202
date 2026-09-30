"""Visual check only; never used to select policies or metrics."""
from pathlib import Path
import argparse,json
import numpy as np
import gymnasium as gym
import torch
from threadpoolctl import threadpool_limits
from PIL import Image,ImageDraw
from motion import H,Actor

def render(seed):
    pair=json.loads((H/f'seed{seed}/pair.json').read_text());reset=pair['common_resets'][0] if pair['common_resets'] else 31001
    for stage in ['early','late']:
        step=pair[stage]
        if step is None:continue
        actor=Actor(H/f'seed{seed}'/f'step{step:07d}')
        env=gym.make('Walker2d-v5',render_mode='rgb_array',width=600,height=400).unwrapped
        obs,_=env.reset(seed=reset);frames=[]
        for t in range(1250):
            if t%5==0:
                im=Image.fromarray(env.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,600,27),fill='white');draw.text((8,7),f'{stage}: seed {seed}, step {step}, t={t*.008:.2f}s',fill='black')
                frames.append(im)
            obs,_,terminated,truncated,_=env.step(actor(obs))
            if terminated or truncated:
                im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.text((8,7),f'Unhealthy termination: {t+1} steps',fill='red');frames.extend([im]*25);break
        env.close();frames[0].save(H/f'preview_{stage}.gif',save_all=True,append_images=frames[1:],duration=40,loop=0)
        print('PREVIEW',stage,seed,step,len(frames),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=200);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):render(a.seed)
