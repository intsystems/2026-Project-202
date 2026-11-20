from pathlib import Path
import sys,json,time,pickle,hashlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from generator_screen import H,R,ARMS,evaluate,forecast
from campaign import feat,MODELS
sys.path.insert(0,str(R/'research_generator'))
from generator import target_spec,train_multi,init_multi,fidelity_multi
TASKS={'T1':('torus',1),'T2':('torus',2),'T3':('torus',3),'T4':('torus',4),'H2':('harmonic',2),'H4':('harmonic',4),'M4':('mixed',4)}
SCENARIOS=[(1,512),(0,512),(2,32)]
def one(seed):
 folder=H/f'fresh_seed{seed}';folder.mkdir(exist_ok=True);rows=[];info=[]
 with (H/'g_frozen.pkl').open('rb') as f:frozen=pickle.load(f)
 for arm in ARMS:
  file=folder/f'{arm}.npz';meta=folder/f'{arm}.json'
  if file.exists():
   obs=np.load(file)['obs'];record=json.loads(meta.read_text())
  else:
   started=time.perf_counter()
   if arm=='chaos':
    j,u,x=init_multi(seed,256,1,gain=1.5);w=np.zeros((256,1));spec=None
   else:
    spec=target_spec(*TASKS[arm],seed);j,u,w,x,_=train_multi(seed,256,spec,40000,gain=1.2)
   training=time.perf_counter()-started;a=j+u@w.T
   for _ in range(5000):x=.9*x+.1*(a@np.tanh(x))
   obs=np.empty((8192,3));zs=np.empty((8192,w.shape[1]))
   for t in range(8192):
    r=np.tanh(x);obs[t]=r[:3];zs[t]=r@w;x=.9*x+.1*(a@r)
   fidelity=fidelity_multi(zs,spec)[0] if spec else None
   record=dict(seed=seed,arm=arm,fidelity=fidelity,training_seconds=training,n=256,steps=40000 if spec else 0)
   meta.write_text(json.dumps(record,indent=2));np.savez_compressed(file,obs=obs,zs=zs)
  info.append(record)
  for neuron,h in SCENARIOS:
   x=obs[4096:6144,neuron];y=obs[6144:6144+h,neuron];v=x.var();row=dict(seed=seed,arm=arm,neuron=neuron,horizon=h,**feat(x));span=max(128,h)
   for m in MODELS:
    tic=time.perf_counter();pred=forecast(x,h,m);row['time_'+m]=time.perf_counter()-tic;row['loss_'+m]=float(np.mean((pred-y)**2)/v)
    tic=time.perf_counter();row['val_'+m]=float(np.mean([np.mean((forecast(x[:o],h,m)-x[o:o+h])**2)/v for o in range(len(x)-span,len(x)-h+1,h)]));row['vtime_'+m]=time.perf_counter()-tic
   rows.append(row)
  print('fresh',seed,arm,record['fidelity'],flush=True)
 d=pd.DataFrame(rows);d.to_csv(folder/'features.csv',index=False)
 decisions=[]
 for n,h in SCENARIOS:
  models,fixed=frozen[f'{n}_{h}'];decisions.append(evaluate(d[(d.neuron==n)&(d.horizon==h)],models,fixed))
 pd.concat(decisions).to_csv(folder/'decisions.csv',index=False)
 pd.DataFrame(info).to_csv(folder/'training.csv',index=False)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  for seed in map(int,sys.argv[1:]):one(seed)
