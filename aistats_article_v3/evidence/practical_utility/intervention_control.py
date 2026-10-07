from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from run import H,SOURCE,STEPS,TOTAL,branch
FEATURES=['MG','entropy','increments','slope','level','std','lag1']
def choose(f,rule):
 if rule['kind']=='fixed':return rule['step']
 g=f[f.obs==rule['obs']].sort_values('step');x=g[rule['feature']]
 valid=np.isfinite(x);trigger=valid&((x<=rule['threshold']) if rule['sign']==1 else(x>rule['threshold']))
 return int(g.loc[trigger,'step'].iloc[0]) if trigger.any() else TOTAL
def frames(seed,noise):
 folder=H/f'seed{seed}_noise{noise:g}';return pd.read_csv(folder/'features.csv'),pd.read_csv(folder/'metrics.csv').set_index('step')
def outcome(seed,noise,action,step):
 f,m=frames(seed,noise)
 if action=='stop':return dict(step=step,**m.loc[step].to_dict(),mac_ratio=step/TOTAL)
 if step==TOTAL:return dict(step=step,**m.loc[TOTAL].to_dict(),mac_ratio=1.)
 p=H/f'seed{seed}_noise0'/(action+'_'+str(step)+'.json');return json.loads(p.read_text())
def objective(seed,noise,action,rule):
 f,_=frames(seed,noise);step=choose(f,rule);r=outcome(seed,noise,action,step);return r['val_acc']-.02*r['mac_ratio']
def freeze():
 saved={}
 for action,noise in [('stop',.4),('stop',.6),('freeze',0),('prune',0)]:
  key=f'{action}_{noise:g}';rules={}
  for method in FEATURES+['fixed']:
   options=[]
   if method=='fixed':options=[dict(kind='fixed',step=step) for step in STEPS+[TOTAL]]
   else:
    for obs in ['norm','probe']:
     vals=np.concatenate([frames(s,noise)[0].query('obs==@obs')[method].to_numpy() for s in [0,1,2]]);vals=vals[np.isfinite(vals)]
     for sign in [1,-1]:
      for t in np.unique(np.r_[-np.inf,np.quantile(vals,np.linspace(0,1,11)),np.inf]):options.append(dict(kind='feature',feature=method,obs=obs,sign=sign,threshold=float(t)))
   best=max(options,key=lambda rule:np.mean([objective(s,noise,action,rule) for s in [0,1,2]]));rules[method]=best
  rules['never']=dict(kind='fixed',step=TOTAL);saved[key]=rules
 (H/'frozen_rules.json').write_text(json.dumps(saved,indent=2));(H/'freeze_manifest.json').write_text(json.dumps(dict(protocol=hashlib.sha256((SOURCE/'PROTOCOL.md').read_bytes()).hexdigest(),rules=hashlib.sha256((H/'frozen_rules.json').read_bytes()).hexdigest(),pilot_seeds=[0,1,2],confirmation_seeds=list(range(10,18))),indent=2))
 print(json.dumps(saved,indent=2))
def evaluate(seeds):
 rules=json.loads((H/'frozen_rules.json').read_text());rows=[]
 for key,methods in rules.items():
  action,noise=key.split('_');noise=float(noise)
  for seed in seeds:
   f,m=frames(seed,noise)
   for method,rule in methods.items():
    step=choose(f,rule)
    if action!='stop' and step!=TOTAL:branch(seed,action,step)
    r=outcome(seed,noise,action,step);rows.append(dict(seed=seed,noise=noise,action=action,method=method,step=step,test_acc=r['test_acc'],val_acc=r['val_acc'],mac_ratio=r['mac_ratio']))
   if action=='stop':
    step=int(m.val_acc.idxmax());rows.append(dict(seed=seed,noise=noise,action=action,method='clean_validation_best',step=step,test_acc=m.loc[step,'test_acc'],val_acc=m.loc[step,'val_acc'],mac_ratio=step/TOTAL))
 d=pd.DataFrame(rows);file=H/f'decisions_{min(seeds)}_{max(seeds)}.csv';d.to_csv(file,index=False);print(d.groupby(['action','noise','method'])[['test_acc','step','mac_ratio']].mean().round(4).to_string())
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 with threadpool_limits(limits=1):
  if '--freeze' in sys.argv:freeze()
  else:evaluate(list(map(int,sys.argv[1:])))
