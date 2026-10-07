import json,time
import numpy as np,pandas as pd,mujoco
from motion import Actor,make_env,reduced,STATE,SCALE

def restore(env,state,phase):
    mujoco.mj_setState(env.unwrapped.model,env.unwrapped.data,state,STATE);env.phase=float(phase)

def probe(cp,reset,anchors=(0,1024),eps=.001,force=False):
    root=cp/f'reset{reset}';tag=f'probe_eps{eps:g}_a{len(anchors)}'
    if (root/(tag+'.json')).exists() and not force:return json.loads((root/(tag+'.json')).read_text())
    d=np.load(root/'trajectory.npz');actor=Actor(cp);env=make_env();env.reset(seed=0);horizon=600;T=152
    errors=[];norms=[];spectra=[];jacobians=[];rows=[];started=time.perf_counter();zero=0.
    for anchor in anchors:
        i=list(d['anchor_ids']).index(anchor);state=d['integration_states'][i];phase=d['anchor_phases'][i]
        nominal=reduced(d['qpos'][anchor:anchor+horizon+1],d['qvel'][anchor:anchor+horizon+1])
        restore(env,state,phase)
        for t in range(horizon+1):
            zero=max(zero,float(np.max(abs(reduced(env.unwrapped.data.qpos,env.unwrapped.data.qvel)-nominal[t]))))
            if t<horizon:env.step(actor(env.observation()))
        assert zero<1e-8,f'zero replay {zero}'
        values=np.full((34,horizon+1,17),np.nan);falls=[]
        for j in range(17):
            for n,sign in enumerate([-1,1]):
                k=2*j+n;restore(env,state,phase)
                if j<8:env.unwrapped.data.qpos[j+1]+=sign*eps*SCALE[j]
                else:env.unwrapped.data.qvel[j-8]+=sign*eps*SCALE[j]
                values[k,0]=reduced(env.unwrapped.data.qpos,env.unwrapped.data.qvel)-nominal[0];fall=-1
                np.testing.assert_allclose(np.linalg.norm(values[k,0]),eps,atol=1e-12)
                for t in range(1,horizon+1):
                    _,_,terminated,truncated,_=env.step(actor(env.observation()))
                    values[k,t]=reduced(env.unwrapped.data.qpos,env.unwrapped.data.qvel)-nominal[t]
                    if terminated or truncated:fall=t;break
                dist=np.linalg.norm(values[k],axis=1);a=float(np.median(dist[-150:])/eps) if fall<0 else None
                rows.append(dict(anchor=anchor,coordinate=j,sign=sign,fall_step=fall,amplification=a));falls.append(fall)
        J=np.column_stack([(values[2*j+1,T]-values[2*j,T])/(2*eps) for j in range(17)])
        spec=np.linalg.svd(J,compute_uv=False) if np.isfinite(J).all() else np.full(17,np.nan)
        eig=np.linalg.eigvals(J) if np.isfinite(J).all() else np.full(17,np.nan,dtype=complex)
        jacobians.append(J);spectra.append(spec);norms.append(np.linalg.norm(values,axis=-1));errors.append(values[:,T])
    elapsed=time.perf_counter()-started;env.close();frame=pd.DataFrame(rows)
    result=dict(reset=reset,anchors=list(anchors),epsilon=eps,horizon=horizon,terminal_steps=150,one_cycle_steps=T,probes=len(rows),zero_replay_error=zero,seconds=elapsed,falls=int((frame.fall_step>=0).sum()),median_amplification=float(frame.amplification.median()),spectral_radius=[float(np.max(abs(np.linalg.eigvals(j)))) if np.isfinite(j).all() else None for j in jacobians])
    frame.to_csv(root/(tag+'.csv'),index=False);np.savez_compressed(root/(tag+'.npz'),norms=np.array(norms),jacobians=np.array(jacobians),singular_values=np.array(spectra),cycle_differences=np.array(errors))
    (root/(tag+'.json')).write_text(json.dumps(result,indent=2));print('PROBE',cp.parent.name,json.dumps(result),flush=True);return result
