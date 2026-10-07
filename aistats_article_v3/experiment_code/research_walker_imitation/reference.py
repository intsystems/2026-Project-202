from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from motion import reduced,state_reference
H=Path(__file__).resolve().parent

class Orbit:
    def __init__(self):
        d=np.load(H/'reference.npz');self.points=d['points'];self.sigma2=float(d['sigma2']);self.P=len(self.points)
        self.tree=cKDTree(self.points);self.index=self.P//4;self.origin=self.points[self.index]
        tangent=self.points[(self.index+1)%self.P]-self.points[(self.index-1)%self.P]
        self.normal=tangent/np.linalg.norm(tangent)
    def distance(self,x):return self.tree.query(x,k=1)
    def metrics(self,x):
        distances,phase=self.distance(x);g=(x-self.origin)@self.normal
        candidates=np.flatnonzero((g[:-1]<=0)&(g[1:]>0));sections=[];times=[];last=-1e9
        for t in candidates:
            alpha=float(-g[t]/(g[t+1]-g[t]));point=x[t]+alpha*(x[t+1]-x[t]);idx=int(self.distance(point)[1])
            delta=abs(idx-self.index);circular=min(delta,self.P-delta)
            if circular>self.P/6 or t+alpha-last<self.P//2:continue
            sections.append(point);times.append(float(t+alpha));last=t+alpha
        sec=np.array(sections).reshape(-1,17);var=float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
        disp=float(np.mean(np.sum((sec-sec.mean(0))**2,axis=1))/max(var,1e-30)) if len(sec)>=2 else None
        result=dict(D_section=disp,section_count=len(sec),section_eligible=len(sec)>=8,orbit_distance2=float(np.mean(distances**2)),orbit_loss=float(np.mean(1-np.exp(-distances**2/(2*self.sigma2)))))
        return result,sec,np.array(times)

def prepare():
    source=H.parent/'research_walker_repair/seed210/step0000000/reset41001/trajectory.npz';d=np.load(source)
    r,_=state_reference(d['qpos'],d['qvel']);p=r['period'];x=reduced(d['qpos'],d['qvel'])
    closure=np.sum((x[p:2048]-x[:2048-p])**2,axis=1);start=int(closure.argmin());points=x[start:start+p]
    ds,_=cKDTree(points).query(x,k=1);sigma2=max(1e-6,float(np.median(ds**2)))
    np.savez_compressed(H/'reference.npz',points=points,sigma2=sigma2,source_qpos=d['qpos'],source_qvel=d['qvel'])
    meta=dict(source=str(source.relative_to(H.parent)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),period=p,start=start,closure_norm=float(np.sqrt(closure[start])),sigma2=sigma2,dt=.008,calibration_reward=json.loads(source.with_name('metrics.json').read_text())['mean_reward'])
    (H/'reference.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
    orbit=Orbit();periodic=np.tile(points,(30,1));m,_,_=orbit.metrics(periodic);shift,_,_=orbit.metrics(np.roll(periodic,17,axis=0))
    rng=np.random.default_rng(1);perturbed=periodic+np.repeat(rng.normal(0,.05,(30,17)),p,axis=0);noise,_,_=orbit.metrics(perturbed)
    assert m['section_count']>=28 and m['D_section']<1e-20 and shift['D_section']<1e-20
    assert noise['D_section']>1e-5 and noise['section_count']>=20
    check=dict(periodic=m,shifted=shift,perturbed=noise,passed=True)
    (H/'section_test.json').write_text(json.dumps(check,indent=2));print('SECTION TEST PASSED')

if __name__=='__main__':prepare()
