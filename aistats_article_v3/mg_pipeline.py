"""Measure a CSV scalar log with explicit flags; never certify dimension automatically."""
from pathlib import Path
import argparse, csv, sys
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'measurement_code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate

def measure(x,window,stride,E,tau,k,theiler,min_std,dither_seed=0):
    cfg=EstimatorConfig(max_E=E,tau=tau,k_neighbors=k,theiler=theiler,
                        theiler_cap=theiler)
    for end in range(window,len(x)+1,stride):
        seg=x[end-window:end]
        std=float(np.std(seg)); status='change_statistic'
        row=dict(start=end-window,end=end,raw_std=std,MG=np.nan,MG_2E=np.nan,
                 rho_E=np.nan,n_points=0,degenerate=True,frac_floor=np.nan,
                 frac_sumfloor=np.nan,status='unusable')
        if np.isfinite(seg).all() and std>min_std:
            r=estimate(seg,cfg,seed=dither_seed)
            r2=estimate(seg,cfg.replace(max_E=2*E),seed=dither_seed)
            row.update(MG=r.MG,MG_2E=r2.MG,n_points=r.n_points,degenerate=r.degenerate,
                       frac_floor=r.floor_distance_fraction,frac_sumfloor=r.floor_sum_fraction)
            if r.degenerate or not np.isfinite(r.MG):status='unusable'
            if np.isfinite(r.MG) and r.MG>0 and np.isfinite(r2.MG) and not r2.degenerate:
                row['rho_E']=r2.MG/r.MG
            row['status']=status
        yield row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--column',required=True)
    for flag in ['window','stride','E','tau','k','theiler']:
        p.add_argument('--'+flag,type=int,required=True)
    p.add_argument('--min-std',type=float,required=True,help='Raw signal noise/precision floor chosen on calibration data; 0 only rejects exact constants.')
    p.add_argument('--dither-seed',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if min(args.window,args.stride,args.E,args.tau)<1 or args.k<2 or args.theiler<0 or args.min_std<0:
        p.error('Invalid measurement settings')
    with args.input.open(newline='',encoding='utf-8-sig') as f:
        x=np.array([float(r[args.column]) for r in csv.DictReader(f)])
    rows=list(measure(x,args.window,args.stride,args.E,args.tau,args.k,args.theiler,args.min_std,args.dither_seed))
    if not rows:p.error('Record is shorter than the specified window')
    with args.out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'{len(rows)} windows; {sum(r["status"]=="unusable" for r in rows)} unusable. Read MEASUREMENT.md before interpreting values.')
if __name__=='__main__':main()
