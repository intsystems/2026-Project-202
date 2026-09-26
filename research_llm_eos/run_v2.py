"""Corrected LM/GD experiment. See PROTOCOL.md; initial pilot is not evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from transformer_mg_benchmark import TinyGPT, batches

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))

def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()

def stamp(device):
    sync(device)
    return time.perf_counter()

def init(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

def loss_of(model, x, y):
    return nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())

def lanczos_top(mv, n, device, seed, steps=30):
    """Largest algebraic Ritz value, two-pass reorthogonalization, true residual."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(n, generator=gen, device=device)
    q = q / q.norm()
    Q = torch.empty((n, steps), device=device)
    diagonal, off = [], []
    previous = torch.zeros_like(q)
    beta_prev = torch.tensor(0., device=device)
    for j in range(steps):
        Q[:, j] = q
        z = mv(q) - beta_prev * previous
        alpha = torch.dot(q, z)
        z = z - alpha * q
        for _ in range(2):
            basis = Q[:, :j+1]
            z = z - basis @ (basis.T @ z)
        beta = z.norm()
        diagonal.append(float(alpha))
        if j == steps - 1 or float(beta) < 1e-8:
            break
        off.append(float(beta))
        previous, q, beta_prev = q, z / beta, beta
    T = np.diag(diagonal)
    if off:
        T += np.diag(off, 1) + np.diag(off, -1)
    values, vecs = np.linalg.eigh(T)
    v = Q[:, :len(diagonal)] @ torch.tensor(vecs[:, -1], device=device, dtype=Q.dtype)
    hv = mv(v)
    lam = float(torch.dot(v, hv))
    residual = float((hv-lam*v).norm()) / max(abs(lam), 1e-8)
    return lam, residual, len(diagonal)+1

def hessian(model, x, y, seed):
    params = list(model.parameters())
    sizes = [p.numel() for p in params]
    loss = loss_of(model, x, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    def mv(v):
        pieces = [a.reshape_as(p) for a, p in zip(v.split(sizes), params)]
        hs = torch.autograd.grad(grads, params, grad_outputs=pieces, retain_graph=True)
        return torch.cat([h.detach().flatten() for h in hs])
    lam, residual, calls = lanczos_top(mv, sum(sizes), x.device, seed, 30)
    if residual > .03:
        lam, residual, extra = lanczos_top(mv, sum(sizes), x.device, seed, 60)
        calls += extra
    return dict(lambda_max=lam, ritz_residual=residual, hvp_count=calls)

@torch.no_grad()
def activation_pr(model, x):
    h = model.tok(x) + model.pos(torch.arange(x.shape[1], device=x.device))[None]
    for block in model.blocks:
        h = block(h, model.mask[:x.shape[1], :x.shape[1]])
    h = model.ln(h).flatten(0, 1)
    h = h - h.mean(0)
    c = h.T @ h
    return float(torch.trace(c).square() / c.square().sum().clamp_min(1e-20))

def gradient_proxy(model, x, y):
    """Trace(sample gradient covariance) / squared mean gradient, four groups."""
    grads = []
    for xx, yy in zip(x.chunk(4), y.chunk(4)):
        g = torch.autograd.grad(loss_of(model, xx, yy), tuple(model.parameters()))
        grads.append(torch.cat([z.detach().flatten() for z in g]))
    G = torch.stack(grads)
    mean = G.mean(0)
    var = (G-mean).square().sum() / (len(G)-1)
    return float(var / mean.square().sum().clamp_min(1e-20))

def data(seed, batch, context, device, fixed_batch=128):
    source = HERE/'data/tinyshakespeare.txt'
    raw = source.read_bytes()
    text = raw.decode('utf-8')
    vocab = sorted(set(text))
    ids = {v:i for i,v in enumerate(vocab)}
    d = torch.tensor([ids[v] for v in text])
    cut = int(.9*len(d))
    stream = batches(d[:cut], batch, context, torch.Generator().manual_seed(10000+seed))
    fx, fy = next(batches(d[:cut], fixed_batch, context, torch.Generator().manual_seed(20000+seed)))
    vx, vy = next(batches(d[cut:], batch, context, torch.Generator().manual_seed(30000+seed)))
    return stream, (fx.to(device),fy.to(device)), (vx.to(device),vy.to(device)), vocab, hashlib.sha256(raw).hexdigest()

def run(args, seed, mode, lr):
    device = torch.device(args.device)
    torch.manual_seed(seed)
    batch, context = 16, 64
    stream, fixed, probe, vocab, digest = data(seed, batch, context, device,
                                               fixed_batch=args.gd_fixed_batch)
    model = TinyGPT(len(vocab), context, args.width, 4, args.layers).to(device)
    model.apply(init)
    opt = (torch.optim.SGD(model.parameters(), lr=lr) if args.arm=='gd'
           else torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.999), weight_decay=.01))
    key = f'{args.arm}_{mode}_lr{lr:g}_s{seed}'
    out = args.out/key
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists():
        print('Already complete:', key, flush=True)
        return
    meta = dict(arm=args.arm, mode=mode, seed=seed, lr=lr, steps=args.steps,
                switch=args.steps//2 if mode=='switch' else None, batch=batch, context=context,
                width=args.width, layers=args.layers, gd_fixed_batch=args.gd_fixed_batch,
                parameters=sum(p.numel() for p in model.parameters()),
                vocab=len(vocab), dataset='Tiny Shakespeare', sha256=digest, device=str(device),
                torch=torch.__version__, threads=torch.get_num_threads(), platform=platform.platform(),
                hessian_objective='full fixed training set' if args.arm=='gd' else 'fixed validation probe')
    (out/'config.json').write_text(json.dumps(meta,indent=2))
    rows, diag = [], []
    train_sec = probe_sec = norm_sec = 0.
    wall = stamp(device)
    for step in range(args.steps):
        if mode=='switch' and step==args.steps//2:
            for group in opt.param_groups:
                group['lr'] = lr/10
        t = stamp(device)
        x,y = fixed if args.arm=='gd' else tuple(a.to(device) for a in next(stream))
        opt.zero_grad(set_to_none=True)
        loss = loss_of(model,x,y)
        if not bool(torch.isfinite(loss)) or float(loss.detach())>1e6:
            meta['diverged_step']=step
            break
        loss.backward()
        train_sec += stamp(device)-t
        t = stamp(device)
        gn = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
        norm_sec += stamp(device)-t
        t = stamp(device)
        with torch.no_grad():
            vl = float(loss_of(model,*probe))
        probe_sec += stamp(device)-t
        row = dict(step=step,train_loss=float(loss.detach()),probe_loss=vl,grad_norm=gn,
                   lr=opt.param_groups[0]['lr'])
        rows.append(row)
        if step % args.every==args.every-1 or step==args.steps-1:
            d = dict(step=step, lr=row['lr'])
            t = stamp(device)
            d.update(hessian(model,*(fixed if args.arm=='gd' else probe),seed=seed+40000+step))
            d['hessian_seconds']=stamp(device)-t
            d['gd_ratio']=row['lr']*d['lambda_max']/2 if args.arm=='gd' else float('nan')
            t = stamp(device)
            d['activation_pr']=activation_pr(model,probe[0])
            d['activation_seconds']=stamp(device)-t
            t = stamp(device)
            d['gradient_proxy']=gradient_proxy(model,*fixed)
            d['gradient_seconds']=stamp(device)-t
            diag.append(d)
            print(f"{key} step {step}: loss {row['train_loss']:.3f}, H {d['lambda_max']:.2f}, residual {d['ritz_residual']:.3g}",flush=True)
        t = stamp(device)
        opt.step()
        train_sec += stamp(device)-t
    meta.update(train_seconds=train_sec,probe_seconds=probe_sec,norm_seconds=norm_sec,
                wall_seconds=stamp(device)-wall,completed_steps=len(rows))
    pd.DataFrame(rows).to_csv(out/'trace.csv',index=False)
    pd.DataFrame(diag).to_csv(out/'diagnostics.csv',index=False)
    torch.save(model.state_dict(), out/'final.pt')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('Complete',key,meta['wall_seconds'],flush=True)

def validate():
    # Indefinite operator: largest magnitude -20 is deliberately not algebraic max 7.
    diag=torch.tensor([-20.,-2.,1.,3.,7.])
    lam,res,_=lanczos_top(lambda v:diag*v,5,'cpu',12,5)
    assert abs(lam-7)<1e-4 and res<1e-4,(lam,res)
    torch.manual_seed(11)
    model=TinyGPT(7,8,16,4,1).double()
    model.apply(init)
    x=torch.randint(7,(2,8)); y=torch.randint(7,(2,8))
    ps=list(model.parameters()); n=sum(p.numel() for p in ps)
    v=torch.randn(n,dtype=torch.float64); v/=v.norm()
    vv=[a.reshape_as(p) for a,p in zip(v.split([p.numel() for p in ps]),ps)]
    g=torch.autograd.grad(loss_of(model,x,y),ps,create_graph=True)
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(g,ps,grad_outputs=vv)])
    eps=1e-5
    original=[p.detach().clone() for p in ps]
    gs=[]
    for sign in (1,-1):
        with torch.no_grad():
            for p,o,d in zip(ps,original,vv):p.copy_(o+sign*eps*d)
        gs.append(torch.cat([z.flatten() for z in torch.autograd.grad(loss_of(model,x,y),ps)]))
    err=float(((gs[0]-gs[1])/(2*eps)-hv).norm()/hv.norm())
    assert err<1e-5,err
    print('Lanczos indefinite test and HVP finite-difference test passed; error',err)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['adam','gd'],default='adam')
    p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--every',type=int,default=128)
    p.add_argument('--lrs',type=float,nargs='+',default=[.003])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--modes',nargs='+',choices=['constant','switch'],default=['constant','switch'])
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--gd-fixed-batch',type=int,default=128)
    p.add_argument('--out',type=Path,default=HERE/'v2')
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    if a.validate:
        validate();return
    for lr in a.lrs:
        for seed in a.seeds:
            for mode in a.modes:run(a,seed,mode,lr)

if __name__=='__main__':main()
