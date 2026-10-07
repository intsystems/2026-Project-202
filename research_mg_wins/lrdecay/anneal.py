"""S3b: DECAY-AND-STOP under an average compute budget (second protocol of S3).

PROTOCOL (written 7 Oct 2026 ~05:45 UTC+3, BEFORE any S3b run and BEFORE looking at any S3
(main.py) test-unit rule result; I have seen only the S3 CALIBRATION rule results, where a
fixed decay at 15/16 T beat every diagnostic by ~1 point of accuracy (regret 0.0035 vs >= 0.013):
with a known budget the answer to "when to decay" is trivially "late". The convergence-
diagnostic literature (Pflug; Chee & Toulis 2018; Lang et al. 2019; Pesme et al. 2020) uses
the diagnostic to stop wasting constant-LR iterations once SGD is stationary, i.e. to save
compute. S3b scores exactly that.)

Policy. Constant LR lr0 until the decision checkpoint t_j = j*T/16 (first checkpoint >= the
rule's alarm, j = 2..14; no alarm by t_14 -> t_14), then K = T/8 steps at lr0/10, then STOP.
Compute used = (t_j + K)/T in [0.25, 1.0]. Score: clean test accuracy at stop (secondary: the
same with a K/2 anneal, test loss).
Measurement. Same 36 units, conditions, seeds and batch sequences as main.py (12 calibration
units A-F x {0,1}; 24 test units A-F x {10,11}, G-J x {10,11,12}). The trunk is replayed
(deterministic; checked against the stored parameter-norm log) and from each t_j a K-step
anneal at lr0/10 is run. Rules read the SAME stored trunk logs and windows as in main.py.
Rules and grids: exactly those of main.py/rules.py (fixed_step = decide at f*T, f in
{2..14}/16; plateau train/val; Pflug/Chee-Toulis; SASA/SASA+; Pesme distance; MG and the 12
scalar competitors with level / relative-change / stabilisation logic on param-norm or
mini-batch-loss windows; MG = E=20, tau=1, k=20).
Calibration (calibration units only): maximise mean test accuracy at stop subject to mean
compute <= C. Primary C = 0.5, secondary C = 0.75.
Test metrics (test units): mean accuracy, mean compute, and the PRIMARY metric
"gain over the fixed-step frontier at matched compute": acc(rule) minus the mean test accuracy
of the fixed decision time with the same mean compute (linear interpolation of the fixed-step
test curve, f = 2..14/16, measured on the same test units). Paired bootstrap CI of MG minus
each competitor (unit-level), and of MG's frontier gain.
Prediction (hypothesis): MG's frontier gain > 0 and >= every competitor's; plateau_val_loss
(uses held-out data) is the strongest competitor. Null expectation: all gains ~ 0.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_anneal"
RES = HERE / "results"


def job(args):
    name, seed, split = args
    sys.path.insert(0, str(HERE))
    tag = f"{name}_s{seed}"
    if (OUT / f"ann_{tag}.json").exists():
        return tag
    try:                                   # claim the unit (several runner instances may share the list)
        fd = os.open(OUT / f"lock_{tag}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return tag
    import numpy as np
    import torch
    import torch.nn as nn
    import lrrun
    from main import CONDS
    torch.set_num_threads(1)
    c = dict(CONDS[name], name=name, cid=ord(name), split=split)
    T, lr0 = c["T"], c["lr"]
    X, y, Xv, yv, Xt, yt, batches = lrrun.setup(c, seed)
    lossf = nn.CrossEntropyLoss()
    net = lrrun.model(seed)
    params = list(net.parameters())
    opt = torch.optim.SGD(params, lr=lr0, momentum=lrrun.MOM, weight_decay=lrrun.WD)
    G = [j * T // 16 for j in range(2, 15)]
    K = T // 8
    ref = np.load(RES / f"logs_{tag}.npz")["param_norm"]
    pn = np.empty(T)
    ck = {}
    t0 = time.perf_counter()
    for t in range(T):
        if t in G:
            ck[t] = (copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict()))
        lrrun.step(net, opt, X, y, torch.as_tensor(batches[t]), lossf)
        opt.step()
        with torch.no_grad():
            pn[t] = torch.sqrt(sum((p.detach() ** 2).sum() for p in params)).item()
        if t >= G[-1]:
            break
    dev = float(np.nanmax(np.abs(pn[:G[-1] + 1] - ref[:G[-1] + 1])))
    out = {"cond": c, "seed": seed, "grid": G, "K": K, "replay_max_abs_dev_param_norm": dev, "anneal": {}}
    for tj in G:
        sd, osd = ck.pop(tj)
        net.load_state_dict(sd)
        opt = torch.optim.SGD(params, lr=lr0, momentum=lrrun.MOM, weight_decay=lrrun.WD)
        opt.load_state_dict(osd)
        for g in opt.param_groups:
            g["lr"] = lr0 / 10
        rec = {}
        for k in range(K):
            lrrun.step(net, opt, X, y, torch.as_tensor(batches[tj + k]), lossf)
            opt.step()
            if k + 1 in (K // 2, K):
                tl, ta = lrrun.evaluate(net, Xt, yt, lossf)
                rec[f"K{k + 1}"] = {"test_loss": tl, "test_acc": ta}
        out["anneal"][str(tj)] = rec
    out["t_total"] = time.perf_counter() - t0
    json.dump(out, open(OUT / f"ann_{tag}.json", "w"), indent=1)
    print(tag, f"dev {dev:.2e}", f"{out['t_total']:.0f}s", flush=True)
    return tag


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    from main import CAL, TEST
    jobs = [(k, s, "cal") for k, s in CAL] + [(k, s, "test") for k, s in TEST]
    if len(sys.argv) > 2:
        jobs = [j for j in jobs if f"{j[0]}_s{j[1]}" in sys.argv[2].split(",")]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3, maxtasksperchild=6) as p:
        for _ in p.imap_unordered(job, jobs):
            pass
    print("ALL DONE", flush=True)
