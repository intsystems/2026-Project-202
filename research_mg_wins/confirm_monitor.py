"""E16: fresh confirmation of the monitor comparison of E15.

PROTOCOL (written after E15, before any E16 run). New seeds 30-33, training as E5.
Event arms at step 4 000: lr10, lr100, freeze_head, freeze_bias, prune50, prune80, prune95
(28 strong events). No-event arms: base, batch_up, resume_exact, opt_reset, reseed, bf16,
clip (28 runs). Every variant of E15 is scored with its frozen rule from
results_variants/rules.json.
Hypotheses, stated before the runs:
  H1  MG_E20_t4_k50 (best MG variant of E15, picked after seeing tests) gets >= 24/28 hits
      and <= 2/28 false alarms;
  H2  it has fewer false alarms than every self_repeat and recurrence variant that gets
      >= 24/28 hits;
  H3  MG_E20_t2_k20 (the variant chosen on calibration) has fewer false alarms than the
      calibration-chosen self_repeat (SR_1_50) and recurrence (RR_0.1) variants.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "research_trajectory_reference"
sys.path.insert(0, str(REF)); sys.path.insert(0, str(HERE))
from cifar_events import model  # noqa: E402
from cifar_cache import load  # noqa: E402
import detector as D  # noqa: E402
from variants import VARIANTS  # noqa: E402

OUT = HERE / "results_confirm"
EVENTS = ("lr10", "lr100", "freeze_head", "freeze_bias", "prune50", "prune80", "prune95")
NULLS = ("base", "batch_up", "resume_exact", "opt_reset", "reseed", "bf16", "clip")
EVENT, STEPS = 4000, 10000
_DATA = None


def train(args):
    global _DATA
    arm, seed = args
    path = OUT / f"pn_{arm}_s{seed}.npy"
    if path.exists():
        return
    torch.set_num_threads(1)
    if _DATA is None:
        _DATA = load()
    X, y = _DATA[0], _DATA[1]
    net = model(seed)
    params = list(net.parameters())
    names = [n for n, _ in net.named_parameters()]
    mk = lambda ps: torch.optim.SGD(ps, lr=0.02, momentum=0.9, weight_decay=5e-4)  # noqa: E731
    opt = mk(params)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1000 + seed)
    bs, masks, bf16, clip = 64, None, False, None
    pn, gn = np.empty(STEPS), np.empty(STEPS)
    for t in range(STEPS):
        if t == EVENT:
            if arm.startswith("lr"):
                for g in opt.param_groups:
                    g["lr"] /= float(arm[2:])
            elif arm.startswith("freeze"):
                keep = {"freeze_head": lambda n: n.startswith("10."), "freeze_bias": lambda n: n == "10.bias"}[arm]
                for n, p in zip(names, params):
                    p.requires_grad_(keep(n))
                opt = mk([p for n, p in zip(names, params) if keep(n)])
            elif arm == "batch_up":
                bs = 256
            elif arm.startswith("prune"):
                frac = float(arm[5:]) / 100
                w = [p for p in params if p.dim() > 1]
                thr = torch.quantile(torch.cat([p.detach().abs().reshape(-1) for p in w]), frac)
                masks = [(p.detach().abs() > thr).float() for p in w]
                with torch.no_grad():
                    for p, m in zip(w, masks):
                        p.mul_(m)
            elif arm == "resume_exact":
                sd, so = copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict())
                net.load_state_dict(sd); opt = mk(params); opt.load_state_dict(so)
            elif arm == "opt_reset":
                opt = mk(params)
            elif arm == "reseed":
                rng = np.random.default_rng(777 + seed)
            elif arm == "bf16":
                bf16 = True
            elif arm == "clip":
                clip = 2 * float(np.median(gn[3000:4000]))
        idx = torch.as_tensor(rng.integers(0, len(X), bs))
        opt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            loss = lossf(net(X[idx]), y[idx])
        loss.backward()
        gn[t] = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)).item()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        with torch.no_grad():
            if masks is not None:
                for p, m in zip([p for p in params if p.dim() > 1], masks):
                    p.mul_(m)
            pn[t] = torch.sqrt(sum((p ** 2).sum() for p in params)).item()
    np.save(path, pn)


def windows(args):
    arm, seed = args
    x = np.load(OUT / f"pn_{arm}_s{seed}.npy")
    rows = []
    for a in range(0, STEPS - D.W + 1, D.S):
        seg = x[a:a + D.W]
        r = {"arm": arm, "seed": seed, "start": a, "end": a + D.W}
        for name, (_, fn) in VARIANTS.items():
            try:
                r[name] = float(fn(seg))
            except Exception:
                r[name] = np.nan
        rows.append(r)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(a, s) for s in (30, 31, 32, 33) for a in EVENTS + NULLS]
    with Pool(12) as p:
        p.map(train, jobs, chunksize=1)
        W = pd.DataFrame([r for rr in p.map(windows, jobs) for r in rr])
    W.to_csv(OUT / "windows.csv", index=False)
    rules = json.load(open(HERE / "results_variants" / "rules.json"))
    rows = []
    for name, r in rules.items():
        pr = D.evaluate(W, name, r["M"], r["B"], r["delta"], r["sign"], NULLS, EVENT)
        rows.append({"variant": name, "hits /28": int(pr[pr.arm.isin(EVENTS)].hit.sum()),
                     "FA /28": int(pr[pr.arm.isin(NULLS)].false_alarm.sum()),
                     "early": int(pr[pr.event].false_alarm.sum()),
                     **{f"FA {a}": int(pr[pr.arm == a].false_alarm.sum()) for a in NULLS},
                     "delay": float(np.nanmedian(pr[pr.hit].delay)) if pr.hit.any() else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 300)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
