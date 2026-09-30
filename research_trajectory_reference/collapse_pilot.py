"""E8 pilot: which learning-rate spikes kill ReLU channels in the E4/E5 CNN?

Only the dead fraction, the loss and the accuracy are looked at here -- no MG -- so
the choice of spike doses for E8 cannot depend on the estimator's response.
Seed 99, never used by E8.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cifar_events import load, model  # noqa: E402


def dead_fraction(net, Xp):
    acts = []
    hooks = [m.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
             for m in net if isinstance(m, nn.ReLU)]
    with torch.no_grad():
        net(Xp)
    for h in hooks:
        h.remove()
    dead = [(a.amax(dim=(0, 2, 3)) <= 1e-8) for a in acts]
    return float(torch.cat(dead).float().mean()), [int(d.sum()) for d in dead]


def run(factor, dur, seed=99, steps=6000, event=3000):
    X, y, *_ = load()
    Xp = X[:500]
    net = model(seed)
    opt = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1000 + seed)
    out = []
    for t in range(steps):
        lr = 0.02 * (factor if event <= t < event + dur else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.as_tensor(rng.integers(0, len(X), 64))
        opt.zero_grad()
        loss = lossf(net(X[idx]), y[idx])
        if not torch.isfinite(loss):
            return out, "diverged"
        loss.backward()
        opt.step()
        if t in (event - 1, event + dur + 10, event + 1000, steps - 1):
            f, per = dead_fraction(net, Xp)
            with torch.no_grad():
                acc = (net(X[:2000]).argmax(1) == y[:2000]).float().mean().item()
            out.append((t, round(f, 3), per, round(acc, 3), round(loss.item(), 3)))
    return out, "ok"


if __name__ == "__main__":
    torch.set_num_threads(6)
    for factor in (2, 5, 10, 20, 50, 100):
        for dur in (20,):
            res, status = run(factor, dur)
            print(f"x{factor:<4} dur {dur}: {status}")
            for r in res:
                print("    step %5d dead %.3f per-layer %s train-acc %.3f loss %.3f" % r)
