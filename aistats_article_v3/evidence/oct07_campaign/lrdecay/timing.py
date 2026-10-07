import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "research_trajectory_reference"))
from cifar_cache import load
from cifar_events import model
X, y, Xp, yp, Xt, yt = load()
net = model(0)
opt = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
lf = nn.CrossEntropyLoss()
for bs in (32, 64, 128):
    t0 = time.perf_counter()
    for t in range(200):
        idx = torch.randint(0, len(X), (bs,))
        opt.zero_grad(); l = lf(net(X[idx]), y[idx]); l.backward(); opt.step()
    print(bs, (time.perf_counter() - t0) / 200 * 1000, "ms/step", flush=True)
t0 = time.perf_counter()
with torch.no_grad():
    net(Xt).argmax(1)
print("test eval 2000", time.perf_counter() - t0)
