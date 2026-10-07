import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "pilot")
for f in sorted(D.glob("*.json")):
    t = json.load(open(f))
    oa = np.array([r["online_acc"] for r in t])
    ta = np.array([r["test_acc"] for r in t])
    dm = np.array([r["dormant_0.0"] for r in t])
    sr = np.array([r["srank"] for r in t])
    wn = np.array([r["wnorm"] for r in t])
    blk = lambda a: " ".join(f"{a[i:i + 5].mean():.3f}" for i in range(0, len(a), 5))
    print(f"{f.stem:28s} online: {blk(oa)}")
    print(f"{'':28s} test  : {blk(ta)}")
    print(f"{'':28s} dorm {dm[[0, 9, 19, 29, 39, -1]].round(2)} srank {sr[[0, 9, 19, 29, 39, -1]]} w {wn[[0, 9, 19, 29, 39, -1]].round(0)}")
