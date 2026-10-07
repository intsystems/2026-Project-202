"""Choose the MG configuration for E10 on target signals only (T1, T2, T3: d=1,2,3; Lorenz: 2.06)."""
import sys, numpy as np
from itertools import product
sys.path.insert(0, '.')
import esn_dsr as E, baselines as BL
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
TRUE = {"T1": 1, "T2": 2, "T3": 3, "lorenz": 2.06}
res = []
for P_, E_, tau, k in product([40.37, 60.37, 90.37], [20], ["acorr"], [20, 50]):
    E.PERIOD = P_; dec = 1
    sig = {t: [E.torus(8192 + 10, int(t[1:]), s, period=P_)[0] if t[0] == "T" else E.lorenz(8192 + 10, s)[0] for s in (50, 51)] for t in TRUE}
    cfg = EstimatorConfig(max_E=E_, tau=tau, k_neighbors=k, theiler="autocorr" if tau == "acorr" else "embedding", theiler_cap=320)
    est = {t: np.mean([estimate(x[::dec][:8192], cfg).MG for x in xs]) for t, xs in sig.items()}
    err = np.mean([abs(est[t] - TRUE[t]) for t in TRUE])
    res.append((err, dec, E_, tau, k, est))
    print(f"P{P_} E{E_} tau{tau} k{k}: " + " ".join(f"{t}={v:.2f}" for t, v in est.items()) + f"  err {err:.2f}", flush=True)
best = min(res, key=lambda r: r[0]); print("BEST", best)
