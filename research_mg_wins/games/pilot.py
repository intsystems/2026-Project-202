"""Pilot (seeds 9000+): ground truth and dynamics only -- no MG, no competitors.

Checks: linearisation count == K_live; cyclic subgames keep amplitude, transitive ones
converge; outward drift of `sim`; frequency spread and resonance rejection rate; SNR of the
sampled payoff; wall-clock of one simulation.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim as G

ETA = {"alt": 0.05, "sim": 0.02, "pg": 0.05, "decay": 0.08, "omwu": 0.05}
rows = []
for seed in range(9000, 9024):
    rng = np.random.default_rng(seed)
    K_live = 1 + seed % 4
    K_tr = rng.integers(0, 3)
    game = G.make_game(rng, K_live, K_tr)
    lin = G.linear_cycles(game)
    for learner in ("alt", "sim", "pg", "decay"):
        t0 = time.perf_counter()
        s = G.simulate(game, learner, ETA[learner])
        dt = time.perf_counter() - t0
        cy = np.where(game.cyclic)[0]; tr = np.where(~game.cyclic)[0]
        w1 = slice(G.BURN, G.BURN + G.W); w2 = slice(G.BURN + G.W, G.T_TOTAL)
        th = np.log(s["x"] / (1 - s["x"]))
        amp1 = th[w1][:, cy].std(0); amp2 = th[w2][:, cy].std(0)
        trs = s["x"][w1][:, tr].std(0) if len(tr) else np.array([0.0])
        f = G.subgame_freqs(s, game)
        u = s["u"]
        mu = u + G.drift(game)
        sd1e4 = np.sqrt(G.play_variance(game, s["x"], s["y"]) / 1e4)
        snr4 = mu[G.BURN:].std() / sd1e4[G.BURN:].mean()
        dstd = G.drift(game)[G.BURN:].std() / max(u[G.BURN:].std(), 1e-12)
        rows.append(dict(seed=seed, K=K_live, Ktr=int(K_tr), lin=lin, learner=learner,
                         amp_min=amp1.min().round(2), amp_max=amp1.max().round(2),
                         growth=(amp2 / amp1).max().round(2), tr_std=trs.max().round(3),
                         per_min=int(1 / f.max()) if f.max() > 0 else -1,
                         per_max=int(1 / f.min()) if f.min() > 0 else -1,
                         reso=G.resonant(f), xmin=s["x"][G.BURN:, cy].min().round(4),
                         snr1e4=round(snr4, 1), snr1e3=round(snr4 / np.sqrt(10), 1),
                         drift_rel=round(dstd, 2), sec=round(dt, 2)))
        print(rows[-1], flush=True)
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "pilot_truth.csv", index=False)
print(df.groupby("learner")[["growth", "per_min", "per_max", "reso", "snr1e4", "sec"]].describe().T.to_string())
