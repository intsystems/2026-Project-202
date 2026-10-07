"""E9 confirmatory run: MG of one neuron's activity against the known number of components.

PROTOCOL (written before any MG was computed on a network). Pilots on seed 0 (pilot.py,
pilot2.py; logs pilot*.log) looked only at learning success and Lyapunov exponents. They
fixed: multi-output training (one read-out cannot hold >= 3 incommensurate lines),
gain 1.2, N = 1000 and 100 000 steps (N = 500 fails at q = 4 even with 200 000 steps),
q <= 4 (q = 5 is not learned at any size tried). The MG configuration was chosen in
calibrate.py on the hand-written target signals, never on a network.

Setting. A rate network (N = 1000, gain 1.2, dt = 0.1) with m read-outs, each fed back
through its own random input vector, is trained by FORCE/RLS (shared P, update every 2
steps) so that read-out i generates sin(2 pi f_i t + phi_i). Weights are then frozen and
the network runs autonomously. Frequencies in units of the base 1/(10 pi):
  arm    frequencies               lines  active dimension (independent phases)
  T1     1                           1     1
  T2     1, sqrt2                    2     2
  T3     1, sqrt2, sqrt3             3     3
  T4     1, sqrt2, sqrt3, sqrt5      4     4
  H2     1, 2                        2     1   (harmonics: periodic)
  H4     1, 2, 3, 4                  4     1
  M4     1, 2, sqrt2, 2 sqrt2        4     2   (two bases, two harmonics each)
  chaos  untrained network, gain 1.5, no read-out: chaotic reference
Seeds 1-5 (network, feedback vectors and target phases).
The arms decouple the number of spectral lines from the dimension: H4, M4 and T4 all
have four lines and dimensions 1, 2, 4.

Ground truth, expensive and independent of MG:
  learned  = every read-out puts >= 80 % of its power within +-3 % of its target line;
  n_zero   = number of Lyapunov exponents >= -0.02 among the top 12 (Jacobian QR over
             16 384 steps of the frozen autonomous map). In the pilots the exponents of
             learned tori sit within 0.009 of zero and the next one is below -0.037.

Cheap measurement. The autonomous run (5 000 burn-in, then 32 768 steps). MG of ONE
neuron's rate tanh(x_0), four disjoint windows of 8 192, median. Frozen config
E20_acorr_k20: max_E 20, tau = quarter autocorrelation time, k 20, Theiler = autocorr,
cap 320. Neurons 1 and 2 are a robustness check, not used in P1-P4.

Competitors (what a practitioner would compute instead):
  peaks  = number of spectral peaks of the summed output with >= 5 % of the power;
  PR     = participation ratio of the PCA spectrum of all 1000 neurons (full state).

Predictions (learned runs only; the rest are reported apart). d = active dimension above.
  P1  MG grows with d across T1..T4: Spearman(MG, d) >= 0.8 over learned T runs;
  P2  lines are not dimension: in >= 4 of 5 seeds MG(H4) < MG(M4) < MG(T4), and
      MG(H2) < MG(T2);
  P3  over all learned runs Spearman(MG, d) >= 0.7 and larger than Spearman(peaks, d)
      and Spearman(PR, d);
  P4  MG ranks the runs like the Lyapunov count: Spearman(MG, n_zero) >= 0.7 over all
      learned runs;
  P5  the untrained chaotic network reads higher than the learned T1, T2 and H runs of
      its seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "code"))
sys.path.insert(0, str(HERE))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from generator import fidelity_multi, init_multi, peak_count, rollout, target_spec, train_multi  # noqa: E402

RES = HERE / "results"
N, STEPS, GAIN, BURN, LENGTH, WIN = 1000, 100000, 1.2, 5000, 32768, 8192
CHAOS_GAIN = 1.5          # untrained reference: clearly chaotic
CFG = EstimatorConfig(max_E=20, tau="acorr", k_neighbors=20, theiler="autocorr", theiler_cap=320)
ARMS = {"T1": ("torus", 1), "T2": ("torus", 2), "T3": ("torus", 3), "T4": ("torus", 4),
        "H2": ("harmonic", 2), "H4": ("harmonic", 4), "M4": ("mixed", 4), "chaos": (None, 0)}
DIM = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "H2": 1, "H4": 1, "M4": 2}
LYAP_ZERO = -0.02


def pca_pr(states):
    ev = np.linalg.eigvalsh(np.cov(states.T))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def run_one(arm, seed):
    kind, q = ARMS[arm]
    t0 = time.perf_counter()
    if kind is None:
        j, u, x = init_multi(seed, N, 1, gain=CHAOS_GAIN)
        w = np.zeros((N, 1))
        spec = None
    else:
        spec = target_spec(kind, q, seed)
        j, u, w, x, _ = train_multi(seed, N, spec, STEPS, gain=GAIN)
    t_train = time.perf_counter() - t0
    obs, z, lyap, tt, states = rollout(j, u, w, x, BURN, LENGTH, keep_states=True)
    zs = tt.pop("zs")
    row = {"arm": arm, "seed": seed, "q": q, "d": DIM.get(arm, np.nan), "t_train": t_train, **tt,
           "lyap": lyap.round(5).tolist(), "n_zero": int((lyap >= LYAP_ZERO).sum()),
           "lyap1": float(lyap[0]), "peaks": peak_count(z) if kind else np.nan,
           "PR": pca_pr(states)}
    if spec is not None:
        row["fidelity"], row["per_output"] = fidelity_multi(zs, spec)
        row["learned"] = row["fidelity"] >= 0.8
    else:
        row["fidelity"] = np.nan
        row["learned"] = False
    t1 = time.perf_counter()
    for c in range(obs.shape[1]):
        mg = [estimate(obs[a:a + WIN, c], CFG).MG for a in range(0, LENGTH, WIN)]
        row[f"MG_n{c}"] = float(np.median(mg))
        row[f"MG_n{c}_windows"] = [round(v, 3) for v in mg]
    row["t_MG"] = time.perf_counter() - t1
    return row, obs, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"runs_s{'_'.join(map(str, args.seeds))}.json"
    rows = json.load(open(out)) if out.exists() else []
    done = {(r["arm"], r["seed"]) for r in rows}
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            for arm in args.arms:
                if (arm, seed) in done:
                    continue
                row, obs, z = run_one(arm, seed)
                np.savez_compressed(RES / f"obs_{arm}_s{seed}.npz", obs=obs, z=z)
                rows.append(row)
                json.dump(rows, open(out, "w"), indent=1, default=float)
                print(f"{arm:5s} s{seed}: learned {row['learned']} fid {row['fidelity']:.3f} "
                      f"n_zero {row['n_zero']} lyap1 {row['lyap1']:+.4f} peaks {row['peaks']} "
                      f"PR {row['PR']:.2f} MG n0/n1/n2 {row['MG_n0']:.2f}/{row['MG_n1']:.2f}/"
                      f"{row['MG_n2']:.2f}  train {row['t_train']:.0f}s MG {row['t_MG']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
