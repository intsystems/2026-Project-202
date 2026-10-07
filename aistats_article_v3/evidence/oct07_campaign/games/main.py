"""Setting G: how many independent cycles persist in learning in a zero-sum game, read from ONE
scalar payoff log. PROTOCOL (written before any MG/competitor value was computed on any run;
the pilot `pilot.py`, seeds 9000-9111, looked only at ground truth, periods, drift, SNR, cost).

Why. Learning in zero-sum games (MWU/FTRL, softmax PG, GDA) cycles instead of converging
(Mertikopoulos, Papadimitriou & Piliouras 2018; Bailey & Piliouras 2018). The number of
independent cycling components (intransitive dimension of the "spinning top", Czarnecki et al.
2020; disc-game dimension of the gamescape, Balduzzi et al. 2019) sets the remedies: the
PSRO/league population size (Lanctot et al. 2017), and whether a fix left K' cycles alive.

Game (sim.py). K_tot = K_live + K_tr weakly coupled 2x2 zero-sum subgames: K_live generalised
matching-pennies subgames (interior Nash in (0.25,0.75)^2, scale a_k log-uniform [0.5,2] ->
incommensurate frequencies), K_tr in {0,1,2} transitive subgames (dominant actions, converge),
coupling eps*C, eps ~ U(0,0.15); random payoff unit S in [0.5,2]; logit amplitude of each
cycle U(0.3,3) (large -> strongly anharmonic, near-boundary switching); exogenous slow
transitive drift S*A*tanh((t-tc)/T), A up to the mean cycle scale, T in [5k,40k] steps.
Games whose state-measured cycle frequencies are resonant (ratio within 1.5 % of 1, 1.5, 2, 3)
are redrawn (truth-only filter; attempts logged).
GROUND TRUTH: K_live (by construction); checked: linearisation at the interior rest point has
exactly K_live imaginary pairs (pilot: 96/96).
Decision: population / niche budget for the remaining intransitive dimension: classes
K_hat in {1, 2, >=3}. Cost (pre-set): 2 per missing cycle (unrepresented cycle -> residual
exploitability), 1 per extra (compute).
Log: expected payoff of player 1 per step (+drift), estimated from B sampled plays (Gaussian
with exact per-play variance / B). Burn-in 4000 steps, then 2 windows of W=8192; per-run
statistic = median over the 2 windows.

Arms (24 games per K per cell):
  CAL      alt-MWU eta=.05, payoff, B in {inf,1e5,1e4}, K_live in {1,2,3}; seeds 100000+ (216 runs)
  TEST_IN  same distribution, new games; seeds 200000+                         (216 runs)
  shifts (K_live in {1,2,3}, B=1e5 unless noted; seeds 300000+ per arm, 72 runs each):
  S_sim    simultaneous MWU eta=.02 (outward spiral to the boundary, periods ~500-1600)
  S_pg     alternating softmax policy gradient eta=.1 (other nonlinearity, long periods)
  S_decay  alternating MWU, eta_t=.08/sqrt(1+t/20000) (chirping frequencies)
  S_win    log = win rate Phi(mu/sigma_play) from B=1e5 plays (binomial)
  S_noise  B=1e3 (heavy sampling noise; SNR ~3-17 in the pilot)
  S_K4     K_live in {2,3,4} (class >=3 for K=3,4; also report K3 vs K4 separation)
  S_wide   a_k in [0.3,3], eps up to 0.3, K_tr up to 3

Statistics (stats.py), all on the same windows:
  MG primary: EstimatorConfig(max_E=20, tau='acorr', k=20, theiler='autocorr', cap=320)
     (the E9 configuration that counted phases); MG_tau1 (guide default tau=1) reported too.
  scalar competitors: spectral_entropy, self_repeat (lags 20-250) and self_repeat_long
     (20-2000), roughness, peak_count (rel in .01/.02/.05/.1), harmonic_count (same rel grid;
     fundamentals explaining all strong peaks as integer combinations -- a strong spectral
     counter for anharmonic cycles), perm_entropy, recurrence_rate, corr_dim, twonn,
     linear_pr, crossings, lag1, det_std.
  domain (internals, more than one scalar): xi2 = mean ||xi||^2 of the game vector field
     (Balduzzi 2018); state_pr / state_count_r (PCA of the full strategy state);
     xplay_pr / xplay_count_r (singular values of the double-centred cross-play matrix of 24
     checkpoints, B plays per entry = gamescape dimension, Balduzzi 2019).
  trivial: always 2 (prior), always 3 (never under-provision).
Rule (identical for every statistic): on CAL choose sign and two cut points on the per-run
statistic maximising 3-class accuracy (ties: lower cost, then first); statistics with a grid
(peak_count, harmonic_count, state_count, xplay_count) choose the grid value on CAL. Thresholds
frozen and applied to all test arms. NaN -> class 2.
Metrics: test accuracy (primary, TEST_IN and each shift), mean cost, MAE, Spearman rho(stat, K)
within arm (threshold-free).
Predictions: P1 on TEST_IN MG accuracy exceeds every spectral/periodicity/simple scalar
competitor by >= 5 pp. P2 MG ties (+-5 pp) with twonn / recurrence_rate / corr_dim. P3 xplay
(domain, cross-play) >= MG at B=inf. P4 MG degrades at B=1e3. P5 under shifts MG keeps more
accuracy than the spectral competitors.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results"

ETA = {"alt": 0.05, "sim": 0.02, "pg": 0.1, "decay": 0.08}


def specs():
    S = []

    def add(arm, base, Ks, Bs, learner="alt", obs="payoff", wide=False, n=24):
        i = 0
        for K in Ks:
            for B in Bs:
                for _ in range(n):
                    S.append(dict(arm=arm, seed=base + i, K=K, B=B, learner=learner, obs=obs, wide=wide))
                    i += 1
    add("CAL", 100000, (1, 2, 3), (np.inf, 1e5, 1e4))
    add("TEST_IN", 200000, (1, 2, 3), (np.inf, 1e5, 1e4))
    add("S_sim", 300000, (1, 2, 3), (1e5,), learner="sim")
    add("S_pg", 310000, (1, 2, 3), (1e5,), learner="pg")
    add("S_decay", 320000, (1, 2, 3), (1e5,), learner="decay")
    add("S_win", 330000, (1, 2, 3), (1e5,), obs="winrate")
    add("S_noise", 340000, (1, 2, 3), (1e3,))
    add("S_K4", 350000, (2, 3, 4), (1e5,))
    add("S_wide", 360000, (1, 2, 3), (1e5,), wide=True)
    return S


def run(spec):
    import sim as G
    import stats as ST
    t0 = time.perf_counter()
    rng = np.random.default_rng(spec["seed"])
    for attempt in range(1, 31):
        if spec["wide"]:
            game = G.make_game(rng, spec["K"], int(rng.integers(0, 4)), scale_range=(0.3, 3.0), eps_max=0.3)
        else:
            game = G.make_game(rng, spec["K"], int(rng.integers(0, 3)))
        lin = G.linear_cycles(game)
        if lin != spec["K"]:
            continue
        s = G.simulate(game, spec["learner"], ETA[spec["learner"]])
        if not G.resonant(G.subgame_freqs(s, game), tol=0.015, ratios=(1.0, 1.5, 2.0, 3.0)):
            break
    x = G.observe(game, s, spec["obs"], spec["B"], rng)
    t_sim = time.perf_counter() - t0
    row = dict(spec, B=float(spec["B"]), K_tr=game.K_tr, attempts=attempt, lin=lin, t_sim=t_sim)
    timing = {}
    for w in range(G.NWIN):
        a0 = G.BURN + w * G.W
        seg = x[a0:a0 + G.W]
        for k, v in ST.scalar_stats(seg, timing).items():
            row[f"w{w}_{k}"] = v
        for k, v in ST.domain_stats(game, s, a0, a0 + G.W, spec["B"], rng).items():
            row[f"w{w}_{k}"] = v
    row["timing"] = {k: round(v, 3) for k, v in timing.items()}
    row["t_total"] = time.perf_counter() - t0
    return row


def main():
    OUT.mkdir(exist_ok=True)
    f = OUT / "runs.jsonl"
    done = set()
    if f.exists():
        for line in open(f):
            r = json.loads(line)
            done.add((r["arm"], r["seed"]))
    todo = [s for s in specs() if (s["arm"], s["seed"]) not in done]
    arms = sys.argv[1:]
    if arms:
        todo = [s for s in todo if s["arm"] in arms]
    print(f"{len(todo)} runs to do", flush=True)
    t0 = time.time()
    with Pool(2) as p, open(f, "a") as fh:
        for i, r in enumerate(p.imap_unordered(run, todo, chunksize=1)):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if i % 20 == 0:
                print(f"{i+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
