# Does a 1-D log measure the *active* complexity of a training run?

Three quantities are kept apart throughout, because conflating them is what every earlier
experiment in this repository did:

| | what it is | how it is obtained here |
| --- | --- | --- |
| **available** | how many directions the optimiser is *allowed* to move in | by construction: the adapter is `theta = theta0 + V^T c`, `V` a fixed (k, P) orthonormal frame |
| **functional** | rank of the Jacobian of the model's outputs w.r.t. those directions | measured on held-out data (`Adapter.functional_dim`) |
| **active** | how many directions the optimiser *actually* excites over the window | measured from the trajectory and update covariances (`traj_PR`, `upd_PR`) |

MG -- the MacKay-Ghahramani-pooled Levina-Bickel intrinsic dimension of a delay embedding --
is scored against **active**, never against available and never against nominal r.

```
python e1_calibration.py    # freeze the estimator on calibration seeds AND r      ~50 min
python run_all.py           # e0, e6, e2, e3, e4, e5, figures                      ~3.5 h
python analyze.py           # every table the report quotes  -> results/tables.txt
```

`e1_calibration.py` must run first: everything else reads the frozen estimator configuration
from `results/e1_calibration/frozen_config.json`.  Results in `results/`, figures in
`figures/`, the writeup in [`report.md`](report.md).  Timings are for 12 of 16 cores.

| file | what it answers |
| --- | --- |
| `e0_atlas.py` | is r identifiable at all?  three generator classes, one scalar observer |
| `e6_tau_sensitivity.py` | how much of "MG saturates" is MG and how much is the delay lag |
| `e1_calibration.py` | the estimator's five free parameters, chosen once and frozen |
| `e2_rank_sweep.py` | the main test: k = 10 available, r = 1..8, seven excitation arms |
| `e3_transitions.py` | r_high -> r_low -> r_high: level error, detection rate, lag |
| `e4_controls.py` | seven things that change while r does not |
| `e5_real_logs.py` | where this project's own 120k-step logs sit on the atlas |

**The answer, in one line.** MG recovers the active dimension of a real network with MAE 0.84
on held-out r and held-out seeds -- but only for deterministic, recurrent dynamics whose
oscillation period the delay lag has been matched to.  Mini-batch gradient noise of rank r
gives 15.1 at every r from 1 to 8; a decaying transient gives 29 when the true active
dimension is 1; and all seven of this project's real logs sit outside the identifiable
regime.  See [`report.md`](report.md) section 7.

## What the design had to fix first

Three audits of `../dimension_recovery/exp10`-`exp15` were run before any new code, and each
found a defect that this package is built to avoid. They are recorded here because they are
the reason for design choices that would otherwise look paranoid.

* **`exp13`/`exp14` do not measure the optimiser.** Re-running `exp14` at its winning
  configuration with the learning rate multiplied by **zero** -- no training at all --
  reproduces the headline (`gradient_fro` MAE 2.06 -> 1.87, Spearman unchanged at +0.98,
  series correlation 0.986-0.998 at every k). The observers read the exogenous drive through
  the residual. Every arm here is therefore also run with `eta_zero=True`, and every observer
  except `loss_step` is a function of the optimiser state alone.
* **`exp10`-`exp12` tune the answer.** Their calibration grid contained *system* parameters
  (`cycles_per_window`; the learning rate in exp11/12) alongside estimator parameters, and
  the objective was error against the known k on the grid that was then reported. MG at k=20
  moves 5.3 -> 16.3 across that grid and the winner sits at its boundary. Here the grid
  contains estimator parameters only, one simulation per (seed, r) is scored by every
  configuration, and the split is disjoint in seed **and** in r -- necessary because
  `systems.py` ignores its rng under `band_mode="matched"`, so the frequency geometry was
  bit-identical across the old "held-out" seeds.
* **The nulls beat MG and were never reported.** On the exp10 signals the *linear*
  participation ratio of the delay matrix recovers k better than MG (MAE 1.03-1.23 vs
  1.29-1.46, rho 0.994-1.000 vs 0.949-0.985), and `roughness` -- one line, no embedding --
  reaches |rho| 0.85-0.93 against MG's 0.92-0.94 in exp13/14. `mg.all_estimators` returns
  both on every call so no result here can omit them.
* **`exp15` v1 and v2 fail for identifiable reasons.** v1's Hessian is exactly `I/N`, so all
  modes decay at one rate and the trajectory is a straight line: PR = 1.0000 at every k, and
  MG of a pure linear ramp at that configuration is 1.330 -- the measured value to three
  decimals. v2 used `theiler=0` (MG 1.23 vs 9.58 with the Theiler window on) and frequencies
  in exact arithmetic progression (`f0 - 2 f1 + f2 = 5e-20`), which collapses the torus to
  two dimensions. v3 is *not* broken -- held-out MAE 0.504, rho 0.988, trajectory PR = k
  verified -- but its r is injected by an external teacher the network is slaved to.

## What the estimator does when there is nothing to find

Two defects in `../grokking_analysis/edm/dimension.py`, both silent:

* an exactly recurrent series (a rational frequency ratio, a flat window) drives delay
  vectors together; the 1e-8 distance floor and the 1e-5 log-ratio-sum floor then return
  **0.08** or **n(k-1)-1 = 399975** rather than `nan`. `mg.all_estimators` detects both and
  sets `degenerate`.
* `clamp_to_max_E` returns `max_E` whenever the raw estimate exceeds `2*max_E`, turning a
  divergent estimate into a plausible number. It is off here. (It did *not* bind in exp10-15
  -- checked -- so it explains none of their results.)
