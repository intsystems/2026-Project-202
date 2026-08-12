# Frobenius-norm dimension recovery, k=1..10

## Protocol

One fixed MG configuration is selected on seed 0. Seeds 1 and 2 and the non-monotone transition schedule are held out. No per-k tuning is allowed.

Selected configuration: `window=4000, cycles/window=1000, E=25, kNN=30, tau=4`.

## Stationary validation

| seed | split | MAE | max error | Spearman | inversions |
|---:|---|---:|---:|---:|---:|
| 0 | calibration | 0.397 | 1.383 | 0.988 | 1 |
| 1 | held_out | 0.302 | 1.353 | 1.000 | 0 |
| 2 | held_out | 0.693 | 1.664 | 0.988 | 1 |

Held-out median by true dimension:

| true k | MG | error |
|---:|---:|---:|
| 1 | 1.104 | +0.104 |
| 2 | 2.117 | +0.117 |
| 3 | 3.208 | +0.208 |
| 4 | 4.556 | +0.556 |
| 5 | 5.211 | +0.211 |
| 6 | 5.750 | -0.250 |
| 7 | 7.198 | +0.198 |
| 8 | 7.693 | -0.307 |
| 9 | 8.185 | -0.815 |
| 10 | 8.491 | -1.509 |

## Held-out zigzag schedule

Schedule: `2 -> 7 -> 4 -> 9 -> 3 -> 10 -> 6 -> 1 -> 8 -> 5 -> 2`.

| segment | true k | median MG | error | windows |
|---:|---:|---:|---:|---:|
| 0 | 2 | 2.166 | +0.166 | 5 |
| 1 | 7 | 8.296 | +1.296 | 5 |
| 2 | 4 | 4.973 | +0.973 | 5 |
| 3 | 9 | 8.870 | -0.130 | 5 |
| 4 | 3 | 3.133 | +0.133 | 5 |
| 5 | 10 | 8.752 | -1.248 | 5 |
| 6 | 6 | 6.515 | +0.515 | 5 |
| 7 | 1 | 1.104 | +0.104 | 5 |
| 8 | 8 | 7.946 | -0.054 | 5 |
| 9 | 5 | 5.504 | +0.504 | 5 |
| 10 | 2 | 2.165 | +0.165 | 5 |

## Verdict

Zigzag MAE: **0.481 components**.

The ±0.5-component target is **not** met on every held-out seed. The experiment therefore does not establish exact absolute recovery; the residual error must be reported rather than hidden by rounding.

The selected point is a calibration result for this torus family. It does not by itself transfer to neural-network training logs, whose coverage is unknown.

Runtime: 95.4 seconds. Grid points: 48.