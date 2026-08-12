# Full-batch descent at the edge of stability

Does an undriven training run ever occupy the admissible regime — deterministic and recurrent —
that the article's estimator needs? This is the experiment the article's conclusion previously
named as future work.

Reproduce:

```bash
# training (Colab T4, ~30 min for the grid)
python eos.py --lrs 1e5 3e5 1e6 1.5e6 2e6 2.5e6 2.8e6 3e6 --seeds 1 2 --steps 30000 \
              --sharp-every 100 --outdir ./results/eos
# analysis (local CPU)
python eos_recurrence.py --results ./results/eos           # minutes
python eos_probe.py --results ./results/eos --columns train_loss weight_norm \
                    --subsample 1 10 50 --taus 4 1 --workers 4     # ~1 h
```

## What was run

Full-batch gradient descent, no weight decay, on the quadratic perceptron of `gromov.py`: modular
addition at p = 97, width 500, α = 0.5, 30 000 steps, eight learning rates by two seeds.

Two departures from every other full-batch log in this repository are deliberate.

* **`log_every=1`.** Edge-of-stability oscillation is the two-cycle of the unstable mode, so its
  period is a few optimiser steps. Every other full-batch log here is written at a stride of 10 or
  more, which does not blur that oscillation but removes it. `eos_probe.py` re-reads each log at
  strides 1, 10 and 50 to show exactly how much.
* **Sharpness along the run.** Power iteration on Hessian-vector products, taking the largest
  *algebraic* eigenvalue and not the largest in magnitude. The Hessian is indefinite early in
  training, when the most negative direction is the steeper, and plain power iteration then returns
  a negative sharpness; a shifted second pass fixes it. Verified against an exact
  eigendecomposition on a 120-parameter model: 5e-12 relative at a tight tolerance, 6e-5 at the
  30-iteration setting used here.

## The edge is reached

`eos_runs.csv`. The stability ratio is η·λ_max/2 over the second half of the run.

| η | stability ratio (s1, s2) | rises | outcome |
| --- | --- | --- | --- |
| 1e5 | 0.238, 0.228 | 0.000 | monotone, sharpness landscape-limited |
| 3e5 | 0.749, 0.707 | 0.000 | monotone |
| 1e6 | 0.978, 0.956 | 0.450, 0.500 | at the edge |
| 1.5e6 | 0.966, 0.965 | 0.500, 0.500 | at the edge |
| 2e6 | 0.976, 0.976 | 0.500, 0.500 | at the edge |
| 2.5e6 | 0.989, — | 0.500, — | at the edge; one seed diverges |
| 2.8e6, 3e6 | — | — | diverges |

Below about 4e5 the sharpness saturates at a value the *landscape* sets, ~4.6e-6, independent of η,
so η·λ/2 rises with η and the loss is monotone to the last step. From 1e6 the sharpness instead
self-limits at 2/η and the loss rises on essentially half of all steps. That is progressive
sharpening followed by the edge of stability, in a deterministic undriven run.

**Not round-off.** The largest single-step rise is 0.99 to 4.1 times the loss itself, against a
float32 relative epsilon of 1.2e-7.

## But oscillation is not recurrence

`eos_recurrence.csv`. For each reconstructed point, the distance to its nearest neighbour surviving
a 150-sample Theiler exclusion, divided by how far the orbit travels during that exclusion. Small =
the orbit came back; ~1 = the same curve continuing. Two controls fix the scale.

| | nn/travel |
| --- | --- |
| 2-torus control | **0.008** |
| monotone-decay control | **1.007** |
| below the edge (1e5, 3e5) | 1.006–1.007 |
| at the edge | 0.026, 0.065, and 1.008–1.013 |

The below-edge runs match the monotone control to four decimals. Among the seven surviving
edge-of-stability runs, **two recur and five do not**, and the split is unambiguous — the values
are bimodal with an order of magnitude of empty space between them, so any threshold in
(0.07, 1.0) gives the same partition.

Dividing by the cloud radius instead is the obvious normalisation and is wrong: a trajectory that
merely moves slowly also has close surviving neighbours without ever returning. `nn_over_scale` is
recorded beside it to make that visible.

## The diagnostics do not find the difference

`eos_diagnostics.csv`. On the **training loss**, post-transition, every step, τ = 1, the article's
admissible rectangle (ρ_ident ≤ 1.10 and more than eight trend crossings):

| | admitted | refused |
| --- | --- | --- |
| recurs (2 runs) | 0 | **2** |
| does not recur (5 runs) | **5** | 0 |

A clean inversion. The estimate itself agrees with the recurrence measurement and not with the
rectangle: **3.49 and 2.44** in the two runs that return, against **29.2–30.8** — the transient
pathology value — in the five that do not.

The cause is in the inputs. Every edge-of-stability run has a rise fraction of 0.50, so the
trend-crossing count, which tests non-monotonicity and *not* recurrence, passes all of them;
ρ_ident then happens to sit at unity in the five and at 1.23 and 1.30 in the two. On the
**parameter norm** the rectangle admits all seven, so the verdict is observer-dependent too.

Seven runs, one configuration, one task — no rule should be drawn from it. But it is the failure
mode the article's own section on diagnostics names, occurring.

## And the logging stride hides all of it

Training loss, post-transition, τ = 1, over the seven edge-of-stability runs:

| stride | median crossings | median rises | median MG | admissible |
| --- | --- | --- | --- | --- |
| 1 | 80 | 0.4999 | 29.2 | 5 of 7 |
| 10 | 2 | 0.0000 | 17.0 | 0 of 7 |
| 50 | 2 | 0.0000 | 4.9 | 1 of 7 |

At the stride every other full-batch log in this repository uses, the median trend-crossing count
falls from 80 to 2 and the rise fraction to exactly zero. The published protocol would have
classified every one of these runs as a monotone transient. Whether a log looks like a transient is
in part a property of how often it was written.

## Loose ends

* Rates 5e5 and 7e5, and a float64 replication, were queued as job `07_eos_fill.json` and lost when
  the Colab session's name mapping was destroyed (the OAuth token expired mid-campaign and a
  subsequent poll tried to recreate a live session). Neither is load-bearing: the stability
  threshold is bracketed to within a factor of two by 3e5 and 1e6, and the round-off question is
  settled by the rise amplitudes above. Re-running them would sharpen the threshold only.
* `nn_over_travel` is our own statistic, not one of the article's two diagnostics. It is reported,
  not adopted: it is calibrated on the same kind of system that produced the other two and has not
  been validated out of sample.
