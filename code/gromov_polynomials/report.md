# Reproducing *Grokking modular polynomials*

D. Doshi, T. He, A. Das, A. Gromov, [arXiv:2406.03495](https://arxiv.org/abs/2406.03495).
Runs executed on a Colab T4 on 2026-08-09, on the network of
[`../gromov_arithmetic/`](../gromov_arithmetic/) -- imported, not copied, so the two
reproductions cannot diverge. Method and calibration: [`../gromov_arithmetic/report.md`](../gromov_arithmetic/report.md).

## Why these runs and not more grokking curves

The value here is not a second grokking demonstration. It is **matched pairs**.
Appendix C trains six polynomials in two versions each: the bare polynomial, which has
the form `h(g1(n1) + g2(n2)) mod p` that Hypothesis 5.1 says the architecture can
represent, and the same polynomial plus one extra low-order monomial, which does not.
Both versions reach 100% training accuracy under identical optimisation on identical
data budgets; only one generalises.

For the dimension work this is the cleanest available control. A grokking run and a
non-grokking run that differ by `+ n1 n2` in the label function are matched on
architecture, width, optimiser, learning rate, training fraction, step budget, and
difficulty-of-fit. Anything that separates their logs is a property of the solution
found, not of anything else.

## Two arms, and why the second one exists

**`f_*` paper-faithful.** Adam, `lr = 5e-3`, `weight_decay = 5.0`, `N = 5000`,
`alpha = 0.5`, full batch, MSE -- App. C verbatim. Note that this paper, unlike its
predecessor, regularises heavily; "Gromov's no-weight-decay setup" is the *arithmetic*
paper's, not this one's. The step budget is not stated in the paper; 8 000 was used,
which is 20x the 400 epochs of Fig. 2 and 130x the ~60 of Fig. 3.

**`g_*` Gromov no-weight-decay.** Full-batch GD, `wd = 0`, `N = 500`, the setup of
arXiv:2301.02679 Sec. 2, 100 000 steps logged every 10. This is the arm the dimension
analysis wants: at `wd = 5.0` the weight norm is being pulled by the regulariser, so any
trajectory statistic read off it is partly a statistic of the regulariser.

Two parameters had to be set rather than copied.

*The learning rate* follows `gd_lr(p, N) = 1e5 (p/97)^3 (N/500)`, derived from the
initial gradient scale `2/(p^3 N)` of the mean-field parametrisation and calibrated at
p=97 by the sweep in the companion report. Without the `p^3` the p=23 runs diverge on
the first step -- the gradient there is 75x larger.

*The training fraction* is 0.5 at p=97 (the paper's value, well clear of
`alpha_c ~ 0.29`) but **0.8 at p=23**. Sec. 4 of the arithmetic paper says `alpha_c`
grows as `p` shrinks, and a direct check here -- `add mod 23`, N=200, GD, 60 000 steps,
measured with `../gromov_arithmetic/alpha_sweep.py` -- brackets it between 0.5 and 0.7.
At `alpha = 0.5` test accuracy ends at 4.91% and 8.30% for the two rates tried, against
a 4.35% chance level; at `alpha = 0.7` both rates reach 100%, grokking at steps 750 and
300. The sweep is in
`../gromov_arithmetic/results/sweeps/alpha_sweep_add_p23_N200.csv`. Running p=23 at
`alpha = 0.5` under plain GD would have produced non-grokking logs for a reason that has
nothing to do with the polynomial, making the learnable and perturbed arms look alike
for entirely the wrong reason. The paper avoids this because Adam with
`wd = 5.0` groks at a much smaller `alpha`.

## Chance is not 1/p

Three of these six targets are far from surjective, and reading their failure
accuracies against `1/p` overstates them. `polynomials.py:distinct_outputs` reports the
image size; `compare.py` uses the majority-class share as the chance baseline.

| p | polynomial | image size | majority class | paper's reported acc |
| ---: | --- | ---: | ---: | ---: |
| 97 | `(2 n1 + 3 n2)^4` | 25 / 97 | 4.1% | 100% |
| 97 | `(4 n1 + n2^2)^3 + n1 n2` | 96 / 97 | 1.2% | 2.27% |
| 97 | `(2 n1 + 3 n2)^4 - n1^2` | 97 / 97 | 2.1% | 3.93% |
| 97 | `(5 n1^3 + 2 n2^4)^2 - n2` | 97 / 97 | 1.3% | 72.32% |
| 23 | `(2 n1 + 3 n2)^4 - n1^2` | 23 / 23 | 8.5% | 7.17% |
| 23 | `(5 n1^3 + 2 n2^4)^2 - n2` | 23 / 23 | 4.3% | 2.64% |

Read against the right baseline, the two p=23 failures the paper reports at 7.17% and
2.64% are **at or below chance**, not slightly above it. The `72.32%` for
`(5 n1^3 + 2 n2^4)^2 - n2 mod 97` is the only entry that is genuinely far above its
baseline, and the paper does not comment on it. It reproduces here (below).

## Result 1: the criterion holds without any regularisation

Full-batch GD, `wd = 0`, N=500, 100 000 steps; alpha 0.5 at p=97 and 0.8 at p=23. The
paper only demonstrates this split with Adam at `weight_decay = 5.0`.

| p | polynomial | form | paper | ours | chance | grokked at | verdict |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 97 | `(4 n1 + n2^2)^3` | `h(g1+g2)` | 100% | **100%** | 3.1% | 12 560 | generalised |
| 97 | `(4 n1 + n2^2)^3 + n1 n2` | perturbed | 2.3% | 1.2% | 1.2% | never | chance |
| 97 | `(2 n1 + 3 n2)^4` | `h(g1+g2)` | 100% | **100%** | 4.1% | 7 750 | generalised |
| 97 | `(2 n1 + 3 n2)^4 - n1^2` | perturbed | 3.9% | 1.9% | 2.1% | never | chance |
| 97 | `(5 n1^3 + 2 n2^4)^2` | `h(g1+g2)` | 100% | **100%** | 3.1% | 6 430 | generalised |
| 97 | `(5 n1^3 + 2 n2^4)^2 - n2` | perturbed | 72.3% | **73.1%** | 1.3% | never | partial |
| 23 | `(4 n1 + n2^2)^3` | `h(g1+g2)` | 100% | **100%** | 4.3% | 7 630 | generalised |
| 23 | `(4 n1 + n2^2)^3 + n1 n2` | perturbed | 1.9% | 2.8% | 6.0% | never | chance |
| 23 | `(2 n1 + 3 n2)^4` | `h(g1+g2)` | 100% | **100%** | 8.7% | 5 160 | generalised |
| 23 | `(2 n1 + 3 n2)^4 - n1^2` | perturbed | 7.2% | 7.5% | 8.5% | never | chance |
| 23 | `(5 n1^3 + 2 n2^4)^2` | `h(g1+g2)` | 100% | **100%** | 8.7% | 4 780 | generalised |
| 23 | `(5 n1^3 + 2 n2^4)^2 - n2` | perturbed | 2.6% | 0.9% | 4.3% | never | chance |

**Eleven of the twelve agree cleanly with Hypothesis 5.1; the twelfth is the paper's own
outlier and agrees with neither side of it.** Four things follow.

The criterion is not an artefact of the regulariser. Weight decay of 5.0 is a strong
prior toward sparse, low-norm solutions, and it would have been reasonable to suspect
the clean split depends on it. It does not: plain GD from `N(0,1)` with no decay at all
reproduces every entry.

The 72.32% outlier reproduces at **73.1%** under a completely different optimiser
(GD/wd=0/N=500 against Adam/wd=5.0/N=5000). A number that survives that change of
optimiser is a property of the target, not of the training.

And that outlier is a genuine counterexample to the hypothesis as stated, not a near
miss. `(5 n1^3 + 2 n2^4)^2 - n2` is **provably** not of the form `h(g1 + g2)` (Result 2),
yet it reaches 73% test accuracy against a 1.3% baseline. Hypothesis 5.1 predicts 100%
for decomposable targets and says nothing about the rest, so 73% is outside what it
accounts for. `compare.py` scores it as `partial` rather than as agreement, because
counting it as a success would hide the one entry in the table the hypothesis does not
explain. What it suggests is that representability is not binary -- a target one
monomial away from the class can be approximated well -- and neither paper has a
quantity for "how far outside" a target sits.

Finally, **the outlier is specific to p = 97**. The same polynomial at p = 23 lands at
0.9% against a 4.3% baseline -- below chance, the flattest failure in the whole table --
and the paper reports 2.64% there. So whatever makes `- n2` cheap to approximate at
p = 97 is a property of that modulus, not of the perturbation. The obvious candidate is
the image size: `(5 n1^3 + 2 n2^4)^2 mod 97` takes 49 of 97 values while at p = 23 it
takes 12 of 23, but the perturbed versions are surjective at both, so that alone does not
explain it. Not pursued further here.

## Result 2: the perturbed targets are provably outside the class

`analytic_poly.py` settles representability without training, by two exact arguments
rather than a search (details and validation in [`analytic_check.md`](analytic_check.md)):

* the three learnable polynomials are **representable at 100% with zero training** at
  both p=23 and p=97 once N >= 500, by the periodic construction with the stated
  `(g1, g2, h)` checked on all `p^2` entries;
* the three perturbed polynomials admit **no `h(g1(n1) + g2(n2))` decomposition at all**,
  proved at both moduli by a multiset certificate: after deduplicating rows and columns
  the table has `p` distinct columns, so `g2` must be a bijection of `Z_p` and every row
  must carry the same multiset of values; the rows do not.

So the failure to grok is not the optimiser running out of time, the data budget, or the
width. The target is outside the class the architecture can express. That is a stronger
statement than the paper's, which rests on the empirical accuracies of Table 2.

## Result 3: what separates the pairs in the logs

This is the part the dimension analysis is for. Final values, p=97, no-weight-decay arm:

| run | generalises | train MSE | val MSE | \|W\| | ipr_u1 | ipr_u2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `g_p1_p97` | yes | 0.0000 | 0.0003 | 5.35 | 0.243 | 0.055 |
| `g_p1x_p97` | no | 0.0040 | 0.0224 | 6.16 | 0.040 | 0.040 |
| `g_p2_p97` | yes | 0.0000 | 0.0001 | 4.35 | 0.283 | 0.283 |
| `g_p2x_p97` | no | 0.0036 | 0.0217 | 6.16 | 0.043 | 0.044 |
| `g_p3_p97` | yes | 0.0000 | 0.0000 | 4.53 | 0.044 | 0.054 |
| `g_p3x_p97` | no | 0.0024 | 0.0118 | 6.50 | 0.039 | 0.040 |

**The weight norm separates the two classes perfectly** -- 4.35-5.35 against 6.16-6.50,
with no overlap, and the generalising values bracket the analytic solutions' own norms
(5.170, 5.193, 5.179 for the three polynomials). With `wd = 0` this is not the
regulariser talking.

**The training loss separates them too, and that is the more interesting one.** All six
runs reach 100% *training accuracy*, but only the three representable ones drive the
training MSE to zero. The perturbed runs fall to 2.4e-3 to 4.0e-3 and effectively stop:
each is within 10% of its final value by step 36 610, 42 470 and 48 300 respectively,
and moves by less than that over the remaining half of the run. They fit the argmax and
cannot fit the regression, which is exactly what "the target is outside the representable
class" predicts -- and it is visible in the standard training log, without any
weight-space probe.

**The Fourier IPR is only a valid order parameter when `g1` and `g2` are linear.** This
is a caveat, and a sharp one:

| polynomial | g1 | measured ipr_u1 | **its own analytic ipr_u1** | groks? |
| --- | --- | ---: | ---: | --- |
| `(4 n1 + n2^2)^3` | `4 n1` linear | 0.243 | 1.000 | yes |
| `(2 n1 + 3 n2)^4` | `2 n1` linear | 0.283 | 1.000 | yes |
| `(5 n1^3 + 2 n2^4)^2` | `5 n1^3` nonlinear | 0.044 | **0.062** | **yes** |

The third row is the one that matters, and it took a bug fix to read correctly. Every
polynomial run was initially compared against the *modular-addition* reference of 1.000,
which made `(5 n1^3 + 2 n2^4)^2` look like a total failure to form a periodic
representation despite 100% test accuracy. Its own closed-form solution has
`ipr_u1 = 0.062`, and the trained run reaches 0.044. It is not failing to become
periodic; **the ground truth itself is 0.062 in this basis.**

The reason is that the representation is periodic in `g1(n1) = 5 n1^3`, while the
spectrum is taken over the raw index `n1`, and `cos(2 pi k * 5 n1^3 / p)` is a chirp in
`n1` -- spectrally flat. The IPR of Doshi et al. (Eq. 3, Figs. 2c and 3c) is reported on
modular *addition* and *multiplication*, where `g` is linear or a discrete logarithm, so
the question never arises there.

Two consequences for this project. A spectral concentration measure read in the wrong
basis reports "nothing has been learned" about a network that has learned the task
perfectly -- and any dimension estimate taken in a fixed basis inherits that failure
mode. And an order parameter is only interpretable against *its own* ground truth: the
same measured 0.044 means "at chance" against 1.000 and "essentially converged" against
0.062. `analyze.py` now returns NaN rather than substituting another task's reference.

Note also that even where the basis is right, the trained networks reach only 0.24-0.28
against an analytic 1.000 at 100 000 steps. Generalisation does not require the
representation to be fully periodic; it requires it to be periodic enough.

The weight norm and the training MSE need no basis and separate all three pairs.

## Result 4: the paper-faithful arm does not survive the change of parametrisation

The `f_*` arm was launched with App. C's stated hyperparameters -- Adam, `lr = 5e-3`,
`weight_decay = 5.0`, `N = 5000` -- on top of the arithmetic paper's parametrisation,
and it **collapses to the zero function**:

```
=== f_p1_p97 ===
p1 mod 97 / N=5000 quadratic / adam lr=0.005 wd=5 / full batch / alpha=0.5 / 8000 steps
  step       0  train 1.031e-02/ 0.85%  val 1.031e-02/ 0.72%  |W|=1.001       0s
  step    5000  train 1.031e-02/ 0.77%  val 1.031e-02/ 0.79%  |W|=0.000     111s
[f_p1_p97] grokked at None, final val acc 1.25% (paper 100.00%), 183s
```

This is not a bug in the runner, and it is not evidence against the paper. It is the
two papers' conventions being incompatible at the stated numbers, and it is worth
recording because the failure is silent -- the loss simply sits at `1/p` forever.

In the arithmetic paper's parametrisation the weights start at `N(0,1)` and the forward
pass carries `1/(D N)`, so the task gradient at initialisation is of order
`2/(p^3 N) ~ 4e-10` at p=97, N=5000. The L2 term `torch.optim.Adam` adds is
`wd * w ~ 5`, ten orders of magnitude larger. Adam normalises by the gradient RMS, so
every update is essentially pure decay and the weights are driven to zero before the
task is ever seen. `weight_decay = 5.0` is only a sane number in the Doshi
parametrisation, where the scale is folded into the init (`std ~ 0.5/(2N)^(1/3)`), the
forward pass has no prefactor, and the output is O(1) at initialisation.

Reproducing the `f_*` arm faithfully therefore requires implementing the *second*
parametrisation as well, not merely copying its hyperparameters -- which is precisely
the hazard `../gromov_arithmetic/gromov.py` documents at the top and which this run
walked into. It was stopped after two runs rather than left to burn roughly half an hour
of GPU on twelve runs of the zero function -- the first took 183 s, and the second was
aborted at step 6 365, by which point its weight norm had reached 1.4e-04.

**This does not affect Results 1-3.** The `g_*` arm is internally consistent: its
learning rate was calibrated inside this parametrisation and its weight decay is zero,
so no cross-convention transfer occurs anywhere in it.

## Status of the run set

| arm | status |
| --- | --- |
| `g_*` no-weight-decay, p=97 (6 runs) | complete |
| `g_*` no-weight-decay, p=23 (6 runs) | complete |
| representability analysis, p=97 and p=23 | complete, see [`analytic_check.md`](analytic_check.md) |
| trajectory-rank runs on the `p1` pair, full and mini batch | complete, see the companion report |
| `f_*` paper-faithful (12 runs) | **invalid as configured** -- see Result 4; needs the Doshi parametrisation, not a re-run |

`results/summary.json` is rebuilt from the logs by
`../gromov_arithmetic/rebuild_summary.py`, so it lists everything that produced a
`_train.csv` -- including the two aborted `f_*` runs, which is why 14 records back a
12-run table.

## The trajectory-rank measurement on these pairs

The matched pairs were also run with `../active_rank/`'s trajectory sketch attached, to
test its one-dimensional-bottleneck result under controls it could not construct. The
outcome is reported in
[`../gromov_arithmetic/report.md`](../gromov_arithmetic/report.md): under full-batch GD
the participation ratio never leaves 1.00-1.36, so the statistic has no room to
collapse; with mini-batches it ranges over 1.03-18.9 but shows no dip at generalisation,
and the generalising member of a pair is not reliably deeper than its control.

## The dimension estimators on these pairs

`../gromov_arithmetic/dimension_probe.py` run on the six p=97 no-weight-decay logs,
median over sliding windows:

| pair | MG on `train_loss`, generalising | MG, perturbed | PRdelay | roughness |
| --- | ---: | ---: | ---: | ---: |
| `p1` | 19.42 | 22.58 | 1.0 | 0.001 |
| `p2` | 20.05 | 23.03 | 1.0 | 0.001 |
| `p3` | 19.29 | 22.83 | 1.0 | 0.001 |

MG is consistently ~3 units higher for the non-generalising member, in all three pairs,
with no overlap. That is a real and repeatable separation -- and it should **not** be
read as a dimension. `PRdelay` is exactly 1.0 in all six runs and `roughness` is
0.0003-0.001: full-batch GD has no mini-batch noise, so the traces are smooth and
near-monotone, the delay-embedding covariance is effectively rank one, and MG is sitting
at the ceiling set by `max_E = 20`. What separates the classes is almost certainly the
*shape* of the loss curve -- a plateau against a decay to zero -- not the dimension of
any attractor.

That makes these logs useful in a specific way: they are a case where the estimator
gives a clean, reproducible, and wrong-for-the-stated-reason answer. Validating a
scalar-trace estimator against a known truth on this architecture needs the mini-batch
variant; `batch_size` is a `Config` field and defaults to full batch only because that
is what the papers used.
