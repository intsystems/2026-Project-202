# gromov_polynomials

A reproduction of D. Doshi, T. He, A. Das, A. Gromov, *Grokking modular polynomials*
([arXiv:2406.03495](https://arxiv.org/abs/2406.03495)), on the same network as
[`../gromov_arithmetic/`](../gromov_arithmetic/) -- literally the same, imported through
[`_core.py`](_core.py) rather than copied, so the two folders cannot drift apart.

## Why this paper is the useful one

The arithmetic paper gives one grokking curve and one closed-form solution. This one
gives a **criterion**, and with it a set of matched pairs. Hypothesis 5.1: the two-layer
MLP reaches 100% test accuracy exactly when the target can be written

```
P(n1, n2) = h( g1(n1) + g2(n2) )  mod p                              (Eq. 14)
```

Appendix C then trains six polynomials, each in two versions: the bare polynomial, which
has that form, and the same polynomial plus one extra low-order monomial, which does not.

| polynomial | test acc | perturbed | test acc |
| --- | --- | --- | --- |
| `(4 n1 + n2^2)^3` mod 97 | 100% | `+ n1 n2` | 2.27% |
| `(2 n1 + 3 n2)^4` mod 97 | 100% | `- n1^2` | 3.93% |
| `(5 n1^3 + 2 n2^4)^2` mod 97 | 100% | `- n2` | 72.32% |
| `(4 n1 + n2^2)^3` mod 23 | 100% | `+ n1 n2` | 1.89% |
| `(2 n1 + 3 n2)^4` mod 23 | 100% | `- n1^2` | 7.17% |
| `(5 n1^3 + 2 n2^4)^2` mod 23 | 100% | `- n2` | 2.64% |

Both members of a pair are trained identically, both reach 100% *training* accuracy, and
they differ by one monomial. So anything that separates them in a log is a property of
the solution the network found, not of the data budget, the optimiser, or how hard the
fit was. For a dimension estimator that is the cleanest available test: a grokking run
and a non-grokking run that are matched on everything else.

The 72.32% entry is the paper's own outlier and is not smoothed over here; it is
reproduced and reported as measured.

## Two arms

`runs_poly.py` registers each polynomial twice.

**`f_*` -- paper-faithful.** Adam, `lr = 5e-3`, **`weight_decay = 5.0`**, `N = 5000`,
`alpha = 0.5`, exactly App. C. Note that this paper, unlike its predecessor, *does*
regularise, and heavily. **This arm is invalid as configured and is retained as
documentation** -- `weight_decay = 5.0` is a Doshi-parametrisation number, and in the
parametrisation used here it drives the weights to zero before the task gradient is ever
felt. See `report.md` Result 4.

**`g_*` -- Gromov no-weight-decay.** Full-batch GD, `wd = 0`, `N = 500`, the setup of
arXiv:2301.02679 Sec. 2. This is the arm the dimension analysis wants: at `wd = 5.0` the
weight norm is being dragged by the regulariser, so any trajectory statistic read off it
is partly a statistic of the regulariser rather than of learning.

The training fraction differs by modulus, and deliberately. Sec. 4 of the arithmetic
paper says the critical fraction `alpha_c` *grows* as `p` shrinks, and a direct check
here (`add mod 23`, N=200, GD, 60 000 steps) bracketed it for p=23 between 0.5 and 0.7:
at `alpha = 0.5` test accuracy never left chance, at `alpha = 0.7` it grokked at step
750. Running p=23 at `alpha = 0.5` under plain GD would produce a non-grokking log for a
reason that has nothing to do with the polynomial -- and would make the learnable and
perturbed arms look alike for the wrong reason. So `g_*` uses `alpha = 0.5` at p=97 (the
paper's value, well above `alpha_c ~ 0.29`) and `alpha = 0.8` at p=23.

## Chance is not always 1/p

`polynomials.py:distinct_outputs` reports the size of each polynomial's image, because
several of these targets are far from surjective: `(2 n1 + 3 n2)^4 mod 97` only ever
lands on fourth powers, of which there are 25. A constant predictor therefore scores
well above `1/p`, and "3.93%" has to be read against the right baseline. `compare.py`
uses the majority-class share, not `1/p`, when it calls a run chance-level.

## Running

Same driver as the arithmetic folder -- it bundles both, since they share the core:

```powershell
cd ..\gromov_arithmetic
.\colab_gromov.ps1 -Sync
.\colab_gromov.ps1 -Job .\jobs\03_poly.json
.\colab_gromov.ps1 -Poll
.\colab_gromov.ps1 -Fetch .\results
```

Directly:

```powershell
python run_poly.py pairs97 --outdir .\results      # the six no-wd p=97 runs
python run_poly.py faithful97 --outdir .\results   # the six App. C runs
python analytic_poly.py                            # representability, without training
```

## Outputs

Identical in shape to `../gromov_arithmetic/`: `<key>_train.csv` with
`step, train_loss, val_loss, train_acc, val_acc, weight_norm`, plus `<key>_obs.csv`
(Fourier IPR, effective rank, top singular values) and `<key>_snapshots.npz`.
`summary.json` additionally records each run's `paper_test_acc`, so the reproduction gap
is a number in the file rather than something to eyeball.

## Files

| path | role |
| --- | --- |
| [`polynomials.py`](polynomials.py) | Table 2's six polynomials, their images, the paper's reported accuracies |
| [`runs_poly.py`](runs_poly.py) | the registry: `f_*` faithful, `g_*` no-weight-decay |
| [`run_poly.py`](run_poly.py) | train registered runs, write the logs |
| [`analytic_poly.py`](analytic_poly.py) | representability without training, and the decomposition test |
| [`compare.py`](compare.py) | the runs against Table 2, scored per arm against the right baseline |
| [`_core.py`](_core.py) | the single import path back to `../gromov_arithmetic/gromov.py` |
| [`analytic_check.md`](analytic_check.md) | the decision procedure and its verdicts |
| [`report.md`](report.md) | what was run and what came out |

## Headline

Under Gromov's no-weight-decay setup, five of the six p=97 runs agree cleanly with
Hypothesis 5.1 and the sixth -- `(5 n1^3 + 2 n2^4)^2 - n2`, the paper's own unexplained
72.32% -- reproduces at 73.1% while being *provably* outside the representable class.
The criterion therefore does not depend on the regulariser, and the outlier is a
property of the target rather than of the optimiser.
