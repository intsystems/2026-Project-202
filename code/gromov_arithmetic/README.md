# gromov_arithmetic

A reproduction of A. Gromov, *Grokking modular arithmetic*
([arXiv:2301.02679](https://arxiv.org/abs/2301.02679)), built to produce training
logs for the intrinsic-dimension analysis in [`../active_dimension/`](../active_dimension/)
and [`../dimension_recovery/`](../dimension_recovery/).

The reason to reproduce this paper rather than reuse the transformer logs already in
[`../grokking_train/`](../grokking_train/): Gromov's network is a **two-layer MLP with a
quadratic activation and no regularisation of any kind**, and Sec. 3 gives its grokked
solution *in closed form*. That makes it the only setup in this project where the
dimension of the learned representation is known rather than estimated -- the periodic
solution of Eq. (6)-(7) uses `(p+1)/2` frequencies and nothing else. A dimension
estimator can therefore be scored against a ground truth instead of against intuition.

The companion folder [`../gromov_polynomials/`](../gromov_polynomials/) trains the
same network on the modular polynomials of arXiv:2406.03495, where some targets are
representable by that construction and some provably are not.

## The architecture

Eqs. (1)-(2), verbatim, no biases, mean-field second layer:

```
f(x) = 1/N * W2 phi( W1 x / sqrt(D) ) ,   W1: N x D,  W2: p x N,  D = 2p
```

with `phi(h) = h^2`, `W1, W2 ~ N(0,1)`, one-hot inputs concatenated, one-hot targets,
MSE mean-reduced over batch *and* class axis, full-batch gradient descent,
`weight_decay = 0`.

**The normalisation convention is load-bearing.** The paper puts `1/(D N)` in the
forward pass and leaves the init at `N(0,1)`; the reference implementation of Doshi et
al. does the opposite, folding the scale into the init. Both produce grokking, but they
are different loss landscapes and the usable learning rates differ by orders of
magnitude. This folder implements the paper's convention, and two independent numbers
confirm it is the one the paper's own figures were made with:

| quantity | paper, Fig. 0a | here |
| --- | --- | --- |
| initial train/test MSE | 0.0105 | `1/p` = 0.01031 at p=97 |
| initial normalised weight norm | 1.0 | 1.0 by construction |
| final normalised weight norm | ~3.7 | ~3.5 (measured, p=23) |

## The correctness gate

`analytic.py` builds the closed-form solution of Claim I/II and evaluates it without
training. It has no free scale: substituting Eqs. (6)-(7) into Eq. (4) fixes the
amplitude at `A = (2D)^(1/3)`, so if the forward pass disagreed with the paper by any
constant the peak of the output would not land on 1.

```
$ python analytic.py --p 97
task          N      acc         MSE     peak     |W|
add          50  88.31%   7.813e-02    0.934   5.124
add         100  99.72%   4.148e-02    1.042   5.162
add         500 100.00%   8.542e-03    1.003   5.154
sum_sq      500 100.00%   9.097e-03    1.033   5.176
```

Mean peak 1.003 and 100% accuracy from N≈100-200 upward, matching Fig. 3b's statement
that the analytic solution needs N≈90-100. Run this before trusting any training log.

One deviation from the paper, and it is an improvement rather than a discrepancy: for
`F(f1+f2)` with non-invertible `F`, Eq. (19) builds the readout from `F^{-1}` and
therefore picks one branch, which is why the paper measures ~51% on `(n+m)^2`. Writing
the readout as a *forward* map instead -- row `F(t)` accumulates the frequency content
of `t`, so every preimage contributes constructively to the same output index -- gives
100%. The architecture can represent `(n+m)^2 mod p` exactly; the 51% is a property of
that particular construction.

## The learning rate

**The paper states no learning rate anywhere** -- not in the text, not in a caption, not
in an appendix -- and its own figures are mutually inconsistent about the timescale
(Fig. 0 groks near 20 000 steps; Fig. 3a reports "time to grok" near 1 000 for the same
optimiser). So it has to be chosen, and `lr_sweep.py` chooses it visibly.

It has to be large. The `1/(D N)` prefactor pushes the initial gradient down to
`~2/(p^3 N)`, about 4e-9 at p=97, N=500 -- so useful rates are four to five orders of
magnitude above what an Adam-shaped intuition suggests, and the same setup needs a rate
about `187x` larger at p=97 than at p=23. The selected value and the sweep it came from
are in [`report.md`](report.md).

## Running

Everything runs on Google Colab from Windows via the CLI set up in
[`../prediction_improved/README.md`](../prediction_improved/README.md) -- the one-time
`uv` / `colab_auth.py` steps there are a prerequisite. `colab_gromov.ps1` differs from
`colab_job.ps1` in two ways: it bundles this folder together with
`../gromov_polynomials/` (they share one training core), and `-Job` launches a
*sequence* of commands under a single detached process, so polls and downloads are not
queued behind a foreground run.

```powershell
.\colab_gromov.ps1 -Sync                     # push both folders
.\colab_gromov.ps1 -Job .\jobs\01_sweep.json # launch detached, returns at once
.\colab_gromov.ps1 -Poll                     # cheap, repeatable, interleaves with training
.\colab_gromov.ps1 -Fetch .\results
.\colab_gromov.ps1 -Stop
```

Only one Colab assignment is available at a time; a second `new` returns
`TooManyAssignmentsError` while another session holds the runtime.

**Do not run two pollers at once.** The CLI keeps its name-to-runtime mapping in
`~/.config/colab-cli/sessions.json`, and `StateStore._load_raw` returns `{}` on *any*
read error -- so a read landing mid-write yields "no sessions", and the next write
persists that empty dict. The runtime keeps running, unreachable by name, holding the
GPU. It happened twice here. `colab_recover.py --list` shows what is really assigned,
`--adopt` puts a name back on it, and `--stop-all` frees the quota:

```powershell
$py = "$env:APPDATA\uv\tools\google-colab-cli\Scripts\python.exe"
& $py colab_recover.py --list
& $py colab_recover.py --adopt <endpoint> --name grank   # then fetch normally
& $py colab_recover.py --stop-all
```

Locally, for small `p`:

```powershell
python analytic.py --p 97
python run.py core --outdir .\results
python lr_sweep.py --p 23 --width 200 --steps 60000 --lrs 3e2 1e3 3e3 1e4 3e4
```

`p = 97, N = 500` costs ~2.2 ms/step on a T4 and ~190x more per step than
`p = 23, N = 200`; a 100 000-step run is under four minutes on the GPU and over five
hours on this CPU. Use the GPU.

## Outputs

Three files per run, in `--outdir`:

| file | contents |
| --- | --- |
| `<key>_train.csv` | `step, train_loss, val_loss, train_acc, val_acc, weight_norm` |
| `<key>_obs.csv` | Fourier IPR per layer, effective rank of each weight matrix, top singular values |
| `<key>_snapshots.npz` | 21 log-spaced full weight dumps, float32 |

`<key>_train.csv` is byte-compatible with `../dimension_recovery/results/extended/*_train.csv`
on purpose: the same column names in the same order, logged every 10 steps for 100 000
steps, so 10 000 samples -- the record length `../active_dimension/e1_calibration.py`
froze its window size against. A log produced here drops into `e5_real_logs.py` with no
adapter.

`<key>_obs.csv` carries what the standard columns cannot: `ipr_u1`/`ipr_u2`/`ipr_w` are
the mean inverse participation ratio of each weight block's Fourier spectrum, which is
1.0 for the analytic solution and ~1/p for random init, and `erank_w1`/`erank_w2` are
the linear participation ratio of the singular values. Together they give a *known-truth*
dimension signal to compare an estimator against.

Rows are flushed as they are produced. Colab reclaims free-tier VMs without warning, and
a CSV truncated at step 60 000 is still a usable trajectory.

## Files

| path | role |
| --- | --- |
| [`gromov.py`](gromov.py) | the architecture, the data, the training loop, the observables |
| [`tasks.py`](tasks.py) | the modular functions, grouped by what the paper says happens to them |
| [`analytic.py`](analytic.py) | the closed-form solution; the convention gate |
| [`lr_sweep.py`](lr_sweep.py) | calibrates the one hyperparameter the paper omits |
| [`alpha_sweep.py`](alpha_sweep.py) | brackets `alpha_c` at a given modulus |
| [`rebuild_summary.py`](rebuild_summary.py) | regenerate `summary.json` from the logs |
| [`verify_sketch.py`](verify_sketch.py) | proof that `PR ~ 1` is the trajectory, not the observer |
| [`runs.py`](runs.py) | the registry: `a_*` grok, `x_*` must not, `c_*`/`r_*` controls |
| [`run.py`](run.py) | train registered runs, write the logs |
| [`analyze.py`](analyze.py) | the summary tables, each run against its *own* analytic reference |
| [`dimension_probe.py`](dimension_probe.py) | bridge to `../active_dimension/mg.py` |
| [`rank.py`](rank.py) | the trajectory sketch, reusing `../active_rank`'s CountSketch |
| [`run_rank.py`](run_rank.py) | train the matched pairs with the sketch attached |
| [`rank_dip.py`](rank_dip.py) | the dip test, generalising run against its own matched control |
| [`pr_vs_window.py`](pr_vs_window.py) | the participation ratio against window length, 600 to 120 000 steps |
| [`verify_rank_noninvasive.py`](verify_rank_noninvasive.py) | proof the sketch leaves training bit-identical |
| [`colab_gromov.ps1`](colab_gromov.ps1) | sync / launch detached / poll / fetch / stop |
| [`colab_recover.py`](colab_recover.py) | re-adopt or terminate a runtime the CLI lost the name of |
| [`jobs/`](jobs/) | job sequences, one JSON list of argv per campaign |
| [`remote/`](remote/) | scripts executed on the VM |
| [`report.md`](report.md) | what was run and what came out |

## Applying the other analyses in this repo

The logs are shaped to be consumed by the existing pipelines rather than by new ones:

```powershell
python dimension_probe.py --results .\results\arith        # ../active_dimension/mg.py
python run_rank.py pairs --outdir .\results\rank           # ../active_rank's observer
python ..\active_rank\analyze_rank.py --indir .\results\rank --window 60 --stride 5
python rank_dip.py --indir .\results\rank
python pr_vs_window.py --indir .\results\rank_fb_long --check
```

`run_rank.py` takes `--set`, so the same pairs can be re-run under a different regime --
`--set batch_size=512` is the one that matters, and `report.md` explains why.
