# Known-active-dimension calibration on real data, k = 1..20

Write-up of the experiment in `../../calibration_k20.py` and `../../score_k20_parallel.py`.
It existed only as committed result files until this file was written; appendix C and tables 6
and 7 of `icomp_v2/report.tex` are computed from `scores_frozen.csv`, so the article has been
quoting an experiment with no write-up beside it. Two defects in this directory were found
while writing it and are recorded in the last section.

Everything below was recomputed from the committed CSVs. Nothing is quoted from memory.

`report_auto.md` beside this file is not a second write-up: it is a short summary that
`calibration_k20.py` emits on every run, kept because it is regenerated with the results and
so cannot drift from them. This file is the one to read.

## Design

A frozen tanh MLP backbone on the sklearn digits data, followed by a trainable adapter with
`k = 20` fixed orthonormal parameter directions, `theta = theta0 + V^T c`. The dynamics excite
`r` of those directions, `r = 1..20`. The **truth is measured, not assumed**: the active
dimension is the participation ratio of the trajectory covariance (`traj_pr`), computed from
the stored `C` matrix, and it is not equated to the label `r`. On the recurrent arm the two
agree to within 0.01 components up to `r = 19` and to 0.007 at `r = 20` (`traj_pr = 19.993`).

Arms, all at `T = 12000` steps with a 2000-step burn-in:

| tag | dynamics | ranks | seeds | rows in `scores_frozen.csv` |
| --- | --- | --- | --- | --- |
| `qp` | deterministic recurrent, incommensurate sinusoidal forcing | 1..20 | 0,1,2 | 480 |
| `qp_slow` | the same torus at the slow frequency band | 1,5,10,15,20 | 0 | 40 |
| `noise` | rank-`r` stochastic forcing | 1,5,10,15,20 | 0 | 40 |
| `batch_proj` | projected mini-batch noise | 1,5,10,15,20 | 0 | 40 |
| `gd` | full-batch transient descent, no drive, no preconditioner | 1,5,10,15,20 | 0 | 40 |
| `qp_scale2` | fixed-r control: constant gain on the observer fluctuation | 10 | 0 | 8 |
| `qp_rotate` | fixed-r control: fixed orthogonal rotation of the read-out basis | 10 | 0 | 8 |

Eight observers are scored: `w_fro`, `c_norm`, `g_fro`, `g_proj`, `c_proj1`, `fn_fro`,
`fn_proj1`, `loss_full`. Each series is standardised, `z = (x - mean)/sd`, before the
estimator sees it.

## The frozen estimator

Stage 1 selects one configuration on the calibration ranks `r = 2, 6, 10, 14, 18` and the four
observer families `w_fro, c_norm, g_fro, c_proj1` only. The grid is four points -- `max_E` in
{20, 40} crossed with `tau` in {4, 16}, at a window of 8000 fixed in advance. Selection is on
the **raw** absolute error against the measured `traj_pr`; an isotonic fit is deliberately not
used, because with five calibration ranks it interpolates any monotone curve exactly and makes
every configuration look identical (see the defect section -- that is not hypothetical here).

Median score by configuration, from `config_observer_ranking.csv`:

| cfg_id | max_E | tau | median score | selected |
| --- | --- | --- | --- | --- |
| 0 | 20 | 4 | 3.353 | |
| 1 | 20 | 16 | 2.778 | |
| 2 | 40 | 4 | 2.854 | |
| 3 | 40 | 16 | **2.472** | yes |

Frozen: `max_E = 40, tau = 16, k_neighbors = 20, theiler = "autocorr", window = 8000,
stride = 4000, dither = 1e-9` (`frozen_k20.json`, cfg_id 3).

Two properties of that configuration should be stated with it.

* **The requested exclusion is never applied.** The delay span is `(40 - 1) x 16 = 624`
  samples and the autocorrelation rule asks for at least that, but `mg.THEILER_CAP` truncates
  it to 150. Every one of the 656 rows in `scores_frozen.csv` records `theiler_used = 150.0`
  and `tau_used = 16.0`. The exclusion is therefore smaller than the embedding span at every
  point of this experiment, which is the caveat appendix N measures.
* **There is barely a sliding window.** At `window = 8000, stride = 4000` on a 10 000-sample
  post-burn-in record, `n_windows` is 1 or 2. The "median over sliding windows" convention is
  nearly vacuous here, and the spread columns (`MG_sd` etc.) are correspondingly
  uninformative.

No window in the experiment is degenerate: `frac_degenerate` is 0.0 on all 656 rows.

## Recovery on the recurrent arm

Median over three seeds and eight observers, against the measured `traj_pr`:

| true r | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 17 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MG | 0.89 | 2.35 | 4.41 | 6.66 | 7.38 | 8.99 | 10.61 | 11.40 | 13.50 | 15.09 |
| PR_delay (linear null) | 2.00 | 2.29 | 4.54 | 6.81 | 7.95 | 9.20 | 10.79 | 12.15 | 11.71 | 13.51 |
| spectral PR, 256 bands | 1.01 | 1.25 | 2.58 | 3.68 | 4.18 | 5.48 | 5.98 | 6.53 | 8.11 | 8.38 |
| spread of MG across seeds | 0.00 | 0.20 | 1.49 | 2.15 | 2.11 | 2.95 | 1.50 | 2.84 | 2.76 | 3.33 |

That is table 6 of the article, and `../../paper_tables.py` regenerates it from this directory
and checks it cell by cell.

Over all twenty ranks the estimate tracks the truth at Spearman `+0.973` with a mean absolute
error of `1.624` components, and it saturates: at `r = 20` the measured rank is `19.99` and MG
reads `15.09`. The tracking is not monotone in the fine structure -- MG reads 13.85 and 15.40 at
`r = 15, 16` and then falls back to 13.50 at `r = 17` -- and the across-seed spread reaches 3.33
components, so no single rank's value should be read to better than a few components.

**The estimator is not the best statistic here.** From `paper_tables.py`, over all twenty ranks:

| statistic | MAE | rho |
| --- | --- | --- |
| MG | 1.624 | +0.973 |
| LB | 1.358 | +0.983 |
| TwoNN | 3.484 | +0.889 |
| PR_delay (linear) | 2.185 | +0.979 |
| spectral PR, 256 bands | 5.286 | +0.998 |
| spectral PR, 1024 bands | 4.471 | +0.998 |
| spectral PR, native | 2.784 | +0.934 |
| roughness (null) | 9.895 | -0.568 |

The uncorrected Levina-Bickel average beats the MacKay-Ghahramani pooling on both criteria, and
the purely linear participation ratio of the delay matrix is within 0.6 components of MG. The
roughness null is anti-correlated, which is the one clean result: whatever MG is tracking, it
is not smoothness.

Per-observer held-out performance (`heldout_qp_summary.csv`, test ranks only, after an affine
calibration fitted on the calibration ranks):

| observer | rho | MAE (calibrated) | max error |
| --- | --- | --- | --- |
| c_norm | 0.948 | 1.406 | 5.160 |
| w_fro | 0.948 | 1.406 | 5.160 |
| g_fro | 0.950 | 1.532 | 5.919 |
| g_proj | 0.939 | 1.533 | 6.566 |
| loss_full | 0.904 | 1.926 | 8.152 |
| c_proj1 | 0.909 | 1.950 | 6.599 |
| fn_proj1 | 0.918 | 2.019 | 11.154 |
| fn_fro | 0.902 | 2.136 | 5.855 |

## The other arms

Median MG over the eight observers, against the measured rank:

| arm | r = 1 | 5 | 10 | 15 | 20 |
| --- | --- | --- | --- | --- | --- |
| `qp` truth | 1.00 | 5.00 | 10.00 | 15.00 | 19.99 |
| `qp` MG | 0.89 | 5.51 | 8.99 | 13.85 | 15.09 |
| `qp_slow` truth | 1.00 | 4.99 | 9.81 | 12.97 | 17.27 |
| `qp_slow` MG | 1.14 | 9.04 | 9.96 | 13.51 | 15.80 |
| `noise` truth | 1.00 | 4.98 | 9.93 | 14.84 | 19.74 |
| `noise` MG | 26.38 | 26.74 | 26.79 | 27.26 | 26.42 |
| `batch_proj` truth | 1.00 | 4.02 | 5.97 | 8.41 | 9.43 |
| `batch_proj` MG | 27.28 | 26.41 | 26.77 | 26.63 | 26.62 |
| `gd` truth | 1.02 | 1.04 | 1.09 | 1.03 | 1.04 |
| `gd` MG | 29.09 | 27.55 | 27.89 | 28.80 | 27.88 |

The three non-recurrent arms are flat in the rank at 26-29 components regardless of what was
injected, which is the intended negative result: outside the deterministic recurrent regime the
number is not a count of anything, and the article's admissibility diagnostics exist to refuse
exactly these. Note that `gd`'s measured active dimension is about 1 at every `r` -- a decaying
transient revisits nothing -- while MG reads 28. The value 26-29 is read at the exclusion cap of
150 and is not a level; see appendix P.

## The invariance controls

Both controls hold `r = 10`, seed 0, and are compared observer by observer against
`qp, r = 10, seed 0` -- the same seed, not the three-seed median. `invariance_controls.csv`:

| control | observers moved | largest abs delta MG |
| --- | --- | --- |
| `qp_scale2` (constant gain x2 on the fluctuation) | 0 of 8 | 8.9e-16 |
| `qp_rotate` (fixed orthogonal rotation of the read-out basis) | 1 of 8 | 0.659 (`c_proj1`) |

**The scale control is exact, and it has to be.** `dynamics.simulate` applies
`v.mean() + (v - v.mean()) * 2.0`, so the mean is preserved and the fluctuation doubles; the
scorer then standardises. The gain measured on the stored series is `2.000000000000` with the
mean preserved to 1.8e-15, and the standardised series agree to 4.9e-11 or better on all eight
observers. There is nothing left for the estimator to see. A non-zero result on this control is
therefore a bug in the comparison, never a property of MG -- which is exactly what had been
published (below).

**The rotation control is much weaker than it looks, and the seven zeros are trivial.** The
rotation `R` enters `dynamics.simulate` at one line only, `cr = R @ c`, and `cr` feeds only the
coefficient-projection observers `c_proj1/2/3`. The trajectory itself is untouched. Verified
directly on the stored logs: the raw series of `w_fro, c_norm, g_fro, g_proj, fn_fro, fn_proj1,
loss_full` are **bit-identical** between `qp_r10_s00.npz` and `qp_rotate_r10_s00.npz`. Those
seven observers never saw the rotation, so their zero deltas are not evidence of rotation
invariance. Of the eight scored observers exactly one, `c_proj1`, is actually exposed to the
control, and it moves by **0.659 components**. The honest summary is: this control tests one
observer, and that observer is not rotation invariant.

## Reproducing this directory

```bash
python calibration_k20.py --simulate      # writes trajectories/*.npz (~197 MB, 83 files)
python score_k20_parallel.py              # scores_frozen.csv, heldout_qp_summary.csv,
                                          # frozen_per_r.csv, invariance_controls.csv
python calibration_k20.py --rank-only     # config_observer_ranking.csv, frozen_k20.json
python paper_tables.py                    # checks tables 6 and 7 of the article
```

`score_k20_parallel.py` was re-run against the committed trajectories as a release check.
`scores_frozen.csv` reproduces to a maximum absolute difference of `8.9e-12` (in `TwoNN`, a
KD-tree tie-ordering artefact), `heldout_qp_summary.csv` to `5.3e-15` and `frozen_per_r.csv` to
`8.9e-12`. Note that `calibration_k20.py --score` is a second, serial implementation of the
same scoring, and its `heldout_qp_summary.csv` has different columns from the parallel
scorer's; the committed file is the parallel scorer's. Prefer `score_k20_parallel.py`.

## Two defects found while writing this up

Both are of the class the review's standing check is about: a committed script that does not
regenerate the committed file next to it. Both are now fixed at source, and in both cases the
conclusion survives -- only the published intermediate was wrong.

**1. `invariance_controls.csv` reported impossible deltas.** The committed file reported the
constant-rescaling control moving the estimate by 1.68 to 4.96 components, which cannot happen
for the reason given above. Diagnosis: the published file compared the one-seed control against
the **three-seed median** of the baseline, so it was reporting seed scatter and calling it a
failure of invariance. Reproduced exactly -- recomputing the deltas against the three-seed
median returns the published numbers to 2.2e-16 on all sixteen rows. `score_k20_parallel.py`
already pairs seed 0 against seed 0 and carries a comment saying why; the CSV had simply never
been regenerated after that fix. Re-running the scorer produces the corrected file above. The
superseded values, for the record:

| control | w_fro | c_norm | g_fro | g_proj | c_proj1 | fn_fro | fn_proj1 | loss_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qp_scale2` (withdrawn) | -1.676 | -1.676 | -3.978 | -1.413 | +1.869 | -4.959 | -0.320 | 0.000 |
| `qp_rotate` (withdrawn) | -1.676 | -1.676 | -3.978 | -1.413 | +1.210 | -4.959 | -0.320 | 0.000 |

The old file is not kept as a CSV, deliberately: every CSV in this directory should be
regenerable by a committed script, and a stale one sitting beside the good one invites exactly
the mistake it caused. Git history has it, and the numbers are above.

**2. `config_observer_ranking.csv` was the isotonic version.** Every `cal_mae` in the committed
file was exactly 0.000 and fifteen of sixteen rows tied at a score of 2.8e-17. Diagnosis: the
file was produced by a version of `score()` that fitted an isotonic regression of MG onto the
truth on the same five calibration ranks it then scored -- which interpolates them exactly.
Reproduced: refitting isotonically returns the published file to 4.4e-16 on every row. The
current code rejects that approach explicitly in a comment, so the code was fixed and the CSV
was not. Regenerated with `--rank-only`; `cal_mae` now runs from 2.301 to 4.978 and the four
configurations separate.

The selection outcome is unaffected. Under the stale all-zero ranking the four configurations
tie and `idxmin` would return cfg_id 0; `frozen_k20.json` records cfg_id 3, which is what the
raw criterion selects. So the frozen configuration the article uses was always the right one --
the file that justified it was not.
