# icomp_v2

Draft of the article *Counting the Active Degrees of Freedom of a Training Run: What a Single
Scalar Log Can and Cannot Measure*.

The previous report, on delay-embedding preconditions and convergent cross mapping, is
archived unchanged in [`../icomp_attempts/`](../icomp_attempts/). This draft does not build
on it and shares no text with it.

## Build

```bash
python make_figures.py    # regenerates figures/ from the committed result files
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

`report.pdf` is committed, so neither step is required to read it. The build is clean: no
undefined references, no citation warnings, no overfull boxes.

The submission is **anonymous**: neither `\icompfinalcopy` nor `\icomparxivcopy` is set, so
the style file prints ``Anonymous authors / Paper under double-blind review'' and the running
head reads ``Under review''. Uncomment `\icomparxivcopy` for a named preprint build and
`\icompfinalcopy` for camera-ready.

**Page budget.** ICOMP allows nine pages excluding references. The body ends on page 9 and the
references begin on page 10, with about one line of slack. Anything added to the body needs a
matching cut. The appendices follow the references and are unconstrained, so new material goes
there. There is no keywords line: the style file defines no such field.

Figures are generated at the exact text width of the style file (5.5 in) with 8 pt type, so
LaTeX does not rescale them. Do not include a figure wider than that, and read the design rules
in the `make_figures.py` docstring before changing one — several of them exist because a review
found the figure misrepresenting its own data.

## The argument in one paragraph

The number of independent components of the set an optimiser recurrently visits over a window is
a well-defined quantity, distinct from the number of directions it may move in, from the rank of
the map to the model's outputs, and from the effective rank of its trajectory covariance. It is
defined at a resolution, because over a finite window the orbit closure is a curve for every rank
and a neighbour statistic sees the visited set at the scale of its own neighbourhoods. It can be
estimated from one scalar log by delay reconstruction followed by the pooled Levina-Bickel
estimator. We evaluate that estimate on six systems whose component count is fixed by
construction and then verified by measurement: recovery reaches a mean absolute error of 0.87
components up to eight active directions, and the ceiling above which it fails is consistent with
the embedding condition E > 2d. Two diagnostics, computable from the series without a ground
truth, flag the two regimes in which the value cannot be read as a count. Applied to delayed
generalisation they place both standard experimental settings outside that regime. Measured
directly from the stored trajectory instead, the effective rank collapses by a median factor of
seven near generalisation in four regularised runs and re-expands; the controls fall too, so what
marks the transition is the timing and shape of the fall rather than its depth.

## Where the numbers come from

| section | source |
| --- | --- |
| 5.1 oscillating diagonal matrix | `../code/dimension_recovery/` exp1--exp3, exp9, exp10 |
| 5.2 the ceiling | `../code/dimension_recovery/` exp11, exp12; `../code/active_dimension/results/k20_calibration/` |
| 5.3 the silence control | `../code/dimension_recovery/` exp13, exp14; `../code/active_dimension/` E2 (`qp_eta0`) |
| 5.4 image data, parameter subspace | `../code/active_dimension/` E1, E2, and `results/k20_calibration/` |
| 5.4 image data, function subspace | `../code/dimension_recovery/results/exp15_real_digits_functional_subspace_v3/` |
| 6.1 dynamical regime | `../code/active_dimension/` E0, E2 |
| 6.2 delay lag | `../code/active_dimension/` E6 |
| 6.3 nuisance factors | `../code/active_dimension/` E4 |
| 6.4 effective rank or mode count | `../code/active_dimension/e8_anisotropy.py`, `results/e8_anisotropy/` |
| 7.1 regime classification, label-matched pairs | `../code/active_dimension/` E5; `../code/gromov_arithmetic/dimension_probe.py`; `../code/gromov_polynomials/report.md` |
| 7.2 direct measurement | `../code/active_rank/report.md`; `dip.py` regenerates `results_fine/rank_dip.csv` |
| 7.3 the matched window | `../code/active_dimension/e9_matched_window.py`, `e9_analyse.py`, `e10_surrogate.py`, `results/e9_matched_window/` |
| appendix E, detection of a change | `../code/active_dimension/` E3 |
| appendix H, the collapse run by run | `../code/active_rank/results_fine/rank_dip*.csv` |
| appendix J, full-batch and window length | `../code/gromov_arithmetic/results/rank_fb_long/pr_vs_window.csv` |
| appendix L, the matched-window re-run | `../code/active_dimension/results/e9_matched_window/` |
| appendix M, representation dimension | `../code/gromov_arithmetic/report.md`, `../code/gromov_polynomials/report.md` |
| appendix N, the Theiler exclusion | `../code/active_dimension/e7b_theiler_quick.py`, `results/e7_theiler/` |
| appendix P, the exclusion contrast | `../code/active_dimension/e11_theiler_contrast.py`, `results/e11_theiler_contrast/` |
| appendix Q, edge of stability | `../code/gromov_arithmetic/eos.py`, `eos_probe.py`, `eos_recurrence.py`, `results/eos/report.md` |
| appendix R, the ceiling sweeps | `../code/active_dimension/e10_ceiling_sweep.py`, `results/e10_ceiling/report.md` |

## Appendices

| # | contents |
| --- | --- |
| A | the estimator, as an algorithm |
| B | the twelve observers, defined |
| C | the two frozen configurations and the grids they came from |
| D | the ladder table, per-observer results, and MG against every alternative statistic |
| E | delay lag, nuisance factors, and the change-detection rule |
| F | how the excitation is built, and the effective rank it achieves |
| G | diagnostics and figures for the grokking application |
| H | the collapse, run by run |
| I | the trajectory sketch and the checks behind it |
| J | the full-batch measurement and the window-length sweep |
| K | what falls, and when |
| L | what a windowed estimate on a log can resolve, and the matched-window re-run |
| M | the representation dimension in closed form, per task |
| N | the Theiler exclusion of the twenty-direction configuration |
| O | inventory of every training run the paper uses |
| P | the exclusion sweep: a transient's dimension is 1, and identifiable only without it |
| Q | full-batch descent at the edge of stability, and what the logging stride hides |
| R | sweeping the embedding against the record: neither explains the ceiling |

## Figures

Regenerate with `python make_figures.py`. All are drawn at exactly 5.5 in, the text width of the
style file, at 8 pt so that LaTeX never rescales them.

| figure | what it shows | where |
| --- | --- | --- |
| `fig_regimes` | recovery and admissibility on the image-data system | section 6 |
| `fig_dip` | the collapse of the trajectory effective rank at generalisation | section 7.3 |
| `fig_observers` | per-observer error with across-seed spread: which log to keep | appendix D |
| `fig_tau` | the delay-lag sweep, collapsing at one end and diverging at the other | appendix E |
| `fig_aniso` | the count against the effective rank as the drive is made anisotropic | appendix E |
| `fig_map` | every training log in the plane of the two diagnostics | appendix G |
| `fig_pairs` | the label-matched pairs and the loss curves behind them | appendix G |
| `fig_prwindow` | the participation ratio against window length | appendix J |
| `fig_window` | the windowed estimate on the grokking logs, and what it cannot resolve | appendix L |
| `fig_eos` | the stability ratio, the two-cycle, and what a stride of ten removes | appendix Q |
| `fig_ceiling` | the rank tracked against each hypothesis's knob, and its prediction | appendix R |

Colour encodes the three dynamical regimes of table 1 and nothing else: recurrent `#004488`,
stochastic `#BB5566`, transient `#997700`, with `#666666` for reference lines. This is Paul Tol's
high-contrast qualitative scheme. It replaced an Okabe-Ito-style blue/orange/purple set whose
worst pair separated by only 8.8 ΔE under simulated deuteranopia; the current set measures 45.3 ΔE
protan and 50.7 ΔE deutan at minimum WCAG contrast 4.21 against the page. Re-run those checks
before changing any colour, and note that the earlier README claimed the old palette passed them
when it did not.

Line style and marker shape carry every distinction that is not the regime, so identity never
rests on colour alone. Three further rules exist because reviews found figures misrepresenting
their own data, and should not be undone for tidiness:

* where a plotted point is an aggregate, its spread is drawn with it;
* where points coincide, the multiplicity is made visible — `fig_regimes(b)` plots every raw value
  because it rests on 54 of 2240 rows, and `fig_map` offsets and labels the twelve runs that share
  one coordinate;
* explanatory prose lives in captions, not inside the axes. Inside the axes there is at most one
  short pointer per panel.

Do not set `savefig(bbox_inches="tight")`. A legend wider than the axes then expands the canvas
past 5.5 in, LaTeX scales the figure down to fit, and the type shrinks with it; the figures
reserve space for the legend with `tight_layout(rect=...)` instead.

## The matched window

Section 7.3 is the experiment the earlier draft named as missing and then declined to run. The
frozen configuration's window spans 39 990 optimiser steps; the collapse section 7.2 measures
lasts one to two thousand, so nothing localised to the transition could appear in it, and the
draft said so and stopped. `e9_matched_window.py` re-runs the estimator at the window the direct
measurement uses --- 600 steps, on that measurement's own window midpoints, so the two statistics
are paired sample by sample --- and `e10_surrogate.py` asks whether what appears there is the
observer's shape.

Four things about it are worth keeping in mind when editing:

* **The configuration is not the frozen one and cannot be.** At 60 samples the frozen delay span,
  (20 - 1) x 4 = 76, exceeds the window. Choosing a replacement on the outcome is exactly the
  failure requirement 2 exists to prevent, so the whole 36-cell grid is written out and the
  headline cell is named by a rule that cannot see the answer. 25 of the 26 cells long enough to
  return a value separate the groups; the article reports that count, not the best cell.
* **The linear-detrend arm is negative and is reported.** Removing a linear trend from the window
  --- what `PR^det` does per coordinate --- leaves no cell separating the groups. On a
  1024-dimensional sketch that removal takes one direction of many; on a scalar it takes most of
  the series. The surrogate control, which keeps the shape and destroys only the fine structure,
  is the one that discriminates, and both are in appendix L.
* **The nulls beat the estimator here.** The roughness of the same windows falls by 6.5 to 55
  where the estimate falls by 2. Section 7.3 says so. Any edit that drops that sentence makes the
  article claim more than it measured.
* **The identifiability ratio is not computable at this window** for the headline cell, since it
  needs 2 E_max. The level there is therefore uncertified and only the change is read; appendix K
  states that the weaker claim of the frozen-window analysis still stands.

## Review status

The full external review is in [`review_fable_archive.md`](review_fable_archive.md). The live
copy, [`review_fable.md`](review_fable.md), has resolved items deleted from it and a changelog at
the top recording what each installment changed. What remains open is listed in its priority
list. The sliding-window estimate on the grokking logs it asked for has been run, at a window
matched to the transition, and is section 7.3 and appendix L. So has the edge-of-stability run,
which is appendix Q: full-batch descent does reach the edge, but oscillation turns out not to be
recurrence, and the paper's two diagnostics identify the wrong runs when both are present. So has
the sweep of E_max against the record length, which is appendix R and which refutes both
candidate explanations of the ceiling: the finite-record bound outright, and the Takens form
quantitatively — the ceiling rises about 0.05 components per unit of E_max rather than a half, so
`E_max/2` was right only at the E_max = 20 the paper had frozen. The priority list is now empty.

**A release check this repository has now failed twice.** `active_rank/dip.py` did not regenerate
the file section 7.1 depended on, and `active_dimension/e10_surrogate.py` could not regenerate
`surrogates.csv` in any process, because it seeded from `hash()` of a string and Python salts that
per interpreter. Before release, re-run every committed script and diff its output against the
committed file. Both are fixed; the point is that neither was caught by reading the code.

Two defects in upstream result files were found while assembling the draft and are fixed here but
still need fixing at source:

* `../code/active_dimension/results/k20_calibration/invariance_controls.csv` reports the
  constant-rescaling control moving the estimate by 1.68 to 4.96 components. It cannot: the
  scorer standardises the series and the control applies a constant gain to the fluctuation, so
  the two cancel exactly. Recomputed from `scores_frozen.csv`, the control moves the estimate by
  0.000 on all eight observers, and the rotation control by 0.000 on seven of eight, the
  exception being the fixed random projection, which is not rotation invariant. The published
  deltas are seed scatter.
* The k = 1..20 calibration on real data is written up nowhere in `active_dimension/`, though its
  result files are committed. Appendix C and table 6 of this article are computed from
  `scores_frozen.csv`.
