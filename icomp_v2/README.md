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

`report.pdf` is committed, so neither step is required to read it. The build is clean and no
red bracketed items remain in the PDF; the open items below are editorial.

The submission is **anonymous**: neither `\icompfinalcopy` nor `\icomparxivcopy` is set, so
the style file prints ``Anonymous authors / Paper under double-blind review'' and the running
head reads ``Under review''. Uncomment `\icomparxivcopy` for a named preprint build and
`\icompfinalcopy` for camera-ready.

**Page budget.** ICOMP allows nine pages excluding references. The body ends on page 9 and the
references begin on page 10, so the limit is met, but **with no slack at all**: section 8 fills
the last lines of page 9. Any addition to the body needs a matching cut, and the appendices,
which follow the references, are where new material should go. The body is about 5300 words
excluding floats. What was moved out to reach nine pages, and can be moved back if the limit
ever rises: the regime map (appendix F), the k = 20 table (appendix B), the change-detection
protocol (appendix D), and the label-matched-pair figure (appendix F). Limitations and
conclusion are one section. There is no keywords line: the style file defines no such field and
it cost three lines on page 1.

Figures are generated at the exact text width of the style file (5.5 in) with 8 pt type, so
LaTeX does not rescale them. Do not include a figure wider than that. `fig_dip` moved to appendix F in the
August revision and was enlarged to 2.05 in there, because the review found it unreadable at
the height the body budget allowed.

## The argument in one paragraph

The number of directions an optimiser actively excites over a window is a well-defined
quantity, distinct both from the number of directions it is allowed to move in and from the
rank of the map to the model's outputs. It can be estimated from one scalar log by delay
reconstruction followed by the pooled Levina-Bickel estimator. We evaluate that estimate on six
systems of increasing realism whose active dimension is fixed by construction and then verified
by measurement, establishing that recovery is accurate to under one component up to about eight
active directions and ordinal up to twenty. Two diagnostics, computable from the series without
a ground truth, flag the two regimes in which the value cannot be read as a count. Applied to
delayed generalisation, the instrument places the two standard experimental settings in
different dynamical regimes and separates generalising from non-generalising runs in four
label-matched pairs; measured directly from the trajectory covariance, it records a transient
collapse to essentially one dimension near generalisation in four regularised runs and not in
their two controls. What the estimator counts is resolvable excited modes rather than the
covariance participation ratio, which the two coincide with only because the constructions
equalise the drive amplitudes.

## Where the numbers come from

| section | source |
| --- | --- |
| 5.1 oscillating diagonal matrix | `../code/dimension_recovery/` exp1--exp3, exp9, exp10 |
| 5.2 online linear and logistic regression | `../code/dimension_recovery/` exp11, exp12 |
| 5.3 decoder and constrained perceptron | `../code/dimension_recovery/` exp13, exp14, and the audit in `../code/active_dimension/README.md` |
| 5.4 image data, parameter subspace | `../code/active_dimension/` E1, E2, and `results/k20_calibration/` |
| 5.4 image data, function subspace | `../code/dimension_recovery/results/exp15_real_digits_functional_subspace_v3/` |
| appendix D, detection of a change | `../code/active_dimension/` E3 |
| 6.1 dynamical regime | `../code/active_dimension/` E0, E2 |
| 6.2 delay lag | `../code/active_dimension/` E6 |
| 6.3 nuisance factors | `../code/active_dimension/` E4 |
| 7.1 regime classification | `../code/active_dimension/` E5; `../code/gromov_arithmetic/dimension_probe.py` |
| 7.2 label-matched pairs | `../code/gromov_polynomials/report.md`, `../code/gromov_arithmetic/report.md` |
| 7.3 direct measurement | `../code/active_rank/report.md` |
| appendix H, full-batch setting and window-length sweep | `../code/gromov_arithmetic/report.md`, `../code/gromov_arithmetic/results/rank_fb_long/pr_vs_window.csv` |
| appendix J, representation dimension | `../code/gromov_arithmetic/report.md`, `../code/gromov_polynomials/report.md` |
| 6.4 effective rank or mode count | `../code/active_dimension/e8_anisotropy.py`, `results/e8_anisotropy/` |
| appendix L, the Theiler exclusion | `../code/active_dimension/e7b_theiler_quick.py`, `results/e7_theiler/` |

## Open items

1. None outstanding. The draft has no TODO markers, builds without errors or undefined
   references, and meets the page limit.

## Appendices

| # | contents | status |
| --- | --- | --- |
| A | the estimator, as an algorithm | complete |
| B | the two frozen configurations and the grids they came from | complete |
| C | per-observer results, and MG against every alternative statistic | complete |
| D | delay lag, nuisance factors, and the change-detection rule | complete |
| E | how the excitation is built, and the active dimension it achieves | complete |
| F | the label-matched pairs, with the training loss that explains them | complete |
| G | the trajectory sketch and the three checks behind it | complete |
| H | the full-batch measurement, its resolution limit, and the window-length sweep that removes it | complete |
| I | what falls, and when: the non-specific fall and the censored budgets | complete |
| J | the representation dimension in closed form, per task | complete |
| K | inventory of every training run the paper uses | complete |
| L | the Theiler exclusion of the twenty-direction configuration | complete |

## Figures

Regenerate with `python make_figures.py`. Four figures, all drawn at exactly 5.5 in, the text
width of the style file, at 8 pt so that LaTeX never rescales them.

| figure | what it shows | where |
| --- | --- | --- |
| `fig_regimes` | recovery and admissibility on the image-data system of section 5.4 | section 6 |
| `fig_dip` | the trajectory-rank collapse at generalisation | section 7.3 |
| `fig_map` | every training log in the plane of the two diagnostics, for section 7.1 | appendix F |
| `fig_pairs` | the label-matched pairs and the loss curves behind them, for section 7.2 | appendix F |

Colour encodes the three dynamical regimes of table 1 and nothing else: recurrent, stochastic,
transient. Line style separates variants inside a regime and marker shape separates
experimental settings, so identity never rests on colour alone. The palette is
`#0072B2 / #D55E00 / #5D3A9B`; it passes the OKLCH lightness band, the chroma floor, the
Machado severity-1.0 protan and deutan separation on all pairs, the normal-vision floor and
the WCAG contrast check against the page, with no warnings. Re-run those checks before
changing any colour.

Do not set `savefig(bbox_inches="tight")` here. A legend wider than the axes then expands the
canvas past 5.5 in, LaTeX scales the figure down to fit, and the type shrinks with it; the
figures reserve space for the legend with `tight_layout(rect=...)` instead.

## Corrections carried into this draft

Two defects found while assembling the draft, both fixed here and both needing a fix in the
source repository.

* `../code/active_dimension/results/k20_calibration/invariance_controls.csv` reports the
  constant-rescaling control moving the estimate by 1.68 to 4.96 components. It cannot: the
  scorer standardises the series and the control applies a constant gain to the fluctuation,
  so the two cancel exactly. Recomputed from `scores_frozen.csv`, the control moves the
  estimate by 0.000 on all eight observers, and the rotation control by 0.000 on seven of
  eight, the exception being the fixed random projection, which is not rotation invariant.
  The published deltas are the difference between seed 0 and the median over seeds, that is
  seed scatter. The article quotes the corrected values and quotes the seed scatter
  separately, in table 2, where it belongs.
* The k = 1..20 calibration on real data has never been written up. Its result files are in
  the repository but neither `report.md` nor `README.md` in `active_dimension/` mentions it.
  Table 2 of this article is computed from `scores_frozen.csv`.

## What the August review changed

An external audit of the 22-page draft raised twelve critical points. The ones that were
correct and what was done:

| point | status |
| --- | --- |
| PR of the covariance is an effective rank, not a count of directions | correct; section 3.1 now says so, and section 6.4 measures the difference with a new anisotropy experiment (`../code/active_dimension/e8_anisotropy.py`) |
| the ground truth (covariance PR) and the estimator's target (manifold dimension) are different quantities | correct; they agree only because the constructions equalise the drive amplitudes, which is now stated, and the new experiment shows the estimator follows the mode count |
| positions, increments and detrended positions were all called `d_act` | correct; now `d_pos`, `d_upd`, `d_det`, with the primary endpoint named per section |
| "a transient has active dimension 1" conflates manifold dimension with PR | correct; the measured range 1.02-1.09 is quoted instead |
| table 1 was too categorical | correct; the last column now says what the estimator returns in our constructions |
| "recurrence count" does not measure recurrence | correct; renamed the trend-crossing count, with its exact definition (least-squares line over the window) |
| tau is claimed to be estimated from the autocorrelation time on grokking logs | correct and it was false: tau = 4 is transplanted from calibration while the logs' autocorrelation time is 161-858. Section 6.2 now says so, which strengthens the paper's own conclusion |
| the k = 20 result uses a Theiler window smaller than its embedding span | correct: 150 against a span of 624. The result is now marked exploratory and appendix K measures the cost |
| "active dimension identically one" at zero learning rate | correct, and worse than stated: the parameters do not move at all, so the 1.0 is a floating-point artefact of centring a constant trajectory. The control is now stated without it |
| the fall in the p=211 wd=0 run is used as evidence of trajectory simplification | correct, and this was the paper's most serious logical error. That series is inadmissible by the paper's own diagnostics; abstract, section 7.4 and section 8 now separate the two claims |
| the direct measurement has undocumented confounds | correct: alignment, window centring, detrending and the position/update distinction are now all stated |
| "five systems" when there are six | correct |
| S5 weight decay is 1.0 in the text and 0.2 in the table | correct: it is 0.2, and those runs also step the optimiser twice per batch |
| "never generalises" for censored runs | correct; the run table now carries final validation accuracy |
| Algorithm 1 does not match the code | correct: the dither, the 1 per cent degeneracy tolerance and the (N(m-1)-1)/S form are now in it |
| CountSketch guarantee attributed to Charikar et al. | correct: that is the frequent-items paper, the bound is a pairwise one and does not transfer to a spectrum. The appendix now claims no spectral guarantee and reports the two-sketch disagreement instead (1.1 per cent median, 8.4 per cent worst) |
| the sketch validation looks biased at rank 5 and 10 | the deficit is the estimator's, not the sketch's; the uncompressed column that shows this was in the log and missing from the paper, and is now table 8 |
| appendix H concludes "the same place" from one pair | correct: the two maxima are adjacent points of a 3450-step grid and cannot be separated. The claim is now the negative one only |
| appendix I mixes three different numbers | correct: mode count (49), order parameter (1.000) and weight-matrix effective rank (148.8) are now defined separately, with the note that random initialisation already reads 139.1 |
| "MG > E_max is a ceiling" | correct: no clamping is applied, so it is not a bound. Figure 1 relabelled |

Points not adopted, with reasons, are in the response to the review.
