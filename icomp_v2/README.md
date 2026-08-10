# icomp_v2

Draft of the article *Estimating the Number of Active Degrees of Freedom of a Training Run
from a Single Scalar Log*.

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
which follow the references, are where new material should go. The body is about 4950 words
excluding floats. What was moved out to reach nine pages, and can be moved back if the limit
ever rises: the regime map (appendix F), the k = 20 table (appendix B), the change-detection
protocol (appendix D), and the label-matched-pair figure (appendix F). Limitations and
conclusion are one section. There is no keywords line: the style file defines no such field and
it cost three lines on page 1.

Figures are generated at the exact text width of the style file (5.5 in) with 8 pt type, so
LaTeX does not rescale them. Do not include a figure wider than that. `fig_dip` is drawn at
1.32 in for the page budget, which is below the height at which a spelt-out y-axis label fits;
its label is therefore `PR`, and raising the height is the precondition for lengthening it.

## The argument in one paragraph

The number of directions an optimiser actively excites over a window is a well-defined
quantity, distinct both from the number of directions it is allowed to move in and from the
rank of the map to the model's outputs. It can be estimated from one scalar log by delay
reconstruction followed by the pooled Levina-Bickel estimator. We calibrate that estimate on
five systems of increasing realism whose active dimension is known by construction and then
verified by measurement, establishing that recovery is accurate to under one component up to
about eight active directions and ordinal up to twenty. Two diagnostics, computable from the
series without a ground truth, decide whether a given value is a count or a property of the
embedding space. Applied to delayed generalisation, the instrument places the two standard
experimental settings in different dynamical regimes, separates generalising from
non-generalising runs in four label-matched pairs, and, measured directly from the
trajectory covariance, records a transient collapse to essentially one dimension at
generalisation in the regularised stochastic setting.

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
| appendix I, representation dimension | `../code/gromov_arithmetic/report.md`, `../code/gromov_polynomials/report.md` |

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
| I | the representation dimension in closed form, per task | complete |
| J | inventory of every training run the paper uses | complete |

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
