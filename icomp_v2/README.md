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
| 7.1 regime classification | `../code/active_dimension/` E5; `../code/gromov_arithmetic/dimension_probe.py` |
| 7.2 label-matched pairs | `../code/gromov_polynomials/report.md`, `../code/gromov_arithmetic/report.md` |
| 7.3 direct measurement | `../code/active_rank/report.md`; `dip.py` regenerates `results_fine/rank_dip.csv` |
| appendix E, detection of a change | `../code/active_dimension/` E3 |
| appendix H, the collapse run by run | `../code/active_rank/results_fine/rank_dip*.csv` |
| appendix J, full-batch and window length | `../code/gromov_arithmetic/results/rank_fb_long/pr_vs_window.csv` |
| appendix L, representation dimension | `../code/gromov_arithmetic/report.md`, `../code/gromov_polynomials/report.md` |
| appendix M, the Theiler exclusion | `../code/active_dimension/e7b_theiler_quick.py`, `results/e7_theiler/` |

## Appendices

| # | contents |
| --- | --- |
| A | the estimator, as an algorithm |
| B | the twelve observers, defined |
| C | the two frozen configurations and the grids they came from |
| D | per-observer results, and MG against every alternative statistic |
| E | delay lag, nuisance factors, and the change-detection rule |
| F | how the excitation is built, and the effective rank it achieves |
| G | diagnostics and figures for the grokking application |
| H | the collapse, run by run |
| I | the trajectory sketch and the checks behind it |
| J | the full-batch measurement and the window-length sweep |
| K | what falls, and when |
| L | the representation dimension in closed form, per task |
| M | the Theiler exclusion of the twenty-direction configuration |
| N | inventory of every training run the paper uses |

## Figures

Regenerate with `python make_figures.py`. All are drawn at exactly 5.5 in, the text width of the
style file, at 8 pt so that LaTeX never rescales them.

| figure | what it shows | where |
| --- | --- | --- |
| `fig_regimes` | recovery and admissibility on the image-data system | section 6 |
| `fig_dip` | the collapse of the trajectory effective rank at generalisation | section 7.3 |
| `fig_map` | every training log in the plane of the two diagnostics | appendix G |
| `fig_pairs` | the label-matched pairs and the loss curves behind them | appendix G |

Colour encodes the three dynamical regimes of table 1 and nothing else: recurrent, stochastic,
transient. Line style separates variants inside a regime and marker shape separates experimental
settings, so identity never rests on colour alone. The palette is `#0072B2 / #D55E00 / #5D3A9B`;
it passes the OKLCH lightness band, the chroma floor, the Machado severity-1.0 protan and deutan
separation on all pairs, the normal-vision floor and the WCAG contrast check against the page.
Re-run those checks before changing any colour.

Two further rules were added after a review found figures misdescribing their own data. Where a
plotted point is an aggregate, its spread is drawn with it. Where points coincide, the
multiplicity is made visible rather than hidden — `fig_regimes(b)` plots every raw value because
it rests on 54 of 2240 rows, and `fig_map` offsets and labels the twelve runs that share a single
coordinate. Any edit that reintroduces one mark for many runs must also fix the caption.

Do not set `savefig(bbox_inches="tight")`. A legend wider than the axes then expands the canvas
past 5.5 in, LaTeX scales the figure down to fit, and the type shrinks with it; the figures
reserve space for the legend with `tight_layout(rect=...)` instead.

## Review status

The full external review is in [`review_fable_archive.md`](review_fable_archive.md). The live
copy, [`review_fable.md`](review_fable.md), has resolved items deleted from it and a changelog at
the top recording what each installment changed. What remains open is listed in its priority
list; the two substantive items are an experiment that has not been run (a sliding-window
estimate on the grokking logs, whose input data is already computed) and one that needs training
compute (full-batch descent above the stability threshold, to find whether an undriven run is
ever admissible).

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
