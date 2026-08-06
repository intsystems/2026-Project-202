# icomp_v2

Successor report to [`../icomp_article/`](../icomp_article/). Where the earlier paper
proposed an intrinsic-dimension collapse as a predictor of grokking, this one establishes
the conditions under which delay-embedding methods apply to training logs at all, reports
what does and does not survive those conditions, and states a control protocol.

## Build

```bash
python make_figures.py          # regenerates figures/ from the committed result files
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

`report.pdf` is committed, so neither step is required to read it.

## Contents

| file | role |
| --- | --- |
| `report.tex`, `report.pdf` | the report (8 pages) |
| `make_figures.py` | regenerates every figure from the analysis outputs |
| `figures/` | five figures, PDF for LaTeX and PNG for preview |
| `references.bib` | bibliography, extending the earlier paper's with the surrogate and cross-mapping literature |

## The argument in one paragraph

The embedding theorems of Takens and Stark reconstruct a compact invariant set that the
trajectory revisits. A training run forced by an external driver with its own attractor
satisfies that hypothesis by construction; the intrinsic optimisation transient does not,
and we measure the difference rather than assume it. Three statistics that appear to
detect the grokking transition are shown to separate run configurations instead, each
falsified by a control that differs from the positive runs in one respect only. In the
driven regime, convergent cross mapping recovers an injected driver from a single scalar
loss log in seven of eight coupled runs across two architectures, with no false positives
among five runs whose driver was logged but never applied.

## Where the numbers come from

Every figure and every quoted value is produced by code in
[`../code/edm_validation/`](../code/edm_validation/) and
[`../code/prediction_improved/`](../code/prediction_improved/); the running log of the
investigation, including the false starts and the two ground-truth labels we had to
correct, is [`../code/edm_validation/NOTES.md`](../code/edm_validation/NOTES.md).

| figure | source |
| --- | --- |
| 1 preconditions | `edm_validation/forecast.py::recurrence_profile` |
| 2 confound-free test | `edm_validation/results/phase6_*.csv`, `phase7_*.csv` |
| 3 driver recovery | `edm_validation/results/phase3_ccm.csv`, `phase5_inject_ccm.csv` |
| 4 convergence | `edm_validation/ccm.py::ccm_convergence` |
| 5 delay distribution | `prediction_improved/results/sweep/summary.csv` |

Figures use the Okabe-Ito qualitative palette, which is designed and tested for
colour-vision deficiency; hues are assigned to entities in a fixed order so that a series
keeps its colour across figures.
