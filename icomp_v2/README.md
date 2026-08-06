# icomp_v2

A self-contained report on when delay-embedding methods apply to neural network training
logs. It fixes the validation requirements it holds itself to, measures whether the
preconditions of the embedding theorems are met and what the sample budget of a training
run permits, reports three candidate grokking signals that fail controlled comparison, gives a positive result in the driven regime with its
direction tested, and measures how variable the delayed transition itself is.

It does not depend on any earlier draft: every claim is supported by code and result files
in this repository.

## Build

```bash
python make_figures.py          # regenerates figures/ from the committed result files
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

`report.pdf` is committed, so neither step is required to read it.

## Contents

| file | role |
| --- | --- |
| `report.tex`, `report.pdf` | the report (13 pages) |
| `make_figures.py` | regenerates every figure from the analysis outputs |
| `verify_numbers.py` | checks every number quoted in the report against its result file |
| `figures/` | nine figures, PDF for LaTeX and PNG for preview |
| `references.bib` | bibliography, covering the embedding, surrogate and cross-mapping literature |

## The argument in one paragraph

The embedding theorems of Takens and Stark reconstruct a compact invariant set that the
trajectory revisits. A training run forced by an external driver with its own attractor
satisfies that hypothesis by construction; the intrinsic optimisation transient does not,
and we measure the difference rather than assume it. Three statistics that appear to
detect the grokking transition are shown to separate run configurations instead, each
falsified by a control that differs from the positive runs in one respect only. In the
driven regime, convergent cross mapping recovers an injected driver from a single scalar
loss log in seven of eight coupled runs across two architectures, with no false positives
among the runs whose driver was logged but never applied. Its direction is tested rather
than assumed: the forward map converges more strongly than the reverse in seven of eight
coupled runs while the controls converge in neither, though unidirectionality is not
demonstrable when the driver acts on the loss without delay.

## Where the numbers come from

Every figure and every quoted value is produced by code in
[`../code/edm_validation/`](../code/edm_validation/) and
[`../code/prediction_improved/`](../code/prediction_improved/); the running log of the
investigation, including the false starts and the two ground-truth labels we had to
correct, is [`../code/edm_validation/NOTES.md`](../code/edm_validation/NOTES.md).

| figure | source |
| --- | --- |
| preconditions | `edm_validation/forecast.py::recurrence_profile` |
| dimension artifact | `edm_validation/phase8_dimension_evidence.py` |
| dimension with no temporal exclusion | `edm_validation/phase9_paper_settings.py` |
| function-space velocity | `prediction_improved/results/*_probe.csv` |
| confound-free test | `edm_validation/results/phase6_*.csv`, `phase7_*.csv` |
| driver recovery | `edm_validation/results/phase3_ccm.csv`, `phase5_inject_ccm.csv` |
| direction of coupling | `edm_validation/phase12_directionality.py` |
| local stationarity | `edm_validation/phase14_local_stationarity.py` |
| delay distribution | `prediction_improved/results/sweep/summary.csv` |

`phase13_dense_logging.py` and `phase14_local_stationarity.py` exist because measuring
recurrence over a whole run does not settle whether short windows can be analysed as
stationary systems. They log every optimisation step and test the local hypothesis
directly. The training loss does recur locally; the weight norm does not and is never
locally stationary; and the binding constraint turns out to be the number of independent
samples, which no logging rate changes.

`phase12_directionality.py` exists because detection is not causation. Cross mapping earns
the causal reading only if it distinguishes "driver forces loss" from the converse, and
these logs make that testable: the driver schedule is evaluated from a fixed seed before
training, so the ground truth is unidirectional by construction.

`phase9_paper_settings.py` exists because the Theiler exclusion is itself a fair target for
objection: the estimate is often computed without one. That script repeats the comparison at
`W=0`, `k=5`, `max_E=15` and a 300-sample window, that is with no temporal exclusion at all,
and finds the descent is not reproducible across seeds of the grokking condition itself.

Figures use the Okabe-Ito qualitative palette, which is designed and tested for
colour-vision deficiency; hues are assigned to entities in a fixed order so that a series
keeps its colour across figures.
