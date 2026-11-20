# Manuscript source

Edit [main.tex](main.tex) for the workshop article. The file contains the main text and appendices. [references.bib](references.bib) holds the bibliography, [neurips_2026.sty](neurips_2026.sty) supplies the workshop style, and [figures/](figures/ReadMe.md) contains the vector graphics included by LaTeX.

## Build

From this directory, run:

```bash
latexmk -pdf '-outdir=.build' -interaction=nonstopmode -halt-on-error main.tex
```

The build writes `.build/main.pdf`. A full TeX Live installation provides the required LaTeX packages and fonts.

The [reference PDF](../icomp_artifacts.pdf) is the anonymous submission. To update it, rebuild, inspect the result, and replace that PDF with the approved version.

## Article sections and experiments

Use the TeX labels to find the cited text in [main.tex](main.tex). The experiment pages link back here. When revising a section title or moving a result, update these links and the corresponding experiment README.

### Section 3.3 · Observation model and estimator

TeX label `sec:estimator`. Related code and data: [Delay coordinate reconstruction](../../code/synthetic/delay_embedding_and_limits/delay_embedding/ReadMe.md).

### Section 4 · Validation protocol

TeX label `sec:protocol`. Related code and data: [Estimator calibration](../../code/synthetic/digits_validation/estimator_calibration/ReadMe.md), [Zero learning rate](../../code/synthetic/digits_validation/zero_learning_rate/ReadMe.md).

### Section 5 · Evaluation on systems with a known active dimension

TeX label `sec:ladder`. Related code and data: [Frozen nonlinear decoder](../../code/synthetic/controlled_systems/frozen_decoder/ReadMe.md), [Logistic regression](../../code/synthetic/controlled_systems/logistic_regression/ReadMe.md), [Online linear regression](../../code/synthetic/controlled_systems/online_linear_regression/ReadMe.md), [Oscillating diagonal matrix](../../code/synthetic/controlled_systems/oscillating_matrix/ReadMe.md), [Perceptron in a subspace](../../code/synthetic/controlled_systems/subspace_perceptron/ReadMe.md).

### Section 5.1 · A constrained head on a network trained on image data

TeX label `sec:digits`. Related code and data: [Dimension ceiling](../../code/synthetic/delay_embedding_and_limits/dimension_ceiling/ReadMe.md), [Digits parameter subspace](../../code/synthetic/digits_validation/digits_classifier_head/ReadMe.md), [Zero learning rate](../../code/synthetic/digits_validation/zero_learning_rate/ReadMe.md).

### Section 6.1 · Recovery in each regime

TeX label `sec:regime`. Related code and data: [Theiler exclusion and transients](../../code/synthetic/delay_embedding_and_limits/theiler_exclusion/ReadMe.md), [Digits parameter subspace](../../code/synthetic/digits_validation/digits_classifier_head/ReadMe.md).

### Section 6.2 · The delay lag

TeX label `sec:tau`. Related code and data: [Delay lag sensitivity](../../code/synthetic/delay_embedding_and_limits/delay_lag_sweep/ReadMe.md).

### Section 6.4 · Nuisance factors and nulls

TeX label `sec:nuisance`. Related code and data: [Observer and scale controls](../../code/synthetic/signal_controls_and_sketches/observer_controls/ReadMe.md), [Unequal excitation](../../code/synthetic/signal_controls_and_sketches/unequal_excitation/ReadMe.md).

### Section 7.1 · The diagnostics reject both settings

TeX label `sec:grok-diagnostics`. Related code and data: [Full-batch perceptron grokking](../../code/real/grokking_and_stability/perceptron_grokking/ReadMe.md), [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md).

### Section 7.2 · The trajectory simplifies at generalisation

TeX label `sec:direct`. Related code and data: [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md).

### Section 7.3 · The log estimate falls where the trajectory does

TeX label `sec:matched`. Related code and data: [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md).

### Appendix A · The results behind the estimator

TeX label `app:theory`. Related code and data: [Delay coordinate reconstruction](../../code/synthetic/delay_embedding_and_limits/delay_embedding/ReadMe.md).

### Appendix B.3 · Frozen configurations

TeX label `app:config`. Related code and data: [Estimator calibration](../../code/synthetic/digits_validation/estimator_calibration/ReadMe.md).

### Appendix C · Per-observer results and alternative statistics

TeX label `app:nulls`. Related code and data: [Frozen nonlinear decoder](../../code/synthetic/controlled_systems/frozen_decoder/ReadMe.md), [Logistic regression](../../code/synthetic/controlled_systems/logistic_regression/ReadMe.md), [Online linear regression](../../code/synthetic/controlled_systems/online_linear_regression/ReadMe.md), [Oscillating diagonal matrix](../../code/synthetic/controlled_systems/oscillating_matrix/ReadMe.md), [Perceptron in a subspace](../../code/synthetic/controlled_systems/subspace_perceptron/ReadMe.md), [Delay coordinate reconstruction](../../code/synthetic/delay_embedding_and_limits/delay_embedding/ReadMe.md), [Digits function subspace](../../code/synthetic/digits_validation/digits_function_space/ReadMe.md), [Zero learning rate](../../code/synthetic/digits_validation/zero_learning_rate/ReadMe.md).

### Appendix D · Conditions of validity in detail

TeX label `app:validity`. Related code and data: [Delay lag sensitivity](../../code/synthetic/delay_embedding_and_limits/delay_lag_sweep/ReadMe.md), [Dimension transitions](../../code/synthetic/signal_controls_and_sketches/dimension_transitions/ReadMe.md), [Observer and scale controls](../../code/synthetic/signal_controls_and_sketches/observer_controls/ReadMe.md), [Unequal excitation](../../code/synthetic/signal_controls_and_sketches/unequal_excitation/ReadMe.md).

### Appendix E · System constructions

TeX label `app:constructions`. Related code and data: [Digits parameter subspace](../../code/synthetic/digits_validation/digits_classifier_head/ReadMe.md), [Digits function subspace](../../code/synthetic/digits_validation/digits_function_space/ReadMe.md).

### Appendix F · Diagnostics and figures for the grokking application

TeX label `app:pairs`. Related code and data: [Full-batch perceptron grokking](../../code/real/grokking_and_stability/perceptron_grokking/ReadMe.md).

### Appendix G.1 · The trajectory sketch

TeX label `app:sketch`. Related code and data: [Trajectory sketch validation](../../code/synthetic/signal_controls_and_sketches/trajectory_sketch/ReadMe.md).

### Appendix G.2 · The effective-rank collapse, run by run

TeX label `app:dip`. Related code and data: [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md).

### Appendix H.1 · The full-batch setting

TeX label `app:fb`. Related code and data: [Full-batch perceptron grokking](../../code/real/grokking_and_stability/perceptron_grokking/ReadMe.md).

### Appendix H.2 · Declines of the estimate that are not generalisation

TeX label `app:falls`. Related code and data: [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md).

### Appendix H.3 · Resolution of a windowed estimate on a training log

TeX label `app:window`. Related code and data: [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md).

### Appendix I.1 · The exclusion cap at the twenty-component configuration

TeX label `app:theiler`. Related code and data: [Theiler exclusion and transients](../../code/synthetic/delay_embedding_and_limits/theiler_exclusion/ReadMe.md).

### Appendix I.2 · The Theiler exclusion and the dimension of a transient

TeX label `app:exclusion`. Related code and data: [Theiler exclusion and transients](../../code/synthetic/delay_embedding_and_limits/theiler_exclusion/ReadMe.md).

### Appendix J · The training runs

TeX label `app:runs`. Related code and data: [Full-batch perceptron grokking](../../code/real/grokking_and_stability/perceptron_grokking/ReadMe.md).

### Appendix K · Descent at the edge of stability

TeX label `app:eos`. Related code and data: [Edge of stability](../../code/real/grokking_and_stability/edge_of_stability/ReadMe.md).

### Appendix L · The ceiling: embedding dimension against record length

TeX label `app:ceiling`. Related code and data: [Dimension ceiling](../../code/synthetic/delay_embedding_and_limits/dimension_ceiling/ReadMe.md).

### Appendix M · Two controls separating the estimator from roughness

TeX label `app:clocks`. Related code and data: [Roughness and phase controls](../../code/synthetic/signal_controls_and_sketches/phase_count_and_roughness/ReadMe.md).

### Appendix N · Compute cost and code availability

TeX label `app:compute`. Related code and data: [Trajectory sketch validation](../../code/synthetic/signal_controls_and_sketches/trajectory_sketch/ReadMe.md).

[Article and poster index](../ReadMe.md)
