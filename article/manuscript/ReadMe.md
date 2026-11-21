# Shared article source

The scientific content is the AISTATS v3 revision of *Detecting Simplification in Neural Network Dynamics from Scalar Logs: Active Dimension in Controlled Recurrent Systems*. Edit the [sections](sections/ReadMe.md), [table inputs](tables/ReadMe.md), [figure files](figures/ReadMe.md), or bibliography here. [main_part.tex](main_part.tex) sets the order of the main paper and bibliography. [appendix_part.tex](appendix_part.tex) includes the appendices. All style entry points use both files.

## Build

Run from this directory with TeX Live and `latexmk`.

```bash
latexmk -pdf '-outdir=.build' -interaction=nonstopmode -halt-on-error aistats2027.tex icomp2026.tex main.tex artifacts_camera_ready.tex newinml.tex
```

The PDFs appear in `.build/` under their entry point names. [The article index](../ReadMe.md) lists the filenames used for the checked-in copies. AISTATS uses eight pages for the main paper. The other styles use ten for the same content.

For an editorial copy with blue text, set the flag at build time. For example:

```bash
latexmk -pdf '-outdir=.build' '-jobname=aistats2027_blue' '-usepretex=\def\claudedraft{1}' aistats2027.tex
```

## Article sections and experiment folders

Section numbers and titles apply to every style. The experiment READMEs describe any differences between the bundled runs and the article's reported protocols.

### Section 2.2 · Delay reconstruction and dynamical-systems theory

- [Delay coordinate reconstruction](../../code/synthetic/delay_embedding_and_limits/delay_embedding/ReadMe.md)

### Section 2.3 · Pooled local-dimension estimation

- [Observer and scale controls](../../code/synthetic/signal_controls_and_sketches/observer_controls/ReadMe.md)
- [Theiler exclusion and transients](../../code/synthetic/delay_embedding_and_limits/theiler_exclusion/ReadMe.md)
- [Unequal excitation](../../code/synthetic/signal_controls_and_sketches/unequal_excitation/ReadMe.md)

### Section 3.1 · Analytic oscillators and learning

- [Digits parameter subspace](../../code/synthetic/digits_validation/digits_classifier_head/ReadMe.md)
- [Dimension transitions](../../code/synthetic/signal_controls_and_sketches/dimension_transitions/ReadMe.md)
- [Estimator calibration](../../code/synthetic/digits_validation/estimator_calibration/ReadMe.md)
- [Known modes in linear learning](../../code/synthetic/poster_examples/known_linear_modes/ReadMe.md)
- [Oscillating diagonal matrix](../../code/synthetic/controlled_systems/oscillating_matrix/ReadMe.md)
- [Zero learning rate](../../code/synthetic/digits_validation/zero_learning_rate/ReadMe.md)

### Section 3.2 · Learned generators distinguish phases from harmonics

- [Four-tone RNN learning](../../code/synthetic/poster_examples/four_tone_rnn/ReadMe.md)
- [Roughness and phase controls](../../code/synthetic/signal_controls_and_sketches/phase_count_and_roughness/ReadMe.md)

### Section 3.3 · FORCE training from chaos to periodic output

- [FORCE RNN cycle](../../code/real/scalar_log_diagnostics/force_rnn_cycle/ReadMe.md)

### Section 4.2 · Transferring the detector to ResNet-18

- [ResNet-18 backbone freeze](../../code/real/scalar_log_diagnostics/resnet_freeze/ReadMe.md)

### Section 4.3 · Tracking latent information in a text VAE

- [Text VAE code loss](../../code/real/scalar_log_diagnostics/text_vae_latent_code/ReadMe.md)

### Section 4.4 · Detecting changes during grokking

- [Full-batch perceptron grokking](../../code/real/grokking_and_stability/perceptron_grokking/ReadMe.md)
- [Trajectory sketch validation](../../code/synthetic/signal_controls_and_sketches/trajectory_sketch/ReadMe.md)
- [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md)

### Section 4.5 · Selecting Walker2d policies under actuator shifts

- [Walker2d policy selection](../../code/real/model_selection_and_stopping/walker2d_policy_selection/ReadMe.md)

### Section 4.8 · Forecast model selection from single-neuron recordings

- [RNN forecast selection](../../code/real/model_selection_and_stopping/rnn_forecast_selection/ReadMe.md)

### Appendix A.2 · Recurrent, transient, and stochastic regimes

- [Delay coordinate reconstruction](../../code/synthetic/delay_embedding_and_limits/delay_embedding/ReadMe.md)

### Appendix B · Configurations and reproducibility

- [Delay lag sensitivity](../../code/synthetic/delay_embedding_and_limits/delay_lag_sweep/ReadMe.md)
- [Dimension ceiling](../../code/synthetic/delay_embedding_and_limits/dimension_ceiling/ReadMe.md)
- [Theiler exclusion and transients](../../code/synthetic/delay_embedding_and_limits/theiler_exclusion/ReadMe.md)

### Appendix C.1 · Analytic and driven-learning evaluation

- [Digits parameter subspace](../../code/synthetic/digits_validation/digits_classifier_head/ReadMe.md)
- [Digits function subspace](../../code/synthetic/digits_validation/digits_function_space/ReadMe.md)
- [Estimator calibration](../../code/synthetic/digits_validation/estimator_calibration/ReadMe.md)
- [Frozen nonlinear decoder](../../code/synthetic/controlled_systems/frozen_decoder/ReadMe.md)
- [Logistic regression](../../code/synthetic/controlled_systems/logistic_regression/ReadMe.md)
- [Online linear regression](../../code/synthetic/controlled_systems/online_linear_regression/ReadMe.md)
- [Oscillating diagonal matrix](../../code/synthetic/controlled_systems/oscillating_matrix/ReadMe.md)
- [Perceptron in a subspace](../../code/synthetic/controlled_systems/subspace_perceptron/ReadMe.md)
- [Zero learning rate](../../code/synthetic/digits_validation/zero_learning_rate/ReadMe.md)

### Appendix C.2 · Dense regression with endogenous modes

- [Known modes in linear learning](../../code/synthetic/poster_examples/known_linear_modes/ReadMe.md)

### Appendix C.3 · Learned autonomous generators

- [FORCE RNN cycle](../../code/real/scalar_log_diagnostics/force_rnn_cycle/ReadMe.md)
- [Four-tone RNN learning](../../code/synthetic/poster_examples/four_tone_rnn/ReadMe.md)

### Appendix D.1 · CNN and ResNet interventions

- [ResNet-18 backbone freeze](../../code/real/scalar_log_diagnostics/resnet_freeze/ReadMe.md)

### Appendix D.4 · Text VAE and preservation reference

- [Text VAE code loss](../../code/real/scalar_log_diagnostics/text_vae_latent_code/ReadMe.md)

### Appendix E · Additional learning settings

- [Edge of stability](../../code/real/grokking_and_stability/edge_of_stability/ReadMe.md)
- [Full-batch perceptron grokking](../../code/real/grokking_and_stability/perceptron_grokking/ReadMe.md)
- [Trajectory sketch validation](../../code/synthetic/signal_controls_and_sketches/trajectory_sketch/ReadMe.md)
- [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md)

### Appendix E.1 · Selecting policies for actuator shifts

- [Walker2d policy selection](../../code/real/model_selection_and_stopping/walker2d_policy_selection/ReadMe.md)

### Appendix G · Forecast selection and sampled-query estimation

- [RNN forecast selection](../../code/real/model_selection_and_stopping/rnn_forecast_selection/ReadMe.md)

### Appendix H · Comparing intervention rules

- [Noisy MNIST early stopping](../../code/real/model_selection_and_stopping/mnist_early_stopping/ReadMe.md)

[Article and poster index](../ReadMe.md)
