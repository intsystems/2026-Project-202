# Shared article source

The manuscript for *Detecting Simplification in Neural Network Dynamics from Scalar Logs* is based on the AISTATS v3 revision. Edit the [sections](sections/ReadMe.md), [table inputs](tables/ReadMe.md), [figure files](figures/ReadMe.md), or bibliography here. [main_part.tex](main_part.tex) sets the order of the main paper and bibliography. [appendix_part.tex](appendix_part.tex) includes the appendices. All style entry points use both files.

## Build

Run from this directory with TeX Live and `latexmk`.

```bash
latexmk -pdf '-outdir=.build' -interaction=nonstopmode -halt-on-error aistats2027.tex icomp2026.tex main.tex artifacts_camera_ready.tex newinml.tex
```

The PDFs appear in `.build/` under their entry point names. [The article index](../ReadMe.md) lists the filenames used for the checked-in copies.

The [Codex review copy](../artifacts_camera_ready_codex.pdf) uses the camera-ready layout and shows edits from the latest revision in teal.

```bash
latexmk -pdf '-outdir=.build' -interaction=nonstopmode -halt-on-error artifacts_camera_ready_codex.tex
```

For an editorial copy with blue text, set the flag at build time. For example:

```bash
latexmk -pdf '-outdir=.build' '-jobname=aistats2027_blue' '-usepretex=\def\claudedraft{1}' aistats2027.tex
```

## Reproduce reported measurements

- For the digit-classifier errors, select `qp` records with learning enabled and ranks 1, 3, 5, and 8. Average `abs(MG - traj_PR)` within each recorded quantity. The pooled error uses all quantities except `loss_step` and `acc_probe`. [Digit-classifier records](../../code/synthetic/digits_validation/digits_classifier_head/logs/sweep_raw.csv)
- For the phase comparison, compute the mean and sample standard deviation of `MG_n0` and `PR` over repeats within each task. The experiment page gives the commands to redraw the plots and to train a new generator. [Generator measurements](../../code/synthetic/poster_examples/four_tone_rnn/logs/runs.csv), [Generator code and commands](../../code/synthetic/poster_examples/four_tone_rnn/ReadMe.md)
- For forecasting, average errors across tasks within a repeat, then compute the mean and sample standard deviation across repeats. For times, take medians across repetitions within each input and method, then across inputs. The saved choices also reproduce exact agreement between full MG and 128-query MG. [Forecast scores and timing repetitions](../../code/real/model_selection_and_stopping/rnn_forecast_selection/logs/ReadMe.md)

Full-precision detector parameters and scoring records are in `aistats_article_v3/new_results/` at commit `f6b579b1`. The selected warning scores and calibration records are under `research_mg_wins/spikes/` at the same commit.

## Article sections and experiment folders

Section numbers and titles apply to every style. The experiment READMEs describe any differences between the bundled runs and the article's reported protocols.

<a id="section-22--delay-reconstruction-and-dynamical-systems-theory"></a>

### Section 2.1 · Delay reconstruction

- [Delay coordinate reconstruction](../../code/synthetic/delay_embedding_and_limits/delay_embedding/ReadMe.md)

<a id="section-23--pooled-local-dimension-estimation"></a>

### Section 2.2 · Pooled local-dimension estimation

- [Observer and scale controls](../../code/synthetic/signal_controls_and_sketches/observer_controls/ReadMe.md)
- [Theiler exclusion and transients](../../code/synthetic/delay_embedding_and_limits/theiler_exclusion/ReadMe.md)
- [Unequal excitation](../../code/synthetic/signal_controls_and_sketches/unequal_excitation/ReadMe.md)

### Section 2.3 · Experimental questions

- [Questions, experiments, and comparisons](sections/method.tex)

<a id="section-32--learned-generators-distinguish-phases-from-harmonics"></a>

### Section 3.1 · Separating independent oscillations from harmonics

- [Four-tone RNN learning](../../code/synthetic/poster_examples/four_tone_rnn/ReadMe.md)
- [Roughness and phase controls](../../code/synthetic/signal_controls_and_sketches/phase_count_and_roughness/ReadMe.md)

<a id="section-33--force-training-from-chaos-to-periodic-output"></a>

### Section 3.2 · FORCE training from chaos to periodic output

- [FORCE RNN cycle](../../code/real/scalar_log_diagnostics/force_rnn_cycle/ReadMe.md)

<a id="section-42--transferring-the-detector-to-resnet-18"></a>

### Section 4.1 · Detecting restricted parameter updates

- [ResNet-18 backbone freeze](../../code/real/scalar_log_diagnostics/resnet_freeze/ReadMe.md)

<a id="section-43--tracking-latent-information-in-a-text-vae"></a>

### Section 4.2 · Tracking latent information in a text VAE

- [Text VAE code loss](../../code/real/scalar_log_diagnostics/text_vae_latent_code/ReadMe.md)

<a id="section-48--forecast-model-selection-from-single-neuron-recordings"></a>

### Section 4.4 · Selection from scalar logs

- [RNN forecast selection](../../code/real/model_selection_and_stopping/rnn_forecast_selection/ReadMe.md)

### Appendix A.2 · Recurrent, transient, and stochastic regimes

- [Delay coordinate reconstruction](../../code/synthetic/delay_embedding_and_limits/delay_embedding/ReadMe.md)

### Appendix B · Configurations and reproducibility

- [Delay lag sensitivity](../../code/synthetic/delay_embedding_and_limits/delay_lag_sweep/ReadMe.md)
- [Dimension ceiling](../../code/synthetic/delay_embedding_and_limits/dimension_ceiling/ReadMe.md)
- [Theiler exclusion and transients](../../code/synthetic/delay_embedding_and_limits/theiler_exclusion/ReadMe.md)

<a id="section-31--analytic-oscillators-and-learning"></a>

<a id="appendix-c1--analytic-and-driven-learning-evaluation"></a>

### Appendix C.1 · Oscillations in the diagonal matrix and digit classifier

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

<a id="section-44--detecting-changes-during-grokking"></a>

### Appendix E · Additional learning settings

- [Edge of stability](../../code/real/grokking_and_stability/edge_of_stability/ReadMe.md)
- [Full-batch perceptron grokking](../../code/real/grokking_and_stability/perceptron_grokking/ReadMe.md)
- [Trajectory sketch validation](../../code/synthetic/signal_controls_and_sketches/trajectory_sketch/ReadMe.md)
- [Transformer grokking](../../code/real/grokking_and_stability/transformer_grokking/ReadMe.md)

<a id="section-45--selecting-walker2d-policies-under-actuator-shifts"></a>

### Appendix E.1 · Selecting policies for actuator shifts

- [Walker2d policy selection](../../code/real/model_selection_and_stopping/walker2d_policy_selection/ReadMe.md)

### Appendix G · Forecast selection and sampled-query estimation

- [RNN forecast selection](../../code/real/model_selection_and_stopping/rnn_forecast_selection/ReadMe.md)

### Appendix H · Comparing intervention rules

- [Noisy MNIST early stopping](../../code/real/model_selection_and_stopping/mnist_early_stopping/ReadMe.md)

[Article and poster index](../ReadMe.md)
