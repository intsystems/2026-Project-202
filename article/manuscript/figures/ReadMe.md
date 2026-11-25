# Article figures

| File | Article content |
| --- | --- |
| [icomp_method.pdf](icomp_method.pdf) | Delay reconstruction and neighbor distances for a periodic signal and a text VAE loss record. |
| [components.pdf](components.pdf) | Known-mode regression and the learned generator phase comparison. |
| [force_training_transition.pdf](force_training_transition.pdf) | Fixed-weight FORCE simulations, delay reconstruction, and windowed MG at selected training steps. |
| [vae_real_log.pdf](../../poster/figures/vae_real_log.pdf) | Reconstruction loss on fixed evaluation sentences, MG, and sentence-code mutual information. |
| [resnet_freeze_trace.pdf](resnet_freeze_trace.pdf) | Weight-norm log, MG alarm, and independent update rank after freezing the ResNet backbone. |
| [grokking_matched_traces.pdf](grokking_matched_traces.pdf) | Weight norm, MG@weight norm, detrended logit PR, and accuracy during the rise in validation accuracy on modular addition. |
| [grokking_other_runs.pdf](grokking_other_runs.pdf) | MG@weight norm and train and validation accuracy for the other two modular-addition runs with the same weight decay. |
| [mnist_stop_trace.pdf](mnist_stop_trace.pdf) | Weight norm, loss on fixed training images, decision thresholds, and paired stopping outcomes in the intervention-rules appendix. |
| [ssl_probe_scatter.pdf](ssl_probe_scatter.pdf) | MG@gradient norm and test accuracy of a linear classifier on frozen features by self-supervised configuration. |
| [slingshot_warning.pdf](slingshot_warning.pdf) | A recorded loss spike with the frozen MG and roughness warnings. |
| [single_neuron_forecasts.pdf](single_neuron_forecasts.pdf) | Forecasts from two model-selection rules on mixed- and independent-frequency targets. |
| [icomp_shapes.pdf](icomp_shapes.pdf) | Delay reconstructions in Appendix A.2. |

[Manuscript index](../ReadMe.md)

Figure labels use `MG@series` for MG calculated from the named input. The summary PDFs `components.pdf`, `applications.pdf`, and `icomp_regimes.pdf` use the existing vector plots with updated labels.

## Rebuild the method figure

```bash
python article/manuscript/figures/plot_method_geometry.py --output article/manuscript/.build/reproduced
```

Both panels use the text VAE estimator settings. The recorded example uses steps 2,048 through 2,559 of the [baseline probe loss](../../../code/real/scalar_log_diagnostics/text_vae_latent_code/logs/vae_base_log.csv). The middle delay vector is the query. The plots show the first and last coordinates, and the neighbor search uses the full vectors.

The appendix diagram `icomp_shapes.pdf` is `fig_shapes.pdf` produced by the [delay reconstruction plotter](../../../code/synthetic/delay_embedding_and_limits/delay_embedding/analyze_logs.py). Run that script with `--output article/manuscript/.build/reproduced` from the repository root. Its inputs store the first two forward-delay coordinates.

## Rebuild the FORCE and VAE figures

Run from the repository root after installing the [code dependencies](../../../code/ReadMe.md#run-the-code):

```bash
python article/manuscript/figures/plot_force_training_transition.py --output article/manuscript/.build/reproduced
python code/real/scalar_log_diagnostics/text_vae_latent_code/analyze_logs.py --output article/manuscript/.build/reproduced
```

Both scripts recompute MG from the scalar samples committed under the respective experiment's `logs/`. The VAE script reads the separately measured mutual information from the bundled CSVs. That quantity uses the encoder's posterior distributions.

The FORCE figure shows the median and range of MG across valid windows at each training step. It excludes windows flagged by the estimator and leaves a gap when a simulation has no valid windows.

To generate new rollouts and VAE measurements, run the following in a fresh checkout. The VAE run also requires PyTorch and the two Penn Treebank files downloaded by its training script. The recorded VAE environment used Python 3.13.12 and PyTorch 2.10.0. Retain the bundled logs for exact comparisons with the values in the paper.

```bash
python code/real/scalar_log_diagnostics/force_rnn_cycle/run.py --n 256 --gain 1.5 --seeds 7 \
  --checkpoints $(seq 0 25 1000) 1250 1500 1750 2000 2500 3000 3500 4000 5000 6000 8000 10000 12000 16000 20000 24000 28000 32000 36000 40000 \
  --scalar-rollouts --out article/manuscript/.build/force-replay
python article/manuscript/figures/plot_force_training_transition.py --rollout-dir article/manuscript/.build/force-replay/seed_7 --output article/manuscript/.build/replayed
python code/real/scalar_log_diagnostics/text_vae_latent_code/collect_logs.py --seed 1
python code/real/scalar_log_diagnostics/text_vae_latent_code/analyze_logs.py --run-dir code/real/scalar_log_diagnostics/text_vae_latent_code/generated/seed1 --output article/manuscript/.build/replayed
```

## Rebuild the application traces

```bash
python article/manuscript/figures/plot_application_traces.py --output article/manuscript/.build/reproduced
```

The script uses the bundled [ResNet](../../../code/real/scalar_log_diagnostics/resnet_freeze/logs/ReadMe.md), [grokking](../../../code/real/grokking_and_stability/transformer_grokking/logs/ReadMe.md), and [MNIST](../../../code/real/model_selection_and_stopping/mnist_early_stopping/logs/ReadMe.md) CSVs. The plotted detector alarms, stop decisions, and generalization time come from the saved records. For the two additional modular-addition runs, the script recomputes MG from each recorded weight-norm series using the settings in the validation-rise analysis.

## Rebuild the selection and warning figures

```bash
python article/manuscript/figures/plot_selection_and_warning.py --output article/manuscript/.build/reproduced
```

- The [standalone Walker2d comparison](walker2d_selection.pdf) uses the four evaluation repeats in [checkpoint_selections.csv](../../../code/real/model_selection_and_stopping/walker2d_policy_selection/logs/checkpoint_selections.csv). It compares reward in the perturbed trials with the reward from choosing by performance in the unperturbed trials.
- The SSL points average the five evaluation repeats in [ssl_probe_scores.csv](data/ssl_probe_scores.csv). The columns come from `results_main/test_final.csv` and `results_confirm/confirm_final.csv` under `aistats_article_v3/evidence/oct07_campaign/ssl` at commit `f6b579b1`.
- The slingshot plot uses [logged window statistics](data/slingshot_windows.csv) and [frozen thresholds and onsets](data/slingshot_example.json) from `research_mg_wins/spikes` at commit `f6b579b1`. The loss curve shows recorded block means with minima and maxima. The example is the first eligible held-out event with different warning times and a rise of over one log unit in the recorded loss envelope.
- The forecast panels use the M4 and T4 recordings from evaluation repeat 681 and the corresponding [saved model choices](../../../code/real/model_selection_and_stopping/rnn_forecast_selection/logs/query_fresh_comparisons.csv). The script refits each chosen model on the full prefix and checks that its error matches the saved outcome.
