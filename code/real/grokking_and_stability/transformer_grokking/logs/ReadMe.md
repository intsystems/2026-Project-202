# Transformer grokking data

Modular-addition and permutation-composition runs, including runs with weight decay and controls. Use the rank and matched-window tables to reconstruct the article's main figure.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| `sketched__*_train.csv` (6 files, [example](sketched__mod_wd0_train.csv)), `extended__*_train.csv` (7 files, [example](extended__grokpos_s0_train.csv)), [`mod_wd1_train.csv`](mod_wd1_train.csv) | Recorded train and validation losses and accuracies, plus weight norms and any extra logged gradients. |
| [`milestones.csv`](milestones.csv), [`rank_milestones.csv`](rank_milestones.csv), [`rank_summary.csv`](rank_summary.csv), `train_transformer_*__milestones.csv` (2 files, [example](train_transformer_extended__milestones.csv)) | Memorization and generalization steps, final accuracy, and the run or trajectory window settings. |
| [`rank_windows.csv`](rank_windows.csv) | Windowed participation ratio, sketch variation, weight norm, and displacement. |
| [`rank_dip.csv`](rank_dip.csv), [`rank_dip_controls.csv`](rank_dip_controls.csv), [`rank_dip_controls_aligned.csv`](rank_dip_controls_aligned.csv) | Depth and timing of the dip, recovery measures, and comparisons to controls. |
| [`real_logs_windows.csv`](real_logs_windows.csv), [`real_logs_summary.csv`](real_logs_summary.csv) | Scalar MG diagnostics per window and by run. |
| [`headline_trace.csv`](headline_trace.csv) | Scalar MG and trajectory statistics aligned on the same windows for the main figure. |
| `grid_*.csv` (3 files, [example](grid_train_loss.csv)) | Sensitivity of the matched-window result to embedding dimension, delay, neighbors, exclusion, and detrending. |
| [`surrogates.csv`](surrogates.csv), [`surrogate_summary.csv`](surrogate_summary.csv), [`surrogate_seed_spread.csv`](surrogate_seed_spread.csv) | Observed and surrogate dip scores, per-seed comparisons, and spread. |
| [`exp8_at_20k.csv`](exp8_at_20k.csv), [`exp8_outcomes.csv`](exp8_outcomes.csv) | Extended-run outcomes at 20,000 steps and over the full run. |
