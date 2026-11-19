# Full-batch perceptron grokking data

Modular-arithmetic and polynomial tasks used to compare generalizing and memorizing full-batch perceptron runs.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| `arithmetic__*_train.csv` (7 files, [example](arithmetic__a_add_train.csv)), `polynomial__*_train.csv` (8 files, [example](polynomial__f_p1_p97_train.csv)), `g_p2*_train.csv` (2 files, [example](g_p2_train.csv)) | Training and validation losses, accuracies, and weight norm by step. |
| [`milestones.csv`](milestones.csv), `train_perceptron_*__milestones.csv` (4 files, [example](train_perceptron_poly__milestones.csv)) | Memorization and generalization steps and final accuracies for the original runs and sketch variants. |
| [`dimension_probe.csv`](dimension_probe.csv), [`dimension_probe_poly.csv`](dimension_probe_poly.csv) | Windowed MG and diagnostics for arithmetic and polynomial tasks. |
| [`dimension_probe_summary.csv`](dimension_probe_summary.csv), [`dimension_probe_summary_poly.csv`](dimension_probe_summary_poly.csv) | Run-level summaries of the same scalar diagnostics. |
| [`pr_vs_window.csv`](pr_vs_window.csv) | Trajectory participation ratio and derived statistics as window length varies. |
| [`repr_analytic.csv`](repr_analytic.csv), [`repr_reference.csv`](repr_reference.csv), [`repr_decomposition.csv`](repr_decomposition.csv) | Analytic representations, reference ranks, and checks of expressibility. |
| [`repr_measured.csv`](repr_measured.csv), [`repr_trajectory.csv`](repr_trajectory.csv) | Representation statistics measured on saved run snapshots. |
