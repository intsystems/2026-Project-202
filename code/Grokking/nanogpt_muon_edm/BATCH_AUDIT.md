# Audit of the eight-run fixed-`tau` nanoGPT experiment

## Scope and data integrity

The batch contains eight completed optimizer runs. Every log has a `RUNMETA`
and `RUNEND` record, 2,330 contiguous training-loss observations (steps
0--2,329), and 11 validation observations (steps 0, 250, ..., 2,250, 2,330).
The fixed-delay analysis produces 49 windows per run (400 observations,
stride 40), hence 392 window rows in total. No run is marked as diverged.

The rows are internally consistent, but the experiment has **one run per
optimizer and no logged seed**. The 49 windows overlap by 90% and therefore
must not be treated as 49 independent replicates. All batch-level comparisons
below are descriptive, not inferential.

## Main result

There is no evidence of a late-stage dimensionality collapse that is robust to
the estimator, detrending, and optimizer.

For each optimizer, “early” and “late” are the means of the first and last 12
sliding windows. Across the eight optimizers:

| series | method | decreasing | unchanged | increasing | mean late−early |
|---|---:|---:|---:|---:|---:|
| raw | FNN | 5 | 2 | 1 | -0.32 |
| raw | Cao | 1 | 0 | 7 | +0.46 |
| raw | simplex | 0 | 0 | 8 | +3.61 |
| raw | MLE ID | 0 | 0 | 8 | +1.83 |
| detrended | FNN | 5 | 1 | 2 | -0.27 |
| detrended | Cao | 3 | 0 | 5 | +0.10 |
| detrended | simplex | 0 | 0 | 8 | +1.99 |
| detrended | MLE ID | 0 | 0 | 8 | +0.68 |

Thus the weak FNN decrease is contradicted by simplex and MLE in every run.
Detrending reduces, but does not reverse, the MLE increase. The strongest
defensible claim is a negative/control result: in these scalar training-loss
logs, the proposed collapse is not a universal signature of successful
training and is not useful for ranking these optimizers.

## Optimizer outcomes

Seven final exact-model validation losses lie in the narrow range
3.2785--3.4049. Muon is lowest (3.2785), but single runs and different selected
learning rates (0.06 for the LMO-labelled group, 0.03 for the sign group) do not
support a statistically meaningful optimizer ranking. Runtime is also nearly
identical: 143.1--144.6 seconds, with a coefficient of variation of about 0.42%.

`EF21-MuonSign` needs separate interpretation. Its exact/server-model
validation loss improves only to 4.1967 and ends at 5.5198, while its logged
compressed broadcast model `W` improves monotonically to 3.3213. The original
comparison plot uses the exact model, so this is an algorithm-state mismatch,
not evidence that the transmitted model fails. See
`results_batch_tau1/01b_ef21_muonsign_exact_vs_w.*`.

## Estimator limitations that constrain interpretation

1. `tau=1`, window size 400, `E_max=15`, and five MLE neighbours are fixed
   sensitivity choices, not validated truths for these logs.
2. The legacy nearest-neighbour estimators do not use a Theiler exclusion, so
   temporally adjacent delay vectors can be selected as neighbours. With lag-1
   autocorrelation close to one in many windows, this can strongly bias local
   geometry estimates.
3. Simplex selects the upper boundary `E=15` in 32.7--42.9% of raw windows
   (depending on optimizer), indicating that the search range often does not
   contain a clear interior optimum.
4. MLE is computed in a 15-dimensional delay embedding but frequently returns
   values above 15 (87/392 raw and 212/392 detrended windows; maxima 16.69 and
   18.03). Those values should be described as outputs of the legacy heuristic,
   not literal intrinsic dimensions of the 15-dimensional embedding.
5. The training loss is a stochastic mini-batch observable under a changing
   learning-rate schedule. A one-dimensional, nonstationary observable does
   not by itself identify the topology of the full optimizer state.
6. The early and late groups overlap heavily within each group, and the final
   window is centred at step 2,120. Consequently, the analysis does not isolate
   the 40-step extension beginning at step 2,290.

## Reproducible artifacts

- `analyze_batch_tau1.py`: end-to-end parser, analysis, tables, and plots.
- `results_batch_tau1/optimizer_run_summary.csv`: run outcomes.
- `results_batch_tau1/tau1_early_late_all_methods.csv`: per-run deltas.
- `results_batch_tau1/cross_optimizer_direction_summary.csv`: aggregate
  direction counts shown above.
- `results_batch_tau1/05_all_method_deltas.*`: visual summary.

Recommended follow-up: log seeds, collect at least several independent runs per
optimizer at matched/tuned learning rates, test multiple window lengths,
delays, MLE neighbourhood sizes, and Theiler windows, and analyse richer
observables (weight/gradient statistics or held-out loss sampled densely).
