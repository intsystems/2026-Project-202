# Numerical claims and source map

Paths below are relative to the project root. Copies of numerical inputs are under `evidence/`; `evidence/manifest.json` records SHA256 hashes. The paper uses saved results and does not report any new training performed during this rewrite.

| Claim / manuscript location | Source and interpretation |
| --- | --- |
| Analytic matrix MAE 0.27; controlled digits head error 0.90; driven MLP error 0.92 | `aistats_article/aistats2027.tex`, tables `tab:ladder`, `tab:aggregation`, `tab:obs`. Legacy aggregate values were transcribed, not recomputed from original training. Head errors use the measured covariance reference, not exact phase count. The driven-target MLP fails the zero-learning-rate observer control. |
| Known-mode regression 4.97 to 2.56; rank sweep | `research_known_modes_results/report_ru.md`, `stationary_summary.csv`. This is undamped heavy-ball on a designed Hessian; seeds rotate the same loss dynamics. |
| Generator 35 learned networks, 5 seeds, harmonic ordering, MG 1.15/2.84/5.90/5.93 | `research_generator/results/runs.csv`, `summary.json`, `REPORT_E9_ru.md`. Seven tasks share seeds; four-window medians are the per-run unit. Phase counts describe the intended/reproduced output structure, with finite-time spectral checks; higher counts are not recovered exactly. |
| FORCE 3.56–5.07 to 1.08–1.13; strict criterion 4/5 | `research_force_motion/results_chaotic/report_summary.json`, `PROTOCOL.md`. Five seeds selected by pretraining Lyapunov, not by MG; the original unscreened cohort is separate. |
| CNN AUC 0.93056; strongest comparison 0.98958; baselines | `research_trajectory_reference/results_graded/summary.json`, keys `auc_events_vs_base_batchup`, `auc_strongest_vs_all_controls`. These use different comparison sets. |
| CNN event changes and independent references | `research_trajectory_reference/results_graded/per_run.csv`. Table generated in `prepare_assets.py`; includes all 13 intervention/observer-control arms. |
| 27/28, 0/16; all detector comparators | `research_trajectory_reference/results_detector/test_per_run.csv`, `test_overall.csv`. Recomputed from per-run booleans. Seven strong types on four seeds; four control types per seed, including transformations of base logs. |
| ResNet 6/6, 0/24; -27.6% / -30.8% | `research_trajectory_reference/results_resnet/e7_results/{scratch,finetune}/{per_run.csv,detector_per_run.csv}`. Three seed IDs reused in both training settings, not six disjoint data samples. Other intervention types fail the transferred MG detector. |
| VAE 9/9; q=0.517 [0.480,0.542]; protection ratio 1.286 [1.179,1.418] | `research_text_vae/summary.json`, `protection/summary.json`, `protection/all_seeds.csv`. Pilot zero excluded, same nine branches for both comparisons. CI bootstrap unit is paired seed. |
| Cyclic VAE: all budget counts and 28 vs 38 at budget 12 | `research_text_vae/cyclical/all_scores.csv`, confirmation seeds 101–109. Tables are summed from per-seed scores; 41 eligible events. |
| ReLU death: 12/21, 12/31; competitors | `research_trajectory_reference/results_collapse/summary.json`, `REPORT_E8_ru.md`. Two catastrophic runs excluded from the eligible-event score by the recorded rule; non-events include transformed base logs. |
| Grokking: four positive runs, two controls, 24/26 usable cells | `aistats_article/aistats2027.tex`, Section 7.3 and Appendix H.3. Cells reuse the same runs; no exact active-dimension interpretation. |
| Walker ratios 0.801 / 0.604, 5/5; full-gait 2/5 | `research_walker_overview/report_ru.md`. Reused initial policy; command regularity is distinguished from full gait stability. |
| MNIST: 4/5, median 4.8% | `research_mnist_dynamics/report_ru.md`. Fixed-probe primary result, not the more favorable large-window sensitivity. |
| CNN 9.49ms, 39.06KiB, 559.46MiB | `research_trajectory_reference/results_graded/summary.json`, key `cost`. Existing scalar processing; norm computation still costs O(P). |
| FORCE 14–19x full-spectrum speedup; 5–10x with diagnostic | `research_force_motion/results_chaotic/report_summary.json`, `benchmark_median`; `benchmark.csv`. Ratios of paired endpoint medians: 12.006/.835 and 11.551/.598; diagnostic includes E20+E40. Largest exponent and simple baselines are faster. |
| Oscillators 22–23x at N=128, 1536 samples | `research_sync_control_results/computation_benchmark_repeated.csv`; scalar timing includes MG and LB together. `same_seed_timing.csv` shows the opposite speed ordering at N=64 with 4096-point scalar windows. |
| VAE 12.89ms vs 209.32ms, full cost 54.68s vs periodic 2.72s | `research_text_vae/cyclical/benchmark.json`, `costs.csv`. End-to-end totals are component-cost estimates, not independent runtime measurements. |

## Checks performed for this rewrite

- Figure coordinates and generated tables use the stored CSV/JSON values.
- Main CNN detector counts are recomputed from individual records.
- Generator training-success count, principal VAE counts, and AUC values are checked in `prepare_assets.py`.
- `validate_claims.py` additionally checks the ResNet counts, confidence intervals, monitoring totals, and timing ratios quoted in the text.
- Black/blue PDF text must match exactly, the main text must end by page eight, and the compiled blue PDF must contain blue text.
- These are manuscript/evidence checks. They are not a new full training rerun, a new validation of every historical implementation, or an independent author attestation.
