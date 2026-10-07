# Walker2d: coefficient sweep, 2026-09-30

Read **`report_final_ru.pdf`** or **`report_final_ru.md`** first: the consolidated
Russian report of 2026-10-01 leads with the demonstrated strengths of MG and
contains the setting, reproducible protocol, independent checks, primary and
secondary scalar results, checkpoint analysis, simple-baseline comparison,
runtime and interpretation boundaries. It supersedes the two separate reports
as the entry point; their detailed audit results remain available.

Rebuild the final report from the project root:

```powershell
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\final_figures.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\build_final_report.py
```

The earlier detailed reports are `report_lambda.pdf` and `report_baselines.pdf`.

## Final scalar-baseline audit

Read `report_baselines.pdf` (two pages, Russian) for the final recommendation:
close this setting as a supplementary positive example, not evidence that MG
outperforms cheap diagnostics. MG decreases more consistently under weak
regularization of the primary scalar (5/5 vs 4/5 seeds), but scalar recurrence
also captures the degradation contrast, at about 98 times lower runtime here.
These are descriptive comparisons of five continuation seeds, not statistical
superiority or validated failure-prediction accuracy.

`BASELINE_PROTOCOL.md` fixes the follow-up computations. `baselines_windows.csv`
contains all 1521 signal windows (169 trajectories, 3 scalars, 3 windows).
`baselines_records.csv`, `baselines_pairs.csv`, `baselines_by_seed.csv`, and
`baselines_summary.csv` show each aggregation level. `baselines_common4_*`
restrict every arm to the same 23 surviving reset pairs; `baselines_rebound_*`
compare coefficients 4 and 1 on the same 24 eligible triplets including control.
Both checks retain the conclusion, but cannot eliminate survivor bias relative
to all test episodes. `baselines_timing*.csv` and `baselines_provenance.json`
record the separate scalar-only benchmark. `baselines_seed_curves.pdf` shows
individual seed ratios for all four metrics of the primary signal.

Reproduce from the project root:

```powershell
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\compare_scalar_baselines.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\baseline_figures.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\build_baselines_report.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\package_results.py
```

No policy training was added. The previous report's stale runtime factor was
corrected from 29 to 17 to match its own 0.455/0.0266 timing table; the new
98x factor compares a different reference (scalar recurrence, not full state).

## What was run

- Pilot seed 230, coefficients 0, 0.25, 1, 4. See `pilot_summary.csv`.
- Confirmation seeds 231--235, same four coefficients: 20 trainings, 200 final tests.
- Five fixed checkpoints per confirmation run on reset 62001: 100 evaluations (final tests reused).
- MG action norm primary; action increment norm and mean action secondary. Window 2048, lag 8, E20, k20, Theiler312; E40 diagnostic saved. Three overlapping windows per eligible record.
- Timing: serial, one numerical thread, warmup + five repeats; not concurrent with training or MG workers.

All arms start from the same repair anchor. Seeds change continuation training, not independent initial pretraining. The test reset bank was reused from earlier experiments. No coefficient was selected by MG. This is a continuation replication, not a blinded new-environment benchmark.

## Data and aggregation

`confirmation_summary.csv`: per-training-seed paired ratios, success counts, reward ratios.
`confirmation_aggregate.csv`: median across five seeds, including **median** success count and **median** reward ratio. These are not totals or pooled means.
`confirmation_traces.csv`: only common eligible reset pairs; excluded failures never enter state ratios.
`action_mg_raw.csv`: all eligible records including unpaired ones, window validity and diagnostic ranges.
`action_mg.csv`: paired records; `action_mg_by_seed.csv`: reset medians within seed; `action_mg_summary.csv`: median of five seed medians.
`timecourse_raw.csv`, `timecourse_pairs.csv`, `timecourse_summary.csv`: all checkpoints, explicit eligibility, paired medians and counts.
`timing_raw.csv`, `timing_summary.csv`, `timing_environment.json`: runtime evidence.

The report and quality plot deliberately use total healthy episodes and ratios of pooled padded rewards (all 50 episodes per coefficient). This avoids hiding failed seeds behind a median. Paired MG/state diagnostics have 48, 47, 45, 24 common records for coefficients 0, 0.25, 1, 4. Baseline paired sets can differ across coefficients. The lambda4 result is conditional on surviving episodes and especially sparse on seed232 (one pair).

An early analysis attempt assumed all saved trajectories were complete; it failed on an empty action array after an early fall. The final `action_mg.py` uses eligibility and validates the 4096x6 action shape. No missing record is replaced by a zero MG.

## Reproduce in the existing project workspace

Run from `2026-Project-202`, using `.venv_walker/Scripts/python.exe` on Windows:

```powershell
.\.venv_walker\Scripts\python.exe confirm_smooth_lambda.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\summarize_confirmation.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\action_mg.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\checkpoint_timecourse.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\benchmark.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\figures.py
.\.venv_walker\Scripts\python.exe research_walker_smooth_lambda\build_lambda_report.py
```

Training skips completed runs. Rollouts and MG reuse matching caches; if changing settings, use a new output folder rather than reusing these caches. `train.py` loads `research_walker_repair/anchor`; its copy in this folder is retained for provenance. Estimator dependencies: `research_walker_phase_wide/features.py` and `code/actdim`. PDF requires XeLaTeX and Times New Roman. Full checkpoints, raw trajectories and logs remain in the workspace.

`walker_smooth_lambda_results.zip` is a compact report/data/source snapshot. It includes CSV/JSON evidence and scripts, but excludes policy checkpoints and trajectory NPZ files; therefore it is not a standalone training/replay bundle. The report can be rebuilt from it with XeLaTeX. SHA256 manifest and ZIP integrity are checked by `package_results.py`.

## Scope

The supported result is sensitivity of a scalar control-log statistic to moderate temporal policy regularization. Full-system active dimension is not identified. Smoothness and full-state recurrence checks disagree with phase-section dispersion. Here MG is slower than inexpensive smoothness and state-reference computations; a speed advantage must not be claimed from these results.
