# Walker2d: complete experimental series

Entry point: **report_ru.pdf**, with editable source **report_ru.md**.
This report supersedes the lambda-only summary as the overview of the entire
Walker2d setting. Earlier reports and results remain unchanged. No training or
manuscript edit was performed to create this synthesis.

## Scope and provenance

Eight branches are included, with all pilot outcomes:

| Workspace folder | Scope | Main evidence |
| --- | --- | --- |
| research_walker_gait | Initial PPO, extended once; infeasible final pair | summary.json, report_ru.md, seed200/train_*.json |
| research_walker_repair | Stabilized PPO continuation | summary.json, all_seeds.csv, sensitivity.csv |
| research_walker_smooth | First paired smoothing study; scale audit; action-log follow-up | all_seeds.csv, action_mg_summary.csv, diagnostic_summary.csv, timings.csv |
| research_walker_imitation | Reference imitation without external phase | all_results.csv, selection.json, PROTOCOL.md |
| research_walker_phase | Known-phase reference tracking with narrow penalty | all_results.csv, selection.json, PROTOCOL.md |
| research_walker_phase_wide | Broader tracking penalty; five confirmation pairs | all_results.csv, sensor_robustness_summary.csv, timings.csv, report_ru.md |
| research_walker_combined | Four-arm tracking x smoothing pilot | pilot_summary.csv, PROTOCOL.md |
| research_walker_smooth_lambda | Four smoothing strengths; final scalar baselines | confirmation_summary.csv, action_mg_by_seed.csv, baselines_*.csv, timecourse_summary.csv |

`training_runs.csv` enumerates saved training segments. `series_inventory.csv`
counts policies by branch: 1+6+12+3+2+12+4+24=64. The initial policy was extended
once, so there are 65 segments. Total main-training transitions are 70,254,592;
repair's `additional_steps` and initial pilot's `resume` are handled explicitly.
Short technical validation trainings are excluded. This count does not imply
64 independent replications: arms, pilots and different tasks are included.

`two_cohorts_by_seed.csv` and `two_cohorts_summary.csv` compare the first
post-hoc action analysis (221--225) to the next study's lambda1 arm (231--235).
Each cohort shows 5/5 decreases in the three aggregate action logs. There is
no pooling across windows/logs to inflate the sample size, and all runs share
an initial pretrained policy. The first cohort's seed222 has only six shared
eligible resets and is descriptive under its original >=8-episode criterion.

`phase_confirmation.csv` preserves every confirmation pair and sensor setting.
The illustrative seed271 was selected after reviewing the complete series:
joint full-state improvement is 2/5, all-five-setting MG agreement is 1/5.
The cheap spectral entropy also decreased in that example.

`timing_comparison.csv` aggregates stored serial benchmark trials. Ratios to
perturbation suites are not ratios to a validated Lyapunov spectrum. Full-state
RD and scalar baselines remain cheaper. Different branches' times are not a
single harmonized machine benchmark. The final scalar-only comparison uses a
separate later measurement. Collection time is reported separately from compute.

Source paths and SHA256 hashes are in `source_manifest.json`; verified counts
are in `checks.json`. Authoritative per-branch CSVs/protocols override any older
narrative shorthand. In particular, the early variable-horizon perturbation
comparison was superseded by a fixed600-step audit, and phase-section D differs
from the earlier knee-peak D. No phase-period or spectrum is treated as exact.

## Rebuild

From the project root, using the existing environment:

```powershell
.\.venv_walker\Scripts\python.exe research_walker_overview\collect_evidence.py
.\.venv_walker\Scripts\python.exe research_walker_overview\figures.py
.\.venv_walker\Scripts\python.exe research_walker_overview\build_report.py
.\.venv_walker\Scripts\python.exe research_walker_overview\package_report.py
```

XeLaTeX and Times New Roman are used for the PDF. With the prebuilt figure PDFs,
only the build step is needed to recompile the report. `figures.py` regenerates
its figures solely from CSV files in this overview folder.

`walker_complete_report.zip` is a compact source/data/report snapshot. It
contains this report, audit tables and the root-level protocols, analysis code,
summary CSV/JSON/Markdown and training records of all eight branches. Large
checkpoint archives and raw NPZ trajectories remain in the workspace, so this
is not a standalone training replay package. ZIP integrity and member SHA256
hashes are verified by the packaging script.
