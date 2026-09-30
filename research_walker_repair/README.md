# Walker2d stable continuation experiment

This folder preserves a new experiment; `research_walker_gait` remains the original failed run. Read `report_ru.pdf` / `report_ru.md` and `summary.json` for the actual outcome. Software completion, stable reward and independent dynamical simplification are distinct outcomes.

## Design

PPO starts from the FIRST eligible earlier snapshot of the old experiment, seed200 at655360 transitions. `anchor/` includes exactly those policy and normalization files. Five confirmation runs share this initialization but have different fine-tuning randomness; they must not be described as independently trained-from-scratch policies. The pilot is excluded from confirmation counts. `PROTOCOL.md` records all gates, validation/test seeds, MG configuration selection and timing rules before measurements. Frozen `selection.json` contains its SHA256.

The repair changes learning-rate schedule, rollout/batch/epoch settings, clipping and KL control, freezes normalization and extends training episodes. It is a combined repair, not a causal ablation. The expensive reference is phase-adjusted finite-horizon perturbation growth, not a Lyapunov spectrum. MG measures physical motion under a frozen policy, not the optimizer trajectory.

## Reproduce in a fresh directory

Keep the archive parent-directory layout: measurement code imports the unchanged `code/actdim` package. Use Python3.13 and install `requirements.txt`. `environment_freeze.txt` is an exact snapshot of the original environment, including inherited unrelated packages.

```powershell
python research_walker_repair/train.py --seed 210
python research_walker_repair/evaluate.py --label seed210
python research_walker_repair/evaluate.py --label seed210 --gate
python research_walker_repair/freeze_config.py seed210
python research_walker_repair/confirm.py
python research_walker_repair/evaluate.py --label seed210 --split test --steps 0 1048576
python research_walker_repair/evaluate.py --label seed210 --pair
python research_walker_repair/measure.py --label seed210
python research_walker_repair/benchmark.py
python research_walker_repair/benchmark_aligned.py
python research_walker_repair/old_final_control.py
python research_walker_repair/verify.py
python research_walker_repair/summarize.py
python research_walker_repair/build_report.py
python research_walker_repair/package.py
```

Only if the first pilot fails the predeclared validation gate, use the single reserved `--conservative` restart from the same anchor and pass label `seed210_B` to evaluation/freezing. It was not required in the delivered run. Do not tune on MG or rerun unsuccessful confirmations with replacement seeds.

Training refuses to overwrite a completed run. Evaluation and MG cache completed trace outputs. For a fresh reproduction, copy the scripts, protocol, requirements, `anchor/`, `code/actdim`, and the old-final-control model/normalization into a separate directory, without copying seed result folders or cached metrics. `freeze_config.py` requires no pre-existing MG windows and a passed pilot gate. Intermediate and final snapshots are both taken after PPO updates. `model.num_timesteps` in new snapshots counts additional transitions, with the anchor's original655360 recorded separately.

## Data

- `seed*/train.json`, `progress.jsonl`: actual training time, parameter changes, approximate KL, learning-rate schedule, losses and clipping fractions.
- `seed*/validation.csv`:45 outcomes, nine snapshots times five reset seeds, including failures. `gate.json` records the last-three-snapshot gate.
- `seed*/test.csv`, `pair.json`:20 outcomes, shared anchor and final policy on10 unseen resets. Preserve failed walking and scalar-ineligible traces separately.
- `step*/policy.zip`, `normalize.pkl`: model and frozen normalization. `reset*/trajectory.npz`: physical motion and full integration states for exact replay.
- `metrics.json`, `recurrence.csv`: full-state R/D and cheap scalar baselines; `MG_windows.csv` retains all predeclared sensors/windows/lags/validity flags, with E40 a separate diagnostic.
- `perturb_2*`:68 signed perturbations of one preselected held-out trace; falls and nearly tangent cases retained. `shared_anchor_probes/` is a cache of identical baseline computations, protected by a file lock.
- `shared_anchor_MG/` caches identical baseline windows by full qpos-array SHA256 and frozen tau. It does not alter MG values; copied baseline timing fields are not used in the serial benchmark. Exact initial policy equality is checked in the audit.
- `all_seeds.csv`, `paired_traces.csv`, `sensitivity.csv`: aggregated results without replacing failed repetitions. Pilot rows are explicitly marked.
- `benchmark.json`, `timings.csv`: serial warmed component costs after training; common data acquisition is listed separately. These values are not parallel training throughput.
- `audit.json`: shared starting weights, distinct changed final actors, frozen normalization, complete checkpoint/reset grid, recomputed independent metrics, exact zero replay, raw perturbation arithmetic and MG window coverage.
- `MANIFEST.json`: hashes and sizes; the packager verifies every archived hash and ZIP CRC.

PDF generation uses XeLaTeX and Times New Roman. The report is self-contained in Russian. The old experiment and main manuscript are not changed.
