# Learned Walker2d gait and scalar MG

Read `report_ru.pdf` or `report_ru.md` for the actual outcome. `summary.json` distinguishes a failed pilot eligibility check from a completed MG comparison. Do not infer a positive result merely from successful software execution.

## What is learned and measured

PPO trains an observation-to-action policy in Walker2d-v5. Diagnostics measure autonomous robot motion under **frozen policies**, not the optimizer trajectory. Right-knee angle is the preselected scalar. Full-state recurrence, section dispersion and phase-adjusted perturbation growth provide separate references; none is declared an exact active dimension or Lyapunov spectrum.

`PROTOCOL.md` was written before training or MG. `RESOURCE_AMENDMENT.md` changes only allowed concurrency. `NUMERICAL_AUDIT.md` records the rejected batched-action perturbation implementation and exact-replay fix. There is no model/hyperparameter/seed search after seeing MG.

## Reproduce

Use Python 3.13; create a fresh environment and install `requirements.txt`. `environment_freeze.txt` records the original machine (including unrelated inherited packages), not a minimal install recipe. The archive preserves its parent-directory layout so `code/actdim` can be imported by `features.py`.

```powershell
python research_walker_gait/train.py --seed 200 --steps 2097152
python research_walker_gait/motion.py --seed 200
python research_walker_gait/motion.py --seed 200 --pair 2097152
```

Only if the initial pair fails, extend once:

```powershell
python research_walker_gait/train.py --seed 200 --steps 4194304 --resume 2097152
python research_walker_gait/motion.py --seed 200
python research_walker_gait/motion.py --seed 200 --pair 4194304
```

If this also fails, run `feasibility_report.py` and stop as specified in the protocol. Otherwise run the full four-anchor `perturb.py` on the earliest common pilot reset, then `select_pilot.py` **before any MG or confirmation runs**. The latter freezes lag, horizon and probe count from independent evidence. Run `confirm.py`, plus pilot `perturb.py --seed 200` and `features.py --seed 200`. Finally run `benchmark.py`, `verify.py`, `summarize.py`, `build_report.py` and `package.py`. PDF compilation requires XeLaTeX with Times New Roman. `finish.py` automates this continuation for an already running 4M pilot.

Completed outputs are cached; train.py refuses to overwrite a completed horizon. Reproduce in a new extracted directory, retaining original results separately. Intermediate snapshots are saved before the PPO update at their rollout boundary; the final snapshot is saved after the last update. The pilot extension restarts the SB3 monitor files; `progress.jsonl` preserves checkpoint-level reward summaries for both intervals, but individual episode monitor records from the first interval are not retained.

## Artifact interpretation

- `seed*/evaluation.csv`: every saved checkpoint and all three resets, including falls and ineligible records.
- `step*/policy.zip`, `normalize.pkl`: trusted local model/normalization artifacts.
- `reset*/trajectory.npz`: physical qpos/qvel, burn-in, rewards and integration states for exact replay.
- `metrics.json`, `recurrence.csv`: independent full-state and inexpensive scalar measurements.
- `perturb_*.csv`, `_curves.npz`: every signed perturbation, with falls and nearly tangent cases retained. `exploratory_batched` is excluded from scientific results.
- `MG_windows.csv`: all predeclared windows and validity flags. These files exist only after a usable pilot pair is found.
- `all_seeds.csv`, `paired_traces.csv`, `sensitivity.csv`: descriptive aggregation without seed replacement, if confirmation runs take place.
- `benchmark.json`, `timings.csv`: serial component costs on one fixed confirmation pair, if such a pair exists. Common acquisition is charged separately.
- `MANIFEST.json`: SHA256 and byte count for archive contents; `package.py` checks every stored hash and ZIP CRC.

For the inexpensive autocorrelation score we use `1 - min_lag mean((x[t+lag]-x[t])**2)/(2*var(x))`, lag20..250. It equals a familiar autocorrelation form under stationary matching variances; it is not separately computed Pearson correlation. Spectral entropy excludes DC and is divided by log(number of non-DC Fourier bins). Peak CV uses the fixed peak detector in the protocol. Smaller MG/entropy/CV and larger autocorrelation are descriptive directions, not guarantees of dynamical stability.

The environment is a small simulation. Savings relative to many perturbation continuations do not imply savings over inexpensive recurrence measures or establish equivalent diagnostic power. No manuscript source is changed by this experiment.
