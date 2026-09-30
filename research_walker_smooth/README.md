# Walker2d smooth-policy follow-up

Read `report_ru.pdf` or `report_ru.md` for the actual results. `PROTOCOL.md` was frozen before runs; `DIAGNOSTIC_ADDENDUM.md` explains the additional retrospective fixed-three-window control. Old experiments and manuscript are read-only.

## What is measured

PPO learns a walking policy. Auxiliary temporal regularization penalizes squared change of clipped action means on consecutive nonterminal rollout states. MG is measured on the physical motion of frozen learned policies, not on optimization logs. Smooth actions, recurrent full-state motion, finite-time perturbation growth, and exact active dimension must not be equated.

Pilot220 selected coefficient1 without MG. Five paired continuations221..225 compare zero/selected coefficient from one shared warm-start policy. They are not five independent-from-scratch initializations. Held-out states62001..62010 are common across arms/seeds. Every failure is retained; rates over50 episodes are descriptive, not50 independent training replicates.

## Reproduction

Python3.13, dependencies in `requirements.txt`. The archive includes `code/actdim`, required old helpers/data and the shared anchor. To retrain, use a fresh directory containing source/protocol/dependencies/anchor, without completed seed folders or cached outputs. Completed training refuses overwrite.

```
python research_walker_smooth/validate_training.py
python research_walker_smooth/diagnostic.py
python research_walker_smooth/old_probes.py
python research_walker_smooth/workflow.py pilot
python research_walker_smooth/workflow.py confirm
python research_walker_smooth/measure_new.py
python research_walker_smooth/benchmark.py
python research_walker_smooth/audit.py
python research_walker_smooth/summarize.py
python research_walker_smooth/make_report.py
python research_walker_smooth/build_report.py
```

`prepare_code.py` documents generation of the pinned SB3 PPO variant and copied measurement helpers. Do not regenerate against a different SB3 without reviewing the diff and repeating `validate_training.py`. `smooth_ppo.py` includes the actual used variant. Zero coefficient was tested against unmodified SB3 bit-for-bit over8192 transitions. Mean actions are clipped, gradients flow through both members of each state pair, episode boundaries are masked, and a separate RNG samples pair indices.

`audit.py` needs `anchor/step0000000/policy.zip` and `normalize.pkl` copied from `research_walker_repair/anchor`; the archive includes them. PDF building requires XeLaTeX with Times New Roman and standard packages. The Russian Markdown retains formulas in dollar environments. `benchmark.py` must run serially after training/analysis jobs; it records three repeats per arm for ordinary methods and one full68-probe series per arm. For a fresh timing repeat use a fresh benchmark folder so acquisition is not cached.

## Files

- `seed*/train.json`, `progress.jsonl`, `step*/policy.zip`, `normalize.pkl`: budgets, update losses, all checkpoint policies and frozen normalization.
- `validation.csv`: last3 checkpoints x5 states. `test.csv`: final10 held-out states. Failed trajectories remain saved.
- `trajectory.npz`: positions, velocities, actions, rewards, exact MuJoCo integration states for replay. `metrics.json`: J1/J2, full-state R/D, scalar alternatives and padded reward including failures.
- `MG_windows.csv`: fixed, cycle-matched and left-knee settings, all windows and E40 diagnostics. `MG_seed*.csv`, `paired_traces.csv`, `all_seeds.csv`: aggregates with common-reset counts and explicit assessability.
- `period.json`, `diagnostic/*_period.json`: all recurrence minima and disagreement between record halves. Cycle matching uses full state; it is not an autonomous scalar-only selector.
- `fixed600_2*`:68 signed probes over identical600-step horizon, fixed75-step phase radius, last150-step aggregation. Preserve falls and near-tangent flags. Not Lyapunov exponents.
- `diagnostic_summary.csv`: all five old continuations, matched scales, resampling, equal physical perturbation horizon.
- `audit.json`, `training_implementation_audit.json`: numerical/integrity checks. `MANIFEST.json`: SHA256 archive inventory.

The code copies portions of Stable-Baselines3 PPO.train (MIT license); the installed distribution license is included. The added loss is a temporal policy smoothness regularizer, not a claim of a new RL algorithm. No full CAPS implementation, spatial regularizer, or proof of dimensionality reduction is claimed.
