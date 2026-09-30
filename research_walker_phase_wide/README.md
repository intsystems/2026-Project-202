# Walker2d reference-gait experiments

`report_ru.pdf` / `report_ru.md` summarize THREE separate experiments, including failed pilots. The archive also contains `research_walker_imitation` and `research_walker_phase`. No manuscript changes. The older smoothing experiment remains unchanged.

1. **Phase-free orbit imitation**: 17D autonomous observation, reference selected from old pilot anchor record; add nearest-reference state reward penalty. Pilot240, coefficients0/1/3. Independent validation selected no coefficient. All three arms evaluated on10 new states; no5-seed confirmation launched.
2. **Phase-conditioned local tracking**: same reference plus supplied continuous phase (19D observation). Start close to the reference cycle. Pilot260, coefficients0/3, narrow reward width. Original health criteria/physics, no projection/filter during evaluation. This is not original default-reset Walker2d and not autonomous recurrent motion.
3. **Broad-band phase tracking**: same phase task, reward width fixed from control260 validation error; new pilot270, coefficients0/3. See selection.json for expansion decision. This does not tune MG. All unsuccessful attempts retained.

## Reproduce

Use Python3.13, `requirements.txt`, XeLaTeX + Times New Roman for report. Run scripts in fresh output folders containing sources/reference/anchor/protocol but no completed seed folders. Training refuses completed outputs; measurement caches outputs. Exact initial anchors and reference points are archived.

```
python research_walker_imitation/validate.py
python research_walker_imitation/workflow.py pilot
python research_walker_imitation/postpilot.py
python research_walker_imitation/summarize.py
python research_walker_phase/validate.py
python research_walker_phase/workflow.py pilot
python research_walker_phase/postpilot.py
python research_walker_phase/summarize.py
python research_walker_phase_wide/validate.py
python research_walker_phase_wide/workflow.py pilot
python research_walker_phase_wide/postpilot.py
python research_walker_phase_wide/whole_cycle.py
python research_walker_phase_wide/summarize.py
python research_walker_phase_wide/benchmark.py
python research_walker_phase_wide/audit.py
python research_walker_phase_wide/make_report.py
python research_walker_phase_wide/build_report.py
```

`reference.py` in imitation reproduces reference selection from the included old source trajectory. `prepare.py` in phase reconstructs phase-anchor weights; copied unchanged into wide. The broad width in `bandwidth.json` must be reconstructed from phase/control260/validation.csv when reproducing from scratch; do not retune using MG or held-out data. `adapt.py` documents initial source generation, not a mandatory reproduction step; actual used files are included. Original code is copied to avoid import ambiguity across separate physical tasks.

## Evidence and limitations

Final confirmation: 2/5 pairs meet the independent R/D/quality criterion; MG decreases in all five sensor/lag/window configurations only for seed271. Seed275 meets the independent criterion but left-knee MG increases. Main MG decreases in3/5 pairs. These results do not establish consistent detection across the series. Whole-cycle dispersion is an explicitly post-hoc diagnostic; see `CYCLE_DIAGNOSTIC_ADDENDUM.md`. It does not replace the original success criterion.

`PROTOCOL.md` in every folder records gates, seeds, budgets and settings. `selection.json`, all validation/test CSV, progress JSONL and raw NPZ remain. Exact MuJoCo state and external phase are restored in phase perturbations. Tests verify reset/reward-only intervention, periodic/perturbed section metrics, identical initial actors and frozen normalization. `audit.json` recomputes results from raw data.

`MG_windows.csv` retains primary and ALL secondary configurations with E40/floor diagnostics. `pair*.json` records eligibility, including insufficient paired coverage. Never replace missing data with zero complexity. All record acquisitions include falls; padded reward counts missing steps aszero. Surviving-only ratios are marked and cannot be generalized to failed episodes.

`probe_eps*.npz` contains68 signed physical perturbations and finite-difference17x17 one-cycle matrices (152 steps, continuous reference period~152.215steps). Matrix spectra are finite-horizon, state-dependent, exclude neutral external phase, and are NOT established Floquet/Lyapunov spectra. Epsilon .001/.0001 comparison reveals reliability limitations and must accompany any spectral interpretation. In the phase-free experiment the distance is phase-aligned; in phase-conditioned experiments phases are identical and no realignment is performed. Never compare their absolute amplification values across settings.

Serial `timings.csv` is measured after all jobs; MG E20 and E40, full-state and cheap scalar methods use same three2048 windows. Common acquisition is separate. No claim that MG is cheaper than all alternatives. Archive SHA256 and ZIP CRC are checked.

## Source motivation

DeepMimic: Peng et al., **DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills**, ACM Transactions on Graphics,2018. Author project: https://xbpeng.github.io/projects/DeepMimic/ . This study borrows only the broad idea of learning reference motion; it is not a DeepMimic implementation, benchmark or reproduction. Empirical results in this report are exclusively from the archived local runs.

SB3 PPO2.7.1, Gymnasium Walker2d-v5, MuJoCo3.14.0. PPO unchanged in these experiments; only environment reward and, for phase tasks, phase inputs/reset distribution differ. No claim of novelty for these RL modifications.
