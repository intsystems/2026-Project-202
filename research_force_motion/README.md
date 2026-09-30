# Learned autonomous motion: FORCE and scalar MG

Start with **report_ru.pdf** (Russian, five pages). Numerical summary:
`results_chaotic/report_summary.json`. Experiment date: 2026-09-29.

This is an application-motivated **synthetic recurrent signal-generation task**,
not robot deployment, CV classification or language-model training. A 256-neuron
RNN learns a two-coordinate figure-eight using FORCE/RLS output feedback. MG is
computed on autonomous, frozen-weight rollouts, NOT on optimizer training loss.

## Main findings and limits

- Conditional on independently detected initial chaos, all five runs learned the
  target and primary MG dropped 68--78%. All 15 fixed-observer pairs and all 20
  additional window/lag pairs retained the direction. These are repeated measures
  on FIVE models, not 35 independent training runs.
- Full finite-time Lyapunov spectra and full-state return independently support
  near-periodic dynamics. The predeclared strict endpoint test passed 4/5: seed 9
  has a second exponent -0.00222, weaker than the -0.005 cutoff. No post-training
  exclusions. Numerical finite-time evidence is not an asymptotic proof.
- Initial-chaos screening: first five qualifying seeds in 6..30; 13 candidates
  checked; accepted 7/9/15/16/18. Selection did not use MG or training success.
  Original unfiltered seeds 1..5 and unsuccessful pilot settings are retained.
- Full-spectrum comparison: MG-only 14.4x/19.3x faster before/after training,
  including E40 repetition 5.4x/9.5x faster. One top exponent and simple scalar
  diagnostics are FASTER than MG and also detect the transition.
- Low MG is not proof of correct output. Perturbations include wrong, simple
  cycles; task NRMSE passes 13/15 and joint task/return thresholds 12/15.
- Scalar log 64 KiB vs full trajectory 16 MiB for 8192 float64 samples. This is
  stored-series volume, NOT peak estimator memory. Lyapunov analysis can also be
  streamed instead of storing the trajectory, but needs the state/model/Jacobian
  and tangent vectors. No GPU or large-model scaling claim was tested.

## Reproduce

Run from the **2026-Project-202 root**, or from the root of the compact archive
(it includes `code/actdim`). Dependencies and observed versions are recorded in
`research_force_motion/environment.json`; install the scientific Python packages
listed there in a suitable environment. No torch/GPU needed. A typical run uses
Python 3.13, NumPy, SciPy, pandas, scikit-learn, matplotlib and threadpoolctl.
Scripts set one BLAS thread. Run diagnostic jobs sequentially for timing.

```powershell
# Initial unfiltered test. Use a new output directory if preserving this run.
python research_force_motion/run.py --n 256 --gain 1.5 --seeds 1 2 3 4 5 --out research_force_motion/results
python research_force_motion/analyze.py --phase all --root research_force_motion/results

# Independent chaos screening + train all accepted seeds. Does not compute MG.
python research_force_motion/screen.py
python research_force_motion/analyze.py --phase all --root research_force_motion/results_chaotic

# Warmed, repeated, matched-length diagnostic timings.
python research_force_motion/benchmark.py
python research_force_motion/make_report.py
python research_force_motion/build_report.py
python research_force_motion/package_results.py
```

`analyze.py` caches full spectra in each seed directory. For changed simulation
parameters, use a new results directory to avoid reusing spectra from older
checkpoints. `build_report.py` needs XeLaTeX and Times New Roman. It fails on
LaTeX errors, missing glyphs and overfull boxes. Markdown is also provided.

## Data and protocol

`PROTOCOL.md` preserves pilot selection and the additional cohort decision before
MG computation. No estimator or observer was tuned after inspecting MG.

- `pilot/`: N128/g1.5/seed0, learned but initially nonchaotic.
- `pilot256/`: N256/g1.8/seed0, learning failed.
- `pilot256_g15/`: N256/g1.5/seed0, successful task pilot, selected before MG.
- `results/`: all seeds1..5, including initially periodic/decaying cases.
- `results_chaotic/screening.csv`: all 13 screening candidates; 16384 steps with
  top-exponent checks on full record and second half, threshold .01.
- `results_chaotic/independent_reference.csv`: 256-exponent summaries, full-state
  return, task errors and predeclared pass/fail for 0/8000/40000 training steps.
- `mg.csv`: all raw observers/windows at five checkpoints, E20/E40 estimates,
  degeneracy, cheap controls and timings. `mg_summary.csv`: median of two windows.
- `sensitivity.csv`: ALL window2048/8192 and lag2/8 endpoint comparisons.
- `perturbations.csv`, `scale_control.csv`, `control_audit.json`: controls.
- `benchmark.csv`: each of three matched 8192-sample timing repetitions; use the
  median within each training stage. Variability is not suppressed.
- `seed_*/checkpoint_*.npz`, `rollout_*.npz`: weights and complete raw states.
- `seed_*/lyapunov_*.npz`: all 256 exponents plus first-half estimates and time.

The compact archive includes scripts, the estimator source, report, all CSV/JSON
tables, plots and all Lyapunov spectra, plus checkpoint and scalar-observer data
for seed7 at each stage. Large full-state rollouts and other checkpoints stay in
the workspace and can be regenerated. No article source/PDF was modified.

## Scientific source

Sussillo & Abbott (2009), *Generating Coherent Patterns of Activity from Chaotic
Neural Networks*, Neuron 63:544--557, DOI 10.1016/j.neuron.2009.07.018. Our particular
figure-eight task, screening, scalar diagnostics and numerical results are this
experiment's implementation; do not attribute these measurements to that paper.
