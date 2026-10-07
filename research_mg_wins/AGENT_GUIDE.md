# Guide for experiment agents (MG campaign, 7 Oct 2026)

## The method
MG = pooled MacKay–Ghahramani / Levina–Bickel intrinsic-dimension estimate of the delay
embedding of ONE scalar time series (a training log: parameter norm, mini-batch loss,
gradient norm, a neuron, an action...). Applied claim of the paper: MG is a cheap monitor
of *simplification* of learning dynamics (fewer active directions / components), computed
from a scalar log only, without weights/activations.

```python
import sys; sys.path.insert(0, r"C:\Users\karlo\notebooks\grokking_prediction_original\2026-Project-202\code")
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")  # training-log default
mg = estimate(window_1d_numpy, CFG).MG          # window: e.g. 1000 samples; ~10-100 ms
# variant that was most specific on CNN logs: EstimatorConfig(max_E=20, tau=4, k_neighbors=50, theiler="embedding")
```
Online change detector used before (research_trajectory_reference/detector.py):
windows W=1000 stride 500; D_k = median(last M windows)/median(previous B windows) - 1;
alarm when sign*D_k < -delta; (M,B,sign,delta) calibrated on calibration runs only.
`detector.drop_series(g, stat, M, B, sign)`, `detector.evaluate(...)`, `detector.calibrate(...)`.

## Mandatory competitors
Scalar competitors on the SAME log and windows: `research_mg_wins/baselines.py`
(`self_repeat` = best normalised self-repeat error over lags 20-250 — the strongest
competitor so far; `spectral_entropy`, `roughness`, `perm_entropy`, `recurrence_rate`,
`corr_dim`, `twonn_fit`, `linear_pr`) plus `cifar_events.simple` (crossings, lag1, det_std).
Use `functools.partial(..., tau=1)` for delay-space ones on training logs (see
`research_mg_wins/cnn_competitors.py`). PLUS the domain-standard method from the literature
for your setting (it may use internals: activations, gradients, embeddings) and the trivial
practitioner rules (fixed schedule, plateau rules, validation-based oracle as upper bound).

## What has already been tried (do not repeat)
Negative or tied for MG: training-event detection on CNN/ResNet param-norm logs
(self_repeat equal or better on fresh seeds), learning-rate choice from short probes
(probe loss better), early stopping under label noise (fixed step better), ESN simulator
selection (spectral Hellinger distance better), edge-of-stability counting, dead-ReLU
detection, SGD mini-batch loss logs (MG ~ spectral), HAR/UCR classification, real-series
forecasting, FORCE certification, Walker policy selection (spectral entropy better).
Positive: phases vs harmonics in trained generators (geometric statistics win), robustness
to unit changes / level jumps in logs, 14-23x cheaper than Lyapunov spectra.
Known MG weaknesses: additive measurement noise, abrupt events, mini-batch noise dominated
logs. Known MG strengths: gradual sustained simplification, scale/level invariance,
counting independent oscillatory components.

## Machine and pitfalls (Windows 10, CPU only, 16 cores, ~9 GB free RAM SHARED by 4 agents)
- Your budget: at most 3 concurrent python worker processes, 1 thread each; peak RAM < 2 GB.
- Put `os.environ[v] = "1"` for OMP/OPENBLAS/MKL_NUM_THREADS at the very top of every script
  BEFORE importing numpy/torch (Windows multiprocessing spawns fresh processes);
  `torch.set_num_threads(1)` in workers.
- CIFAR subset (10k train, 2k test, float32 tensors): `research_mg_wins/cifar_cache.py`
  `load()` -> (X, y, Xprobe, yprobe, Xtest, ytest). Never load full CIFAR in many processes.
- MNIST raw files: `research_gan_collapse/data/MNIST/raw`. sklearn `load_digits` available.
  No internet downloads of datasets unless tiny; torch 2.10 CPU, sklearn 1.8, scipy.
- Bash tool is Git Bash. Long jobs: `(nohup python -u script.py > log.txt 2>&1 &)` — the
  parentheses matter, otherwise the job dies with the shell. Or run in foreground with a
  timeout <= 10 min per call and poll. Save results incrementally to disk.
- Write ONLY inside your own folder `research_mg_wins/<your_folder>/`. Do not edit shared
  files (baselines.py, detector.py, estimator code); import them.

## Scientific rules (non-negotiable)
1. Pilot on a separate seed to set up the task (look only at the ground truth / training
   behaviour, never at MG vs competitors).
2. Write the PROTOCOL (task, truth, all rules, calibration and test seeds, metrics,
   predictions) into the docstring of the main script BEFORE running it.
3. Calibrate every rule (MG and competitors alike, same grids) on calibration runs; score on
   held-out test runs/seeds. Report all competitors, all failures, wall-clock costs.
4. Never tune MG on test data. If you look at test results and then change something, the
   change needs a fresh confirmation on new seeds.
5. Report faithfully, including negatives. A tie is a tie.

## Output
`research_mg_wins/<your_folder>/REPORT_ru.md` (Russian, concise; tables; figure labels in
English) and a final message: setting, why it matters (cite papers), protocol, table of MG vs
every competitor on test, verdict (MG better than all / some / none), cost, files.
