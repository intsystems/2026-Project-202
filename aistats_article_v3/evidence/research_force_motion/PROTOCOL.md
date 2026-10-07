# Learned autonomous motion, 2026-09-29

Motivation: monitor a trained recurrent command generator from one neuron rather
than recording/differentiating its whole hidden state. This is an application-
motivated learning benchmark, NOT robot deployment or a measured human dataset.
The task is generation of a two-coordinate figure-eight reference trajectory.
Weights learn; evaluation then freezes them and studies the autonomous dynamics.
MG is NOT applied to the sequence of optimization losses across checkpoints.

Reference: Sussillo & Abbott (2009), Generating Coherent Patterns of Activity from
Chaotic Neural Networks, DOI 10.1016/j.neuron.2009.07.018 (FORCE / RLS feedback).

First pilot: seed 0, N=128, recurrent gain 1.5, Euler dt=.1, time constant=1,
period=10*pi time units. Dense J ~ Normal(0,1.5/sqrt(N)), feedback U ~ Uniform(-1,1)
with two outputs. W starts at zero. x_next=(1-dt)x+dt*(J+U W^T)tanh(x).
Targets are sin(2*pi*t/period) and .5*sin(4*pi*t/period).
P starts at identity; readout W is learned by RLS every second time step.
Target enters ONLY the RLS error, never the recurrent input. J and U stay fixed.
This low-rank feedback parameterization is fixed throughout; no rank is reduced.
Checkpoints: 0, 2000, 8000, 20000, 40000 training steps.
Model size/training duration may be revised after pilot based on task accuracy and
independent recurrent-state stability ONLY. All revisions and pilot failures stay.

After pilot, freeze configuration and run fresh seeds 1..5, no seed rejection.
Use checkpoint training state and discard 2000 autonomous evaluation steps.
Save a further 8192 consecutive steps, including all x for reference.
Observers are tanh(x[0]), tanh(x[1]), tanh(x[2]), set before results. No choice of
neuron based on MG/reference agreement. Primary is neuron 0.

Independent simplification: full-state lagged return residual around the known
target period, plus full Lyapunov spectrum of the actual Euler map. Propagate its
analytic Jacobian, QR every 10 steps. Report full spectrum, top exponent and
positive counts (finite-time thresholds disclosed, not exact active dimension).
Separate cheap single-vector top-Lyapunov baseline; also scalar std, spectral
entropy, autocorrelation and scalar return residual. Never claim MG beats all.
Task metric: raw autonomous error and phase-aligned shape NRMSE, optimal phase
selected using target correlation only, NOT MG. Frequency mismatch still incurs
error over the full evaluation length. Validate nominal target period separately.

MG primary: project pooled estimator, E20, tau4, k20, Theiler156, W4096, first and
last nonoverlapping half-record windows. E40 uses same lag/exclusion. Sensitivity
tau2/8 at W4096 (exclude=39*tau), W2048/8192 at tau4. Report every setting.
No empirical cutoff fitted to held-out seeds; compare paired before/after ratios
and independently identified recurrent stability. Display all failure cases.
Controls: (1) no learning; (2) learning readout with U=0, hidden state autonomous
and unchanged by learning; primary observer must be unchanged under matched
initial conditions; (3) multiplying the same observed record by .1 leaves MG
unchanged up to numerical error. Frozen-learning control doesn't freeze dynamics.

Timing: same machine, same trajectory lengths, one BLAS thread, sequential jobs;
exclude model training from diagnostic runtime; give MG-only, validity checks,
scalar recording, full-state recording, full-spectrum and cheap-baseline cost.
Primary performance conclusion requires a practical signal as well as speed.

## Pilot decisions, frozen before held-out seeds and BEFORE MG computation
1. N128/g1.5/seed0: task learned, but untrained reservoir decayed toward a fixed
   point (top finite-time exponent -0.024); unsuitable for a chaos-to-cycle test.
2. N256/g1.8/seed0: task not learned reliably after 40,000 steps (aligned NRMSE
   about .91); chaos remains. Both negative pilots retained in their directories.
3. N256/g1.5/seed0: initially top exponent +.042, after 8,000 steps aligned task
   NRMSE .018 and small full-state return error. Select this configuration based
   on task/independent reference only. Pilot files: pilot256_g15.
4. Freeze N=256, gain=1.5, all other parameters unchanged; new seeds 1,2,3,4,5.
   Training checkpoints 0/2000/8000/20000/40000, evaluation 8192 after 2000 burn.
   For full Lyapunov primary validation use 0/8000/40000 (all 256 exponents).
   No MG-based hyperparameter revision, run or observer selection is permitted.
5. Independent endpoint thresholds declared here: phase-aligned task NRMSE<.1,
   full-state target-period return residual<.1, top finite-time exponent magnitude
   <.005 and second exponent<-.005. Report continuous values and failures too.
   A stable non-target fixed point cannot pass the task criterion. Perturbed
   initial states also tested after training, without additional learning.

## Additional chaos-conditioned cohort, declared before any MG results
The five unrestricted runs all learned, but initial full spectra show that seeds
1/2 already have periodic dynamics and seed4 approaches a fixed point. Keep and
report all five; they cannot all support chaos-to-cycle simplification.
For a separate conditional test, screen seeds6..30 in ascending order, N256g1.5,
using a 16384-step untrained autonomous record after 2000 burn. Accept the first
five for which both the whole-record and second-half single-vector Lyapunov
estimates exceed .01 (each has 1000 tangent burn). Log every screened seed.
This is explicitly a chaos-conditioned cohort, NOT an unconditional population
success rate. MG is not evaluated during selection. No post-training exclusions.
Train/evaluate all accepted seeds using the same frozen training and MG settings.
Full spectra on the primary 8192-step record will independently audit selection.
Report all failures, baseline-versus-final changes and all three preset observers.
