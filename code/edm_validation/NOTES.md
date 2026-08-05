# Campaign log — where does EDM legitimately apply to training logs?

Running notebook of an overnight solo session (2026-08-06). Written for whoever picks
this up next, including me. Newest entries at the bottom.

## The question

The project's goal is EDM on memory-cheap 1D training logs, with theory grounding as a
hard requirement. The audit of `../grokking_analysis/` established that the published
"attractor dimension collapse" was a straight-line artifact, and `../prediction_improved/`
then showed a well-motivated function-space replacement is confounded with weight decay.
Both failures share one root cause, and it is a *theoretical* one:

> **Takens and Stark require the trajectory to lie on a compact invariant set that it
> revisits. A training run is a transient — it never returns near its past states.**

The theorems were cited but their preconditions were never tested. So the honest question
is not "which statistic predicts grokking" but:

> **On which training logs do the embedding theorems' preconditions actually hold, and
> what can be claimed there that survives a surrogate test?**

That question is answerable, has positive *and* negative controls available in this repo,
and yields a defensible result either way.

## Why `../poisoned_batch/` is the right testbed

It is a controlled experiment we already own: ResNet-18 / CIFAR-10 trained while an
*external driver* modulates the batch, logged as `poison_fraction` alongside
`train_loss`, `val_loss`, `val_accuracy` (7840 rows each).

| family | driver | what it is, dynamically |
| --- | --- | --- |
| `LogisticMap` | logistic map | deterministic chaos on a genuine 1-D attractor |
| `Sinusoidal` | sine | periodic, a 2-D limit cycle |
| `Discrete`, `StochSquare` | square wave | periodic, discontinuous |
| `ProgressiveNoise`, `Random` | stochastic | no low-dimensional structure |
| `Ghost_Const_*` | constant | no variation at all |
| `Ghost_Normal`, `Ghost_Uniform` | i.i.d. noise | negative control |

This matters for theory, not just for convenience. **Stark's theorem for forced systems
is exactly the setting where a driver has its own attractor.** The poisoned runs satisfy
that by construction; the grokking runs do not. So the testbed lets us ask whether EDM
recovers structure *when the theory says it should*, before asking anything about
grokking.

Ground truth + negative controls + a theorem that applies = the opposite of mumbo-jumbo.

## Plan

Ordered so each phase can fail cheaply and stop the next.

1. **Preconditions.** Recurrence quantification (recurrence rate, determinism) on both
   log families. Prediction: the driven runs recur, the grokking transients do not.
   This is the measurement that decides where Takens may be invoked at all.
2. **Surrogates.** IAAFT and phase-randomised nulls (Theiler 1992), validated on cases
   with known answers (Lorenz must reject, AR(1) must not). Mandatory for every later
   claim, per the agreed rigour bar.
3. **Forecastability.** Simplex projection with strictly causal library/prediction splits;
   skill vs. embedding dimension and horizon, always against surrogates. Positive control
   `LogisticMap`, negative control `Ghost_Normal`. Only then look at the grokking logs.
4. **Only if 1-3 hold up:** CCM driver recovery with surrogate-calibrated significance —
   `../poisoned_batch/ccm_pipeline.py` already does CCM, but adding a surrogate null is
   what would make its positives defensible.

Analysis runs locally (numpy/pandas/sklearn are installed); Colab is only needed for
training. That also sidesteps the VM reclaims that cost us two sweeps.

## Standing rules for this campaign

* No claim without a surrogate test. A statistic that fails to beat a phase-randomised
  surrogate is reporting linear autocorrelation, not dynamics.
* State the precondition before applying the theorem. If recurrence is absent, say so and
  do not invoke Takens.
* Every estimator gets a known-answer calibration before it touches project data.
* Negative results are results; write them down here with the numbers.

## Log

### 00:00 — starting state

Carried over from the evening:

* `prediction_improved/` — probe + Colab tooling, committed; function-space signal
  falsified (tracks weight decay, not generalization).
* Split x init sweep: 18/25 cells, gaps **1530–4630**, canonical seed-42 run **12 000**.
  Split explains 43%, init 17%, F(split)=4.31 on (2,8) — just under significance.
* GPU quota exhausted (`TooManyAssignmentsError`); CPU sessions still available.
* Running on Colab CPU: `split_seed=42 x 5 inits`, ~25 min/cell. First cell in:
  **split 42, init 0 -> gap 3420**, i.e. squarely in the same band as splits 0–3. So the
  article's split is *not* special on its own; the 12 000 needs the specific split x init
  combination. Awaiting the other four inits.

### 01:30 — machinery calibrated (`test_edm_validation.py`, 6/6)

Reference systems behave as they must before anything touches project data:

| check | result |
| --- | --- |
| FT surrogate preserves spectrum | rel. err 3e-16 |
| IAAFT preserves spectrum *and* value multiset | exact multiset, 2.6e-3 spectral err |
| simplex ranks chaos/periodic above white noise | 1.00 / 1.00 / -0.00 |
| skill decays for chaos, not for a cycle | chaos 1.00 -> -0.03 at h=32; sine flat 1.00 |
| recurrence separates attractor from transient | Lorenz ratio 0.93, monotone ramp **0.00** |
| surrogate test rejects chaos, spares AR(1) | p=0.025 vs p=0.475 |

Two of these failed first time round and both failures were mine, not the code's:

* The recurrence radius was recomputed *after* the Theiler exclusion, which forces the
  recurrence rate to equal the quantile for any input — it measured nothing. Fixed by
  fixing the radius once and sweeping the exclusion window instead (`recurrence_profile`).
* The horizon test asserted decay by h=8. With N=3000 on a 1-D map, error only reaches
  the attractor scale near `h ~ ln(N)/lambda ~ 15`, so 0.945 at h=8 was correct. The
  assertion was wrong, not the estimator.

### 02:10 — Phase 1: preconditions

`results/phase1_recurrence.csv`. Recurrence ratio = RR(longest exclusion)/RR(none).

* **Raw grokking logs do not recur at all**: `val_loss` ratio **0.000** for both S_5 runs,
  `weight_norm` 0.00–0.07, `train_loss` 0.02–0.87. Transients, exactly as the audit's
  line-constant artifacts implied. Takens/Stark have no invariant set to reconstruct here.
* Detrended poisoned and ghost logs sit at ~0.99.

**Important limitation, found by the controls:** the i.i.d. ghost drivers score ~1.0 too.
White noise has close pairs at arbitrary time separations, so recurrence separates
*smooth transients* from everything else — it cannot separate an attractor from noise.
Necessary, not sufficient. A determinism test has to follow it.

### 03:20 — Phase 2: univariate determinism. Four traps, then a negative result.

Simplex skill vs. IAAFT surrogates. Each trap below was caught by a control, which is the
entire argument for having them:

1. **Pearson skill is unsafe on these logs.** `val_loss` has kurtosis ~2000 and 50-sigma
   excursions; the correlation reports whether one spike was predicted and the surrogate
   ensemble scatters by +-0.34. Switched to Spearman.
2. **`argsort(argsort(x))` breaks ties in index order.** On `poison_fraction` (58–95
   distinct values in 7840 rows) that makes ranks correlate with *time*. Fixed with
   `scipy.stats.rankdata`.
3. **Moving-average detrending manufactures structure on sparse series.** The `Random`
   driver is **95.2 % exact zeros**; subtracting a moving average leaves
   -(smooth local mean) almost everywhere, which is highly rank-predictable. Raw skill
   -0.008, detrended Spearman 0.49, surrogate z **+27**. A pure artifact. First
   differences avoid it.
4. **And then the finding that sinks the approach.** With first differences on
   `train_loss`, the intended positives reject (LogisticMap +7.4, Sinusoidal +12.4,
   Discrete +12.6) — but so do the negatives: ProgressiveNoise **+8.3**, Ghost_Normal
   **+7.3**, and `Ghost_Const_0.1` **+8.1**, whose driver is a *constant*.

> **Negative result.** A univariate nonlinear-determinism test on the training loss is
> not a driver detector. The loss carries its own nonlinear structure — from SGD, the
> schedule, batch effects — so it rejects the linear-surrogate null whether or not
> anything is driving it. "The log contains nonlinear determinism" and "the log reveals
> an external driver" are different claims, and only the first is testable this way.

Also worth noting: on raw (undetrended) `train_loss` nothing rejects at all
(LogisticMap z = -0.43), because the training trend dominates the embedding geometry and
survives into the surrogates. So the univariate route needs a detrending choice, and every
detrending choice we tried either hides the signal or manufactures one.

**Consequence for the plan.** Driver detection is inherently a *two-series* question, and
the right tool is the one the repo already has: cross mapping. Moving to CCM, whose null
these controls can calibrate properly — that is Phase 4 brought forward, and the
univariate phase stands as a documented dead end with four reusable traps.

### 05:40 — Phase 3: CCM recovers a known driver from the loss log alone. 5/5, 0 FP.

`phase3_ccm.py`, results in `results/phase3_ccm.csv`. Embed `train_loss`, cross-map to
`poison_fraction`, test against two independent nulls.

| run | driver | raw rho | p (IAAFT) | p (shift) | verdict |
| --- | --- | --- | --- | --- | --- |
| Random | i.i.d. uniform, applied | **0.949** | 0.025 | 0.025 | detect ✓ |
| LogisticMap | chaotic, applied | **0.888** | 0.025 | 0.025 | detect ✓ |
| Sinusoidal | periodic, applied | **0.812** | 0.050 | 0.075 | detect ✓ |
| Discrete | square wave, applied | **0.802** | 0.025 | 0.025 | detect ✓ |
| ProgressiveNoise | stochastic, applied | **0.379** | 0.025 | 0.025 | detect ✓ |
| Ghost_Normal | logged, **never applied** | -0.011 | 0.750 | 0.850 | silent ✓ |
| Ghost_Uniform | logged, **never applied** | -0.007 | 0.675 | 0.750 | silent ✓ |
| Ghost_Const_0.1 | constant | n/a | — | — | silent ✓ |

**5/5 coupled runs detected; 0/4 false positives.**

Why the ghosts are the control that matters: they are logged exactly like real drivers and
the network trains exactly as usual, but `batch_poisoning.py` returns `images, labels`
untouched. So they share the training trend, the loss scale and the logging cadence with
the positives and differ *only* in whether coupling exists. Their rho of ~0.00 is what
rules out "the trend explains it" — the objection that sank the earlier weight-norm and
velocity results.

`Random` also gives the textbook convergence curve, rho rising 0.59 -> 0.95 with library
size, which is the signature CCM exists to read.

**Two ground-truth corrections along the way, both mine:**

* I first labelled `Random` and `ProgressiveNoise` as negatives because their drivers are
  stochastic. That confuses *determinism* with *coupling* — CCM tests the latter, and a
  random driver drives just as surely as a chaotic one. Reading `batch_poisoning.py`
  fixed the table: every real poisoner alters labels; only ghosts do not.
* Differencing the response, adopted in Phase 2 to defeat the training trend, is a
  **high-pass filter and erases slow drivers**. `Sinusoidal` and `Discrete` have the
  *strongest* direct coupling of any run (|corr| 0.69 and 0.58) yet cross-mapped at
  0.09 and 0.12 differenced — versus 0.81 and 0.80 raw. The trend needs no hand-removal:
  both nulls and the ghost runs already control for it. Kept in the output as a column,
  because the contrast is a result in its own right.

Also added `circular_shift` surrogates: for a periodic driver, IAAFT returns the same
waveform with a new phase, so it has little power; a rotation keeps every value and
changes only the alignment. It agrees with IAAFT everywhere here, and would matter for a
strongly periodic driver.

**Status against the project's stated goal.** This is a memory-cheap 1-D log, a theorem
that applies (Stark, forced systems — true by construction in these runs), ground truth,
negative controls, and a surrogate-calibrated null. It is the "does something good and is
theory-motivated" result the campaign was after, and it is *not* about grokking.
