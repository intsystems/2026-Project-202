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

### 06:30 — Phase 4: specificity check on our own runs

`phase4_grokking_logs.py`. Cross mapping *between metrics of the same run* (no injected
driver anywhere). **11 of 16 directed pairs fire**, including 4/4 on `mod_wd0` and 4/4 on
`lowdata15`, with rho up to 1.000.

This is not a contradiction of Phase 3 and not a bug: `train_loss`, `val_loss` and
`weight_norm` are three observations of one system, so under Takens each should predict
the others. But it does mean **CCM between a run's own metrics is not discriminative** —
it fires nearly everywhere, so no conclusion about grokking can rest on it. Recorded so
nobody later mistakes a rho of 0.99 between two of our own columns for a discovery.

The shift null earns its place here: on `mod_wd1`, `weight_norm xmap train_loss` gives
p=0.025 under IAAFT but **p=0.850** under a circular shift. IAAFT alone would have
over-reported. Requiring both is the right default.

### 07:00 — Phase 5 (running): does Phase 3 transfer to our own architecture?

Phase 3 used ResNet-18 / CIFAR-10 logs from someone else's script. `inject.py` +
`phase5_inject_run.py` repeat the design here: a 1-layer transformer on modular addition,
with `logistic` / `sinusoid` / `iid` drivers that corrupt a driven fraction of each
batch's labels, and a `ghost` that is logged identically but returns the labels untouched.
`grok.loop.train` gained an optional `batch_hook` for this (default `None`, so every
existing run is unaffected).

Smoke test confirms the control is a true no-op: over 400 steps the ghost run ends at
`||w||_2 = 43.98`, exactly the unpoisoned baseline, while the logistic run is pulled to
38.71 — same seed, same everything else.

Analysis is `phase5_inject_ccm.py`, pending the runs.

### 07:10 — split 42 is not special (sweep, resumed on GPU)

The article's own split, run against fresh initialisations: gap **3420** (init 0) and
**2050** (init 1), both squarely inside the 1530–4630 band from splits 0–3. So the
canonical 12 000-step delay is not a property of that split; it needs the particular
split *and* the particular initialisation, and it sits far outside the distribution of
everything else measured. The remaining cells are running.

**Operational note, again.** The CPU session running the earlier split-42 attempt was
reclaimed and its results were lost, because that job predated the incremental-download
watcher. Everything since writes `summary.csv` back to the repo every four minutes.

### 08:30 — Phase 5 result: transfers to our architecture, with a real limitation

`results/phase5_inject_ccm.csv`. Same design, 1-layer transformer on modular addition:

| driver | applied? | rho | p (IAAFT) | p (shift) | verdict |
| --- | --- | --- | --- | --- | --- |
| logistic | 0.293 | **0.934** | 0.025 | 0.025 | detect ✓ |
| iid | 0.252 | **0.919** | 0.025 | 0.025 | detect ✓ |
| ghost | **0.000** | -0.024 | 0.750 | 0.825 | silent ✓ |
| sinusoid | 0.250 | 0.882 | 0.775 | 0.775 | **missed** |

Across both settings: **7 of 8 coupled runs detected, 0 of 5 ghosts fired.**

**The sinusoid miss is a genuine limitation, and it is worth stating precisely.** Its rho
is high (0.882) — but so is the surrogates' (IAAFT 0.87–0.90, shift 0.87–0.94, measured
directly). The reason is structural: the loss reveals *where in the cycle* the run is, and
any same-period series — an IAAFT surrogate, or the driver rotated in time — is a
deterministic function of phase, hence equally predictable from that information.

> **Spectrum-preserving nulls have no power against a strictly periodic driver.**

This is not a bug and not fixable by a better statistic. It also explains why the
poisoned_batch `Sinusoidal` run *did* reject at p=0.050: that driver is quantized to 95
distinct values, so its surrogates depart from it much more than a smooth sinusoid's do
(1044 distinct values here). A periodic driver needs a different null — comparing against
a driver of a *different period* would be the natural one, and is untried.

### Final state of the split x init sweep (30/30 cells)

```
gap: min 1290  median 2875  max 5600  sd 1130
variance explained by split 19.8%,  by init 23.8%,  residual/interaction ~56%
mean gap by split: {0: 3840, 1: 2430, 2: 2540, 3: 3396, 4: 3276, 42: 2888}
mean gap by init : {0: 2953, 1: 2323, 2: 3397, 3: 3895, 4: 2740}
```

**The partial result was misleading and the full one overturns it.** At 15 cells the split
looked dominant (43% vs 17%, F just under significance). With all 30, the two axes are
comparable and *init is marginally the larger* — neither dominates, and more than half the
variance is interaction. So the delay is **not** predictable from the split alone, and the
pre-training-predictor idea in `../prediction_improved/method.md` §10 loses its motivation
unless someone can model the interaction. Noted as a caution about reading partial sweeps:
I had already written the split-dominant reading into the log before the run finished.

What does survive, and is now much stronger: **all 30 cells generalize, and every gap lies
in 1290–5600, against the canonical run's 12 000.** Thirty independent (split, init)
combinations of the article's own configuration, and not one reproduces its headline
delay — including five that use its own split.

### 09:40 — Phases 6-7: a third configuration-tracker, caught in 40 minutes

**The idea.** Everything that failed in this project failed as a statistic on a *single*
transient series. Cross mapping asks a two-series question — do these observables share a
manifold? — which is well-posed on a transient and is what Takens actually licenses. So:
do `train_loss` and `weight_norm`, **both training-side, no validation data**, share a
manifold, and does it change at grokking? That is the article's ambition with a validated
tool and the property the article claimed but never had.

**Phase 6 looked like a hit.** Median coupling z over 7 runs, no overlap between families:

| delayed transition | z | no delayed transition | z |
| --- | --- | --- | --- |
| grok | 1.15 | nogap | 2.87 |
| grok_seed1 | 1.05 | lowdata20 | 3.00 |
| grok_seed2 | 1.00 | lowdata15 | 3.40 |
| | | wd0 | 5.61 |

Two reasons not to believe it, both fatal:

* **No within-run signal.** If it tracked the transition it should move at `t_gen`. In
  `grok` the series runs `+2.7 +3.4 +1.5 -1.4 … +1.3 | +2.3 +1.8 +1.8` — the post-`t_gen`
  values are unremarkable against early windows reaching +3.4 — and `grok_seed1` shows
  nothing. The "before 0.31 -> after 1.56" summary I first wrote was **averaging across
  runs with different `t_gen`**, which manufactures a step present in no individual run.
* **The between-run split coincides with configuration**: low = {fraction 0.3, wd 1},
  high = {fraction != 0.3} ∪ {wd 0}. Precisely the confound that killed the weight-norm
  and the velocity signals.

**Phase 7 settles it.** Eight runs of *one* configuration (`mod_wd1`), varying only the
split and init seeds, which the sweep showed give gaps of 1290-5600. Configuration is
fixed by construction, so a coupling-vs-gap correlation could not be explained by data
quantity or regularization. The reading was pre-registered in the docstring before running.

```
corr(gap, z_median)  : pearson -0.090  spearman +0.214   (n=8)
corr(gap, z_plateau) : pearson +0.431  spearman +0.643   (n=7)
```

> **Negative.** The prediction was a strong *negative* correlation. What came out is
> nothing, and — for the plateau windows — a weak *positive* trend, the wrong sign, and
> not significant at n=7 (Spearman 0.64, p~0.12). Within one configuration the coupling
> spans only 0.80-2.55 against the 1.0-5.6 spread across configurations. **The Phase 6
> separation was configuration, not transition.** Third instance of the same confound.

What is worth keeping is the *procedure*: hold the configuration fixed, vary only seeds,
pre-register the sign. That test costs about forty minutes and would have killed both
earlier signals before they were written up. It should gate every future candidate.

## Where this leaves the project

* **A defensible EDM result exists**, and it is about *externally driven* training rather
  than grokking: cross mapping recovers a known driver from a 1-D loss log, validated in
  two independent settings, with ghost controls, two nulls, and a stated limitation.
* **The grokking claims do not survive** — not the dimension collapse (audit), not the
  function-space velocity (weight-decay confound), and the delay itself is a rare draw.
* **Next, in order of value:** (1) a null with power against periodic drivers; (2) whether
  cross-map rho tracks driver amplitude, i.e. a sensitivity curve — `ProgressiveNoise` at
  rho 0.38 versus `Random` at 0.95 suggests it does, and that would turn a binary detector
  into a measurement; (3) S_5 / other tasks, to check the two settings were not both easy.

## Phase 9: the falsification does not depend on the Theiler window

Alex raised the sharpest available objection to section 3.1: the Theiler exclusion is
doing the work, and the original analysis, which used no exclusion, "worked rather well."
The objection deserved to be answered under the paper's own settings rather than argued
about, so `phase9_paper_settings.py` repeats everything at W=0, k=5, max_E=15, window 300.

**The level argument was already sufficient but incomplete.** Proposition 1 at W=0 returns
1.330 and the WD=0 controls sit there permanently, so turning the exclusion off does not
restore information. But a control whose norm never leaves a straight line is flat by
construction, and a flat control cannot refute a claim about a *shape*. That gap was real
and I had not noticed it. The control that closes it is `lowdata15`/`lowdata20`: weight
decay identical at 1.0, training fraction reduced, never generalises.

**The signal fails an internal check before any control is applied.** Three seeds of the
identical grokking configuration, weight norms ending at 29.89 / 29.12 / 30.39 (agreeing to
4%), give dimension trends of -0.54, **+0.94, +0.90**. Two of the three *rise*. The
published descent is not reproducible across seeds of the condition it was reported in.
Late-training levels are 1.51 / 9.21 / 11.64, a factor of 7.7 within one configuration.

The matched controls then give trends +0.75 and +0.22, inside the generalising range. And
the estimate correlates with `std(diff(x))/std(x)` at Spearman +0.887 pooled over 1026
windows (+0.32 to +0.996 within runs) — a one-line ratio with no embedding, no neighbour
search, no likelihood. The mapping is not run-invariant, so smoothness is a strong
correlate, not a complete description; do not overstate this one.

Note the sign is immaterial to the mechanism, which is a point in its favour: the grokking
norm falls 42 -> 30 and the lowdata norm rises 42 -> 61, and both collapse the estimate,
because a monotone rise is as locally straight as a monotone fall.

**Two of my own metrics were wrong on the first pass and are worth remembering.** I first
reported "collapse ratio" = peak/tail, which for a monotonically *rising* trace measures
noise around the trend and silently reported 1.16-1.18 for the two runs that most sharply
contradict the claim. Replaced with a Spearman trend against step plus early/late medians.
Second, `wd0` shows trend -1.00, which looks decisive and means nothing: the series runs
from 1.34 to 1.33, a rank-monotone drift of 0.7%. Rank statistics need a magnitude beside
them.

## Phase 10: our own identifiability diagnostic was confounded by series length

Auditing phase 8 rather than trusting it. The report argued that the weight-norm estimate
is non-identifiable because it grows with E_max (ratio 3.73 / 4.36) while a Lorenz
reference stays flat (1.66). **That comparison was invalid.** The Lorenz series was 12000
samples and the training logs are 2000-3000, and the diagnostic is strongly
length-dependent:

| n | Lorenz-63 | white noise |
|---|---|---|
| 1000 | 3.87 | 4.45 |
| **2000** | **8.86** | 4.39 |
| **3000** | **8.42** | 4.40 |
| 6000 | 2.21 | 4.53 |
| 12000 | 1.66 | 4.66 |
| 24000 | 1.32 | 4.83 |

At the length the logs actually have, a textbook attractor scores 8.86 — **worse than
white noise (4.39) and worse than the logs themselves (3.73, 4.36)**. The diagnostic has
no power below n ~ 6000, and the "behaves like noise" sentence in the report was produced
entirely by the sample-size mismatch. This is precisely the error I flagged in the
published work: an apparent separation carried by an uncontrolled difference.

Retracted and replaced. The honest claim is about the **data budget**: at n <= 3000 the
estimator cannot resolve a two-dimensional attractor, so no dimension is identifiable from
a log of that length whether or not one exists. That is weaker but it is true, and it is
now a numbered precondition in the protocol section (a calibration reference must match
the length of the series under test).

**Nothing load-bearing was lost.** The falsification of the dimension signal rests on
Proposition 1 (closed form, verified to 1.1%) and on phase 9 (two of three seeds of the
identical configuration rise rather than fall). Neither uses this diagnostic. Table 1's
"tracks E_max as white noise does" cell was rewritten to cite the seed result instead.

Lesson for the remaining claims: **every calibration reference must be length-matched to
the series it calibrates.** Check the CCM convergence figure for the same defect.

## Phase 11: multiplicity in the CCM detection rule (checked, conclusion holds)

The load-bearing positive result deserved the same scrutiny as the negatives. Two defects
in the detection rule, both real:

1. `detected = iaaft.detected or shift.detected` is a family of two tests at 0.05 each,
   combined by OR. Family-wise false-positive rate approaches 0.10, never stated.
2. With n_surrogates=39 the rank p-value floor is (0+1)/(39+1) = **0.025**, which is
   exactly the Bonferroni threshold for two tests. The corrected test could only ever
   reject by the narrowest possible margin, and phase3 Sinusoidal sat at p_iaaft=0.050,
   i.e. on the wrong side of it. So the published 7/8 was one knife-edge away from 6/8.

Re-ran both experiments at 199 surrogates (floor 0.005), ~10 min total. **Conclusion is
unchanged and now robust:** TP=7 FN=1 FP=0 TN=4 under both the original OR-at-0.05 rule
and Bonferroni at 0.025. Sinusoidal moves 0.050 -> 0.0150 and survives the correction
comfortably. Ghost p-values 0.645-0.850, nowhere near rejection.

So this one came out the right way, unlike phase 10. Worth noting the asymmetry: the
identifiability diagnostic collapsed under audit and the CCM result did not. That is the
difference between a statistic validated on a system with a known answer and one applied
straight to project data.

Remaining: n_surrogates=39 is still the default in ccm.py and the phase3/phase5 scripts.
The results are unchanged so nothing needs recomputing, but 199 should be the default for
anything new, since 39 makes any multiple-comparison correction inexpressible.

## Phase 12: direction of the coupling, which we had never tested

Alex: "the recovered quantity is causal rather than correlational - you need to check that
the algorithm shows A->B rather than B->A". Correct, and we had not. Sections 4 asserted a
causal recovery while only ever computing one direction.

**Convention first, calibrated on a known answer** (the phase 10 lesson). Coupled logistic
maps, Sugihara 2012 eq. 1, b_xy=0 and b_yx=0.32 so X drives Y alone:

    Y xmap X (true) : 0.331 -> 0.993   converges
    X xmap Y (false): 0.585 -> 0.569   flat

So ccm.py's convention is right as documented. And the false direction still reaches 0.57:
**convergence is the discriminator, not skill.** Now a permanent test, suite is 7/7.

**On the logs, at matched E=3:** forward converges more than reverse in 7/8 coupled runs.
Ghosts converge in neither direction, |gain| <= 0.039 -- the test does not manufacture
direction. The one exception is the transformer sinusoid: reverse gain 0.170 vs forward
0.040, and its forward detection also fails at p=0.77. Same periodic case as phase 3.

**But the strict test fails, and this must be said plainly.** Granting the reverse the best
of five embedding dimensions, it converges and beats its own null in most runs: 0/8 coupled
runs read unidirectional, 5/8 read bidirectional, 2/8 read reverse. The E-scan is a
selection over five comparisons and inflates the reverse, but that is not the whole effect.

The reason is structural, not an estimator defect: the driver is applied every step, so the
loss depends on it contemporaneously, and a delay vector of the driver already contains the
value the loss is a function of. The reverse map succeeds by direct functional dependence
with no manifold involved. CCM does not separate causation from strong instantaneous
coupling (generalised synchrony, Rulkov 1995; Ye 2015).

**What the report may now claim:** coupling is detected; the dominant direction is right in
7/8; unidirectionality is NOT established. Softened the abstract and conclusion accordingly
and added a fifth protocol requirement (run both directions, matched E, calibrate the
convention, read direction from convergence not skill). To establish unidirectionality one
would need the driver to act with a delay so the contemporaneous term is absent.

**Cost note.** First version ran a full 199-surrogate ensemble at every E of the scan:
~15 min per run, ~2 h total. Killed it and measured instead of guessing. IAAFT generation
on 7840 samples dominates, not the cross-mapping. Scanning E on convergence curves alone
and running the ensemble only at the selected E cut it to ~20 min with no loss of rigour.

## Colab: a harness bug that cost three runs

colab_job.ps1 -Run executes training in the *foreground* of a colab exec, holding the
session for the whole job, so every later exec queues behind it and results cannot be
copied off while training runs. Free-tier VMs get reclaimed without warning; three velocity
replicates died that way, twice with the training already finished. Fixed with
remote/launch_detached.py (start_new_session, returns at once) plus remote/list_outputs.py
(cheap poll: liveness, files, log tail). Verified: poll 1 returned while training was live.

Also: Git Bash rewrites an absolute remote path such as /content/job_cmd.txt into
C:/Program Files/Git/content/... and the upload 500s. Use PowerShell or MSYS_NO_PATHCONV=1.

Velocity replicates as of now: lowdata15 at two seeds (5.30e-2, 5.49e-2, agreeing to 3%),
lowdata20 at one, three grokking seeds (1.94/1.84/1.38e-2). No overlap, factor >= 2.7.
That is enough for the claim; the remaining two seeds are a nicety, not a dependency.

### Velocity replicates: final state, and the harness fix paying for itself

Fourth VM reclamation, but this time it cost nothing that mattered. The detached launcher
plus guarded poller had already pulled `lowdata15_s2` down before the session went away;
under the old foreground design it would have died with the VM like the three before it.

Final comparison, median normalised-logit velocity over the second half of training:

    lowdata20      7.256e-02      never generalises, WD=1
    lowdata15_s1   5.486e-02      never generalises, WD=1
    lowdata15_s2   5.416e-02      never generalises, WD=1
    lowdata15      5.304e-02      never generalises, WD=1
    grok_seed2     1.940e-02      generalises
    grok           1.840e-02      generalises
    grok_seed1     1.381e-02      generalises
    wd0            1.531e-04      never generalises, WD=0

Four never-generalising runs at WD=1 against three grokking runs, no overlap, closest
members separated by 2.73x. The reduced-data condition is replicated across three seeds
which agree to 3.4%. `lowdata20_s1/s2` were not recovered and are not needed: the claim is
that a statistic proposed as a generalisation precursor sits *higher* in runs that never
generalise, and four independent never-generalising runs establish it.

Stopping here rather than spending a fifth VM. The remaining two seeds would thicken an
already non-overlapping comparison, not change it.

**Poller bug worth remembering.** The first guarded-less poller hung for over an hour on a
single `colab exec` that never returned, while the session itself answered a manual exec
immediately. Every CLI call in a long-running poller needs a hard timeout (Start-Job +
Wait-Job -Timeout), not just the CLI's own --timeout flag.
