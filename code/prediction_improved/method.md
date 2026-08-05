# Early warning of grokking from function-space trajectories

**TL;DR:** A sustained, causally-labelled rise in the non-straightness of normalized
probe-logit trajectories, after memorization, is proposed as an early warning of
grokking that — unlike weight-norm signals — cannot be explained by weight decay alone.

**Abstract.** Grokking — generalization emerging long after training accuracy
saturates — appears unpredictable from standard training metrics. Prior work, including
ours, reported that an intrinsic-dimension estimate of delay-embedded training logs
collapses before generalization; a code audit showed this statistic in fact tracks the
local smoothness of the weight norm, is confounded with weight decay, and is beaten by
a trivial null model. We propose a minimal, falsifiable early-warning method built on
that diagnosis. The observable is the trajectory of centered, L2-normalized logits on a
fixed training-distribution probe set, projected onto fixed random directions —
invariant to the uniform scale mode that weight decay drives, and free of validation
data. The statistic is departure from local linearity (residual roughness, or the
participation ratio of a short delay-embedded segment), which has exact reference
values on a straight trajectory and therefore needs no per-run calibration. The
detector is a single causal rule: warn when the statistic rises above its floor,
sustained, after memorization. Specificity is established against two null models
(logit velocity; weight-norm roughness) and new controls, chiefly weight decay without
grokking. Every experimental outcome, including full failure, has a defined reading
about where grokking's hidden reorganization lives — in weight space or in function
space.

## 1. Problem

Grokking: training accuracy saturates at step $T_{mem}$, validation accuracy stays at
chance through a long plateau, then jumps at $T_{gen} \gg T_{mem}$. We want a **causal
early-warning signal** from training-side observations only — fires in
$(T_{mem}, T_{gen})$ on runs that will grok, silent on runs that memorize forever.

## 2. Diagnosis of the previous pipeline

The article estimated an intrinsic dimension of the delay-embedded weight-norm log and
read its collapse as an attractor contracting before grokking. The audit
([`../grokking_analysis/README.md`](../grokking_analysis/README.md), machine-checked)
found:

1. **Not a dimension.** At feasible window lengths the estimator returns closed-form
   constants of a straight line; both published plateaus are reproduced by a line with
   no dynamics, and no absolute dimension is identifiable. What was tracked is the
   local smoothness of $\|w\|_2$.
2. **Confounded.** Controls differ from treatments only in weight decay, and $\|w\|_2$
   is the quantity WD acts on directly — "detects grokking" and "detects weight decay"
   are indistinguishable in the existing data.
3. **Beaten by a null.** The residual of a linear fit (three lines of code) detects the
   same transition earlier, with the same silent controls.

Honest current state: *the weight-norm trajectory departs from local straightness well
before grokking, and never does so without weight decay.* Real signal; specificity
unproven; "dimension" framing an artifact.

## 3. Method

**Observable.** Fix, before training: a probe set of training-distribution inputs (no
labels, no validation data) and a few random unit vectors $v_r$. At each log step, in
`eval` mode: logits on the probe set; center each example's logit vector over classes
and L2-normalize it; record $x_r(t) = \langle \mathrm{vec}(\widetilde Z_t), v_r\rangle$.
Each $x_r$ traces the model's *function* up to per-example scale — the mode weight decay
drives is projected out. Log every step (logging cadence, not the estimator, sets the
detection floor).

**Statistic.** Over short causal segments of each series: **roughness** = std of the
residual after a linear fit / std of the segment; alongside it the **participation
ratio** of the segment's delay-embedded singular spectrum. Exact floors — 0 and 1 on a
straight trajectory — so no baseline collection or per-run calibration. Defined on tens
of samples; no neighbour search, no Theiler window. Zero-variance segments → `NaN`.

> **Superseded by measurement (§7).** On the probe-logit series at 10-step logging this
> statistic saturates at its *ceiling* (~0.97 of 1) in both treatment and control: the
> series is noise-dominated at that sampling, so a linear fit explains nothing inside a
> short window. It is the mirror image of the weight norm being pinned at its floor —
> both are non-measurements. The statistic that separates the runs is the velocity below.

**Detector.** Segments labelled by their right edge (centre labels read half a window
into the future). After memorization is confirmed from training accuracy, warn when the
statistic exceeds its floor by a margin over several consecutive segments. Two knobs
(margin, persistence), fixed on calibration runs, then frozen. Rise-based polarity
makes a "still moving" guard unnecessary: a frozen or merely rescaling model gives
straight series → floor → silence.

> **Polarity corrected by measurement (§7).** The observed signal is not a rise. Logit
> velocity is *high and sustained* through the plateau and *decays* through
> generalization, monotonically — so the rule is "the function is still moving long
> after memorization", not "something spikes just before grokking". The argument above
> survives in one respect that matters: a frozen or merely rescaling model still gives
> silence, which is exactly what the control does.

**Offline tier.** MacKay–Ghahramani intrinsic dimension with Theiler exclusion and an
identifiability check (does the estimate move when the embedding dimension doubles?) is
computed offline for interpretation. It joins the detector only if it detects something
roughness misses; on all existing logs it does not.

## 4. Why a signal should exist

Mechanistic wager: interpretability work (Nanda et al. 2023; Chughtai et al. 2023)
shows the generalizing circuit forms *gradually during the plateau*, and some of its
progress measures are logit-functions — outputs drift while accuracy, which sees only
the argmax, is flat. Predicted shape in our coordinates: frozen directions (floor)
while the model only rescales a memorized solution; wandering directions (above floor)
while memorized and generalizing structures compete; return to floor after cleanup.
Rise-then-return — the weight norm's shape, but in coordinates where the weight-decay
explanation is unavailable.

## 5. What it solves

| Previous failure | Fix |
| --- | --- |
| Observable mechanically coupled to WD | Scale-invariant probe logits |
| No reference value → baseline machinery, clamps, fabricated E=1 | Exact floors; degenerate segments `NaN` |
| Unidentifiable "dimension" quoted as a measurement | Detector uses departure-from-floor only; dimension demoted to offline, identifiability-gated |
| Centre-labelled (acausal) windows | Right-edge labels |
| Validation loss as observable | Probe from training distribution |
| Controls vary only WD → specificity unprovable | New WD>0 controls (§6) |

## 6. Validation: nulls, controls, readings

Must beat a null hierarchy, with parameters frozen beforehand and splits by whole runs:
**N0** — logit velocity $\|\widetilde Z_t - \widetilde Z_{t-H}\|$ (if plain drift
already separates all controls, the geometry layer is unjustified); **N1** — weight-norm
roughness (current champion; the new observable must beat it on *specificity*, not
necessarily lead time).

Runs: (1) grokking, several seeds; (2) WD=0, memorizes forever; (3) **WD>0, never
generalizes** (e.g. train fraction below the grokking threshold) — the
confound-breaker, run it first; (4) **WD>0, no gap** — no fabricated lead allowed.

Readings: fires on (1), silent on (2)–(3) → genuine generalization predictor. Fires on
(1) and (3) → detects regularization pressure, not generalization — a publishable
correction of the weight-norm story. No signal on (1) → weight-space reorganization
precedes function-space change; informative about where the transition lives. Every
branch is a result.

## 7. First results (measured)

Modular addition, $p = 113$, T4, probe verified non-invasive (training logs bit-identical
with and without it). `mod_wd1` grokked at step **13700** against the documented ~13810 —
the shift is GPU float64 reduction order — and `mod_wd0` never generalized, so fidelity
to the published runs holds. Reproduce with `analyze_probe.py`.

**Roughness does not discriminate.** 0.97–0.98 in *both* runs, against a ceiling of 1.
Per-row projection noise is ~4.4e-3 against a full-run range of ~0.31, so at 10-step
logging each short window is noise. See the note in §3.2.

**Velocity separates the runs by ~360× and is shaped like a predictor.** Median
per-example movement of the normalized logits:

| phase | `mod_wd1` (grokks at 13700) | `mod_wd0` (control) |
| --- | --- | --- |
| 0–4k | 1.7e-01 | 1.8e-01 → 1.0e-03 |
| 8k–12k | 1.1e-01 → 8.5e-02 | 3.3e-04 → 2.1e-04 |
| 13k–14k (grokking) | 5.0e-02 | 1.7e-04 |
| 15k–20k | 1.4e-02 | 1.2e-04 |

Two readings. The control's function **freezes within ~2000 steps** — three orders of
magnitude — and only rescales thereafter, which is precisely what the scale-invariant
observable was built to expose. The grokking run keeps **materially reorganizing for
13 000 steps** while validation accuracy sits near zero, then settles once generalization
completes.

**This is not yet a generalization predictor.** Both effects track weight decay, the
confound of §2, and `mod_wd0` cannot separate the two hypotheses. The WD>0-that-never-
generalizes run of §6 remains the decisive experiment; if it also shows sustained
velocity, this detects regularization pressure and the negative result stands.

Note also that N0 — the intended *null model* — is the statistic that works, exactly the
pattern the audit found for the weight norm. The geometry layer has not earned its place.

## 8. Risks

- **Noise floor**: weight jitter may lift the floor above 0/1; fallback is one per-run
  reference median after memorization — not the draft's baseline state machine.
- **Lead time**: outputs may move later than weights; specificity may cost lead — report
  the trade, don't hide it.
- **Spikes** (slingshots) raise any roughness statistic; persistence filters isolated
  ones.
- **Scope**: developed on algorithmic tasks; transfer untested, but cost (one probe
  forward pass per log step) is scaling-compatible.

## 9. What was dropped from the MG-dimension draft

Most of that document was machinery for problems an exact-floor statistic does not have:

| dropped | why |
| --- | --- |
| baseline state machine (EMA gate, hysteresis, MAD bands) | exists only because an MG dimension has no absolute reference and must be compared to a per-run baseline; roughness and PR have exact floors |
| `V_t` "still moving" guard | needed only for a *drop*-based detector, which stabilization mimics; a rise-based one is silent by construction |
| fast + slow drop branches, Theil–Sen slope | replaced by one rise rule |
| consensus fraction over 16 projections | correlated observers make the fraction uninterpretable as evidence, as the draft concedes; report projections individually |
| PCA family, fit/reference splits, IncrementalPCA | an optional confirmation tier by the draft's own admission — machinery before evidence |
| two-tier WARNING / CONFIRMED | no evidence one tier is insufficient |
| ~22-parameter table | every free parameter must be pinned down on calibration runs before the test runs mean anything; ~5 remain |

Two corrections carried over: the grokking gap is judged *relative* to $T_{mem}$ (the
draft's absolute 10 000-step floor would exclude `s5_wd1`, measured gap 3 900 steps), and
$T_{mem}$ / $T_{gen}$ follow the article's Definition 1 ($\epsilon = 0.05$ for both)
rather than the draft's mismatched 0.99 / 0.95.

## 10. Open questions

1. **Does the WD>0/no-grokking control fire?** The decisive one — it decides whether §7's
   separation is about generalization or about weight decay.
2. Answered in part by §7: the projections show sustained-then-decaying movement, not the
   weight norm's rise-then-collapse. Open: is the *decay through grokking* sharp enough
   at finer logging to time the transition, rather than only to flag "still moving"?
3. Does any geometry statistic add power over plain velocity once the sampling noise is
   handled (longer windows, or logging every step)? So far it does not.
4. How few probe inputs and projections suffice before the signal degrades?

## 11. Reuse

Estimators, causal sliding driver, degeneracy handling, tests:
[`../grokking_analysis/edm/`](../grokking_analysis/edm/). Training/logging for new
runs: [`../grokking_train/`](../grokking_train/). Only the probe-logit hook is new.
[`README.md`](README.md) is how to run it.
