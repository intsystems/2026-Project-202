# Rebuttal draft

This draft responds to the three official reviews of Submission 55. It refers to the current
revision of the paper, whose main text is nine pages before the references.

## Opening response

We thank the reviewers for recognising the paper's conceptual distinction between available,
functional, active, and covariance effective dimensions, as well as the validation protocol and the
negative result for scalar grokking logs. We agree that the main remaining issue was presentation:
the previous version made the reader reconstruct the problem statement, the scope of the estimator,
and the role of the diagnostics from a dense sequence of definitions.

In the revision we made the following changes:

1. We added an explicit **Problem statement** section that separates the target quantity
   (d_{\mathrm{act}}(R)) from its finite-sample estimator and states when the estimate must not be
   interpreted as a dimension.
2. We rewrote **Contributions** around what was done experimentally: a scalar-log protocol, its
   controlled validation on systems with known active dimension, and its application to grokking.
3. We added a compact reader's guide to the four quantities and reorganised the three dynamical
   regimes as a bullet list.
4. We state the relevant Takens, Sauer, Stark, and Levina--Bickel results without proofs in the
   appendix, including their assumptions and the fibre-wise limitation of the forced-system result.
5. We added a compact table of window choices. It distinguishes the delay span
   (S=(E_{\max}-1)\tau), the recurrent period (P), and the local estimation window (W), and
   states which settings support a dimensional level and which support only a change statistic.
6. We surfaced the main limitations earlier and shortened the main text to nine pages. The detailed
   sweeps and per-run tables remain in the appendix/supplement.

The scientific scope is deliberately narrow: the scalar estimate is a dimension only in the
validated recurrent regime. For both grokking settings, the diagnostics reject that interpretation.
The positive grokking result is instead a reversible simplification of saved trajectory sketches,
with a weaker matched-window scalar-log change statistic.

## Reviewer Mh7z

### M1. The paper is too dense and needs a reader's guide

**Response.** We agree. The revision adds a table immediately after the definitions of the four
quantities. It gives, for each quantity, what it measures, what data it needs, and what cannot be
concluded from its decrease. In particular, it makes explicit that covariance effective rank can
fall because of unequal excitation while the active dimension remains fixed.

We also rewrote the regimes as three bullets: deterministic recurrent, deterministic transient, and
stochastically forced. The surrounding text now states the operational consequence in one sentence:
when recurrence, identifiability, or a stable scaling range fails, the scalar output is a statistic of
the window and is not reported as an active dimension.

**Suggested rebuttal wording.**

> We agree that the previous exposition required too much reconstruction by the reader. We added a
> four-quantity guide and a three-regime bullet list, and moved the key scope limitation next to the
> estimator: outside the validated recurrent regime, the scalar output is retained as a statistic but
> is not interpreted as a dimension. We also added a compact window table distinguishing the delay
> span (S) from the local window (W).

### M2. A short guide to the four quantities and three regimes is needed

**Response.** Fixed in the revision. The quantity guide is Table 1. The three regimes are listed in
Section 3.4, with their consequences for the estimator. The new window table is also in the main
text because windowing was one of the places where the previous presentation was hardest to follow.

The window notation is now explicit:

- (S=(E_{\max}-1)\tau) is the span of one delay vector;
- (P) is the recurrent period;
- (W) is the number of samples pooled into one local estimate;
- (N) is the full record length.

Thus (S/P) controls whether the delay coordinates resolve the orbit, whereas (W) controls
locality and transition resolution. The table reports (W=8000) for controlled recurrent
systems, approximately (4000) for a coarse training-log diagnostic, (60) for the matched
transition comparison, and (600) for the published saved-trajectory measurement; the longer
full-batch sweep is reported separately because it tests resolution rather than producing a single
recommended window.

### M3. The scalar estimator has a low saturation ceiling

**Response.** We agree and now state this limitation quantitatively rather than presenting the
method as a general high-dimensional estimator. With the frozen eight-component configuration, the
scalar estimate saturates above approximately eight components. A separate sweep shows that the
rank-tracking limit does not simply equal (E_{\max}/2) or (2\log_{10}N), and does not exceed
approximately eleven components under the tested settings. We therefore claim accurate recovery only
in the validated low-dimensional range and treat higher values as ordinal/diagnostic.

This ceiling is part of the method's scope, not a hidden failure: the grokking application does not
use the absolute level as a dimension, because the diagnostics reject that interpretation before any
such claim is made.

**Suggested rebuttal wording.**

> We agree that the saturation ceiling should be more prominent. We now state that the frozen
> configuration is quantitatively reliable only up to about eight active components, while the
> tracking limit in our sweeps remains below roughly eleven. We do not interpret larger scalar
> outputs as dimensional counts; in the grokking experiments the validity diagnostics reject the
> dimensional interpretation altogether.

### M4. The result is sensitive to windowing, especially in grokking

**Response.** We agree, and the revision separates two distinct issues that were conflated in the
previous version. The delay span (S) must be matched to the recurrent period in the controlled
systems. A training log does not provide that period, so we do not interpret its absolute scalar
level as an active dimension. The local window (W) determines what temporal changes can be
resolved.

For grokking, the coarse windows are used only for diagnostics and cannot resolve a transition that
lasts a few hundred optimiser steps inside a record of tens of thousands of steps. The matched
short-window analysis uses (W=60) and reports a change statistic, not a dimension. It tracks the
same event as the saved trajectory measurement in four regularised runs, with (24/26) usable cells
separating the four positive runs from two controls; surrogate comparisons retain the effect. We
state explicitly that observer smoothness remains an alternative explanation for a scalar-log
change.

The saved full-batch trajectory was also evaluated over a range of windows. The published (W=600)
measurement is too short to reveal the relevant full-batch feature; a feature appears only at scales
around (10^4) steps and does not separate the paired runs. This is why the full-batch result is
reported as a resolution limitation rather than as evidence that no trajectory simplification can
occur.

**Suggested rebuttal wording.**

> We agree that window sensitivity must be stated more explicitly. The revision distinguishes the
> delay span (S), which controls reconstruction, from the local window (W), which controls
> temporal resolution. In grokking, the scalar level is rejected as a dimension by the diagnostics;
> the short-window result is only a change statistic. The saved full-batch trajectory is also swept
> over longer windows, where the apparent short-window null is identified as a resolution limit.

### M5. The method's practical limitations should be more visible

**Response.** We moved the limitations into the conclusion and made them explicit in the problem
statement and reader's guide. The method requires recurrence, a suitable lag, and a stable local
scaling range; the lag is not identifiable from an arbitrary training log. The controlled systems
are externally forced, so the validation establishes a conditional result. The grokking logs fail
the validity checks, and their scalar outputs are therefore not dimension estimates. The direct
trajectory result is a covariance effective-rank collapse and recovery, not a proof that
(d_{\mathrm{act}}(R)) changed.

## Reviewer ZLX4

### Z1. Strength: the paper separates available, functional, active, and covariance dimensions

**Response.** We thank the reviewer. This distinction is central to the revision and is now made
more accessible through the new quantity guide. We also state the data requirements and interpretation
limits for each quantity so that a decrease in one is not read as a decrease in another.

### Z2. Strength: the paper includes controls and failure diagnostics

**Response.** We thank the reviewer. The revision makes the role of the diagnostics more explicit:
the identifiability ratio detects dependence on the embedding dimension, while the trend-crossing
count separates monotone transients from non-monotone records. Neither statistic alone proves
recurrence, so the appendix also documents the recurrence/exclusion audit. The diagnostics are used
as a gate on interpretation, not as evidence that every scalar log contains a valid dimension.

### Z3. Weakness: low saturation ceiling

**Response.** We agree. See the response to Mh7z's M3 above. We now report the ceiling as a scope
limitation, distinguish the returned level from the highest rank that is tracked, and avoid treating
the scalar estimate as a count above the validated range.

### Z4. Weakness: sensitivity to windowing and phase transitions

**Response.** We agree. See the response to Mh7z's M4 above. The main text now includes the window
table and explicitly labels the grokking analysis as a change comparison. The longer window sweeps
are in the appendix and show why the full-batch short-window measurement cannot resolve a transition
at that scale.

## Reviewer Jgcu

### J1. The contributions and problem formulation are difficult to identify

**Response.** We agree. The revision adds a dedicated Problem Statement section before the formal
definitions. It now asks a concrete question: given only a scalar training log, can we estimate the
number of independent components of the recurrent regime visited by the optimiser, and can we detect
when the log and window do not support that interpretation?

The revised contributions emphasise what we did rather than presenting the definitions as the main
contribution:

1. We proposed and calibrated a scalar-log protocol for estimating active dimension, with explicit
   validity diagnostics.
2. We tested it on six constructed systems with independently known or measured ground truth,
   including controlled failure modes.
3. We applied it to grokking and found that the standard scalar logs fail the dimensional validity
   checks, while saved trajectories show a reversible local simplification near generalisation in
   regularised runs and in some controls without weight decay.

The method is computationally light and preserves training dynamics through a scalar observer under
the delay-embedding assumptions, but the revision avoids claiming that the scalar output is a valid
dimension on arbitrary training runs.

**Suggested rebuttal wording.**

> We agree that the previous version foregrounded definitions before stating the problem. We added a
> dedicated Problem Statement and rewrote Contributions around the actual work: a calibrated
> scalar-log protocol, controlled tests with known active dimension, and the grokking application.
> The central grokking conclusion is now stated conservatively: standard logs fail the dimensional
> validity checks, while saved trajectories reveal a reversible simplification signal that is not by
> itself a proof of active-dimension reduction.

### J2. The theoretical results are cited but not presented

**Response.** We agree that the assumptions should be visible. Appendix A now states, without proofs,
the four results used by the method:

- Takens' delay-embedding theorem;
- the Sauer--Yorke--Casdagli prevalence extension for fractal sets;
- Stark's delay embedding result for forced systems, stated fibre by fibre;
- the Levina--Bickel maximum-likelihood intrinsic-dimension estimator and its local assumptions.

The main text now states what these results do and do not imply. In particular, the Stark result does
not certify a single mini-batch training trajectory as a sample from one invariant fibre: a realised
training run may cross forcing fibres. This is why the scalar estimate is paired with diagnostics
and is rejected as a dimension in the grokking settings.

**Suggested rebuttal wording.**

> We added theorem statements without proofs in Appendix A, including the assumptions and the
> fibre-wise qualification for forced systems. We also clarified that the theorems justify the
> reconstruction mechanism under their conditions; they do not make every finite stochastic training
> log a valid sample for intrinsic-dimension estimation.

### J3. The appendix is very long, and it is unclear what belongs in the main paper

**Response.** We agree that the previous version did not guide the reader through the paper's two
levels of material. The main paper is now nine pages before the references and contains the material
needed to understand the claim: the problem statement, quantity guide, estimator, three regimes,
diagnostics, main validation result, window-size table, grokking result, and limitations.

The appendix/supplement contains reproducibility and audit material: theorem statements, the frozen
configuration, per-observer and per-run tables, construction details, window sweeps, Theiler audits,
and the complete grokking inventory. These details support the main claims but are not required to
follow the argument. We kept the window table and the four-quantity guide in the main text because
they directly address the reviewers' clarity concerns.

**Suggested rebuttal wording.**

> We reorganised the paper so that the main text is self-contained and nine pages before the
> references. It now includes the problem statement, reader's guide, estimator, regime diagnostics,
> principal validation result, window choices, grokking result, and limitations. The long appendix is
> explicitly labelled as supporting material containing theorem statements, audits, full tables, and
> reproducibility details.

### J4. The definitions of (\mathrm{PR}^{\mathrm{pos}}), (\mathrm{PR}^{\mathrm{upd}}), and
### (\mathrm{PR}^{\mathrm{det}}) are unclear

**Response.** We agree. The revised definitions are placed at the first introduction of the
participation ratio. For a spectrum \(\lambda_i\),

\[
\mathrm{PR}(\lambda)=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
\]

In the current notation, \(\mathrm{PR}^{\mathrm{pos}}\) is the participation ratio of the
trajectory covariance in a window, while \(\mathrm{PR}^{\mathrm{det}}\) is computed after removing
the least-squares line from that trajectory and therefore measures variation around the local drift.
The unused \(\mathrm{PR}^{\mathrm{upd}}\) notation was removed from the current paper rather than
leaving a quantity that is not used in the reported results. The quantity guide also states that
neither effective-rank variant is the active dimension.

### J5. Which theorems are meant by “the theorems require a deterministic flow on a compact invariant set”?

**Response.** The sentence now names the results directly: Takens' theorem and the Sauer--Yorke--
Casdagli extension are the deterministic delay-embedding results used for the recurrent setting;
Stark's result is the forced-system extension, with a fibre-wise conclusion. The Levina--Bickel
result concerns the local maximum-likelihood estimator, not the validity of delay reconstruction.
The revised text separates these roles and states that the assumptions are conditions for
interpretation, not facts automatically satisfied by a training run.

### J6. Section 3.3 should explain the dynamical regimes in bullet form

**Response.** Fixed. The regimes are now presented as three bullets:

- deterministic and recurrent: the scalar estimate can be interpreted as active dimension if the
  other diagnostics pass;
- deterministic and transient: the estimator returns a window statistic because the record does not
  sample a recurrent occupation measure;
- stochastically forced: a single run can cross forcing fibres, so the fibre-wise theorem does not
  license a dimension estimate for the whole record.

The section ends with the operational rule that a failed condition removes the dimensional
interpretation but does not prevent us from reporting the statistic as a descriptive quantity.

### J7. It is unclear what Figure 1(b) represents and how it connects to Figure 1(a)

**Response.** We revised the caption and surrounding text. Panels (a)--(c) now explicitly refer to
the same one-phase scalar record: (a) shows the first 400 scalar samples, (b) reconstructs the whole
record in its first two delay coordinates, and (c) shows the neighbours of one reconstructed point.
The repeated turns in panel (b) explain why the neighbours in panel (c) are returns from other passes,
rather than adjacent samples. This makes the figure an illustration of the estimator's data flow:
one scalar log, delay reconstruction, and a neighbourhood statistic.

## Points that are already fixed versus points to state cautiously

### Already fixed in the current revision

- explicit problem statement and separate target/estimator distinction;
- contributions focused on the protocol, validation, and grokking application;
- reader's guide to four quantities;
- bullet-format explanation of the three regimes;
- theorem statements and assumptions in Appendix A;
- definitions of the participation-ratio variants at first use;
- main-text table of working window sizes;
- explicit distinction between dimensional levels and transition/change statistics;
- limitations moved into the main conclusion;
- nine-page main text before references.

### Claims to phrase carefully in the rebuttal

- Do not say that the scalar log measures active dimension in grokking. The current result is that the
  diagnostics reject that interpretation.
- Do not say that the trajectory effective-rank collapse proves active-dimension reduction. It is a
  local simplification signature, compatible with changes in scale or anisotropy.
- Do not describe the low ceiling as solved. State it as a quantified limitation and explain that the
  method is intended for a validated low-dimensional regime.
- Do not claim that a short window identifies the exact transition time. State its resolution and the
  fact that the matched-window result is a change statistic.
- Do not present the long appendix as part of the central contribution. It is support for validation,
  audits, and reproducibility.

## Compact rebuttal version

We thank the reviewers for recognising the paper's conceptual distinction between available,
functional, active, and covariance effective dimensions, and for highlighting the validation protocol
and the negative result for scalar grokking logs. We agree that the previous version was too dense and
introduced definitions before stating the problem.

We revised the paper in five ways. First, we added an explicit Problem Statement: given only a scalar
training log, estimate the number of independent components of a recurrent training regime and detect
when the log does not support that interpretation. Second, Contributions now emphasises the work
performed: a calibrated delay-reconstruction/Levina--Bickel scalar-log protocol with validity
diagnostics; controlled tests on six systems with known or independently measured ground truth; and
the grokking application. Third, we added a table of the four quantities and rewrote the three
dynamical regimes as bullets. Fourth, Appendix A now states the Takens, Sauer, Stark, and
Levina--Bickel results and their assumptions, including the fibre-wise scope of the forced-system
theorem. Fifth, we added a main-text window table distinguishing the delay span (S), recurrent
period (P), local window (W), and record length (N), and made the limitations explicit.

The central scope is now stated conservatively. In the controlled recurrent regime the scalar
protocol recovers the active rank to about one component up to a ceiling near eight; above that range
we do not interpret the output as a dimensional count. In both grokking settings the diagnostics
reject the scalar level as an active dimension. Saved trajectory sketches nevertheless show a
reversible local effective-rank collapse near generalisation in four regularised runs, with a similar
change statistic in a matched short-window scalar analysis. Because effective rank and scalar-log
levels can also respond to scale, anisotropy, smoothness, or finite-window effects, we present this
as evidence of local simplification rather than proof of active-dimension reduction. The main text is
now nine pages before the references; the appendix contains theorem statements, full tables, audits,
and reproducibility details.
