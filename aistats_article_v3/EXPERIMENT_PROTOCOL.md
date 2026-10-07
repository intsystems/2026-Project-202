# Additional confirmation protocol

Fixed on 2026-10-01, after inspection of historical CNN seeds 10–13 and before new training.

The previous test set is now a development comparison. The new evidence is kept separately.

## Scalar comparisons

Same scalar parameter norm; windows 1000, stride 500. MG E20, lag1, k20,
Theiler19. New features: mean absolute increments; increments / window standard
deviation; permutation entropy (order5, delay1, stable tie ordering); sample
entropy (m2, tolerance0.2 SD, no self-matches, equal template origins for m and
m+1); normalized periodogram entropy after linear detrending. All missing or
undefined features remain missing. Every feature uses its own preprocessing.

Two detectors are crossed with these features: the historical median-block rule,
and a one-sided Page CUSUM standardized by the feature values through step3500.
CUSUM is applied to window features, including the scalar window mean, rather
than claimed to implement BOCPD or a likelihood-optimal detector. Drift choices
0.25/0.5/1; signs +/-; null threshold is the largest calibration-run maximum +0.5.
Block choices M2/3/4, B3/4/6; null margin0.02; MG block uses only decreases.
Calibration is limited to historical seeds0–3, nulls base/batch-up and events
LR/freeze/prune. Maximize event hits, tie-break by shorter delay, freeze rules.
For family-held-out analysis, repeat calibration excluding the tested intervention
family; never fit thresholds to the fresh results.

## Fresh CNN confirmation

Ten new initialization/data-order seeds300–309. Original 14666-parameter CNN,
10000-image CIFAR-10 split, SGD0.02, momentum0.9, WD0.0005, batch64. No augmentation.
Event times, independent of results: a seed-specific pseudorandom draw between
4500 and7000, inclusive, generated with seed20261001. Save the assignment before
training. Each prefix is shared exactly by its branches. The ordinary-training
control runs for14000 updates. Intervention branches run for5000 updates after
the assigned event: LR/10, freeze all but the head, magnitude prune80% with mask.
Freezing rebuilds SGD as in the historical protocol. Controls for rescaling and
causal16-point smoothing are derived from the base and reported separately.
There is no fresh batch-up training claim in this confirmation.

Primary outcome: first alarm after the event and within5000 updates; any earlier
alarm is false and precludes a hit. Report each seed, event family, control type,
and detection delay. Retain all runs. Use seed-level paired bootstrap intervals
only as descriptive uncertainty; no distribution-free false-alarm guarantee.
All branches within a seed share a prefix and are not independent repetitions.
Independent update diagnostics use a fixed256-coordinate CountSketch and actual
mean update magnitude; the mask/trainable fraction verifies imposed restrictions.
These references are not exact active-dimension labels.

## Generator stress tests

Saved H4/M4/T4 records, all5 seeds, neurons0/1/2. Test first windows4096/8192,
white observation noise0/0.01/0.05 times record SD, deterministic noise seeds.
Compare MG with entropy/increment features and a spectral harmonic grouping
baseline. The latter greedily groups significant FFT peaks related by integer
multiples (2–8), tolerance2 FFT bins; it estimates harmonic families, not general
incommensurate dimension. Report the same-neuron scalar input for all methods.
These are post-hoc robustness checks of trained networks, not35 new trainings.

## Scope and reproducibility

No outcome-dependent selection of seeds, windows or observables. Timings are
feature-analysis costs unless acquisition is explicitly included. VAE and RL
remain substantive applications; a tie with a cheaper statistic is retained.
The manuscript must be revised around actual results, including any lost claim
of superiority over simple statistics. No theoretical assertion is added merely
to make the paper appear novel.
