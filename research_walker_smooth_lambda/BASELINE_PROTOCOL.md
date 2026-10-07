# Final scalar-baseline audit (2026-09-30)

Written before calculating the baseline results. No new policy training and no
new coefficient selection. This is exploratory follow-up to already observed
MG results; it is not an independent confirmation or failure-prediction test.

Use all eligible final records of seeds 231--235, coefficients 0,.25,1,4.
Primary signal: Euclidean action norm. Secondary: norm of action increments,
mean action. Same 2048-sample windows ending at 2048,3072,4096 as cached MG.
Median windows within record; median paired reset ratios within training seed;
median and sign counts across five seeds. Reused reset bank: 62001--62010.

Fixed metrics (smaller means more concentration/repetition/smoothness):
- MG: existing E20, tau8, k20, Theiler312; all three windows must be nondegenerate.
- Spectral entropy: subtract window mean, rectangular window, real FFT, omit
  DC, normalize nonnegative bin powers, Shannon entropy divided by log(bin count).
- Recurrence error: min over lags 20..250 of mean squared difference divided
  by twice the population window variance; retain minimizing lag and whether
  it hits the search boundary.
- Normalized increment energy: mean squared first difference / (2*variance).
Zero-variance windows are invalid, never zeros indicating success.

Question 1: sign and magnitude of moderate-arm changes relative to control.
Question 2: rebound at coefficient4 versus coefficient1, using the SAME eligible
resets in both arms and control (common triplets). Also repeat all coefficient
curves on resets eligible in all four arms to expose survivor-composition effects.
An increase at coefficient4 is a descriptive group contrast, not evidence of
early-warning accuracy; no outcome labels are used to tune thresholds.

Timing: same single in-memory scalar window, one CPU numerical thread, warmup,
seven repetitions in shuffled method order. MG20 only versus each baseline;
also include the E40 diagnostic separately. Data acquisition and file I/O excluded.
Output: raw windows, per-record/per-seed/aggregate tables, matched-set checks,
runtime evidence and concise Russian report. Report all three signals.
