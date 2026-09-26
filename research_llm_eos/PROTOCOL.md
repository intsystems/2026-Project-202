# Protocol v2 (written before the Tiny Shakespeare runs)

The initial `results/` directory is a debugging pilot on project documents. It is
NOT evidence for edge of stability: it used AdamW, an inappropriate GD stability
ratio, and a power iteration that did not distinguish largest algebraic from
largest-magnitude eigenvalues. Use `v2/` for the corrected experiment.

Two separate questions:
1. Practical arm: stochastic AdamW next-character prediction on Tiny Shakespeare,
   constant LR 0.003 versus a tenfold drop halfway through training. Matched seeds
   and batches. A schedule change is known; a reduction in active dimension is NOT
   assumed. Primary scalar is fixed validation-probe loss, with train loss and
   gradient norm reported too. Probe acquisition cost counts toward MG overhead.
2. Mechanistic arm: plain full-batch GD on a fixed 1024-token training subset of
   the same corpus. The Hessian is of this SAME objective. Scan LR using seed 0,
   then repeat selected stable and oscillatory configurations with seeds 1 and 2.
   Only this arm interprets eta*lambda_max/2 as a local GD stability ratio.

Architecture: 2 decoder blocks, width 64, 4 heads, MLP 256, context 64,
LayerNorm, GELU, tied embeddings, normal(0,0.02) linear/embedding initialization.
No dropout, clipping, artificial oscillatory input, or rank restriction.
Corpus: first 90% training, last 10% validation; immutable SHA-256 recorded.

Primary MG: repository pooled estimator, E=20, tau=1, k=20, window=512,
stride=128, Theiler exclusion=39 (also covers doubled E=40 embedding).
E=40 identifiability ratio, trend crossings, and degeneracy always reported.
This is a declared new geometry, not the article's frozen calibration.
Sensitivity: W=256/1024, tau=1/4, with exclusion 39*tau. No setting selected
to improve agreement with the Hessian. Report missing/invalid estimates.

Baselines: largest algebraic Hessian Ritz value using fully reorthogonalized
Lanczos (30, then 60 steps if residual >3%); activation covariance participation
ratio via a small d-by-d Gram matrix, not an intentionally slow full SVD;
four-microbatch gradient variance proxy (not an unbiased gradient noise scale);
cheap scalar std, detrended std, lag-one autocorrelation and spectral entropy.
Hessian is on the validation probe in AdamW (no EoS inference), and the full fixed
training subset in GD. Report the operator, Ritz residual and actual HVP count.

Time accounting: same device/thread limit, no concurrent benchmark jobs; raw
per-checkpoint times; MG-only and MG+validity checks separately, warmed repeats;
MG+probe acquisition end-to-end cost; same diagnostic timestamps where possible.
Simple loss statistics and activation PR may be faster than MG: report this.

Success requires a reproducible dynamic signal, not only cheap computation.
A transient dip caused by a window straddling the LR change is not sufficient
evidence of a sustained dimension decrease. Scalar MG and Hessian/PR are different
quantities; no equality, replacement of Hessian, or exact dimension is presumed.
No article claims will be added until results support them.
