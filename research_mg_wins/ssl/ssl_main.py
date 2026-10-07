"""S2: label-free selection of SSL configurations / checkpoints and collapse warning -- PROTOCOL.

Written 7 Oct 2026 05:17 (UTC+3) after the pilot on seed 99 (only ground truth and training
behaviour were looked at: final linear-probe accuracy and RankMe per config, see pilot.log)
and BEFORE any MG or competitor statistic was computed on any run.

Setting. Joint-embedding SSL (SimSiam, BYOL, VICReg, SimCLR) can collapse completely or
dimensionally (Jing et al. ICLR 2022). Without labels, configurations/checkpoints are chosen
by label-free proxies: RankMe (Garrido et al. ICML 2023), alpha-ReQ (Agrawal et al.
NeurIPS 2022), LiDAR (Thilak et al. ICLR 2024), the SSL loss. Question: can MG of a cheap
scalar training log rank configurations by downstream quality as well as these proxies and
the scalar competitors, and warn early of collapse?

Model/data (ssl_core.py). MNIST, MLP encoder 784-256-256 (BN, ReLU) = representation h
(256-d); projector 256-256-emb (BN) = embedding z; predictor emb-64-emb for SimSiam/BYOL.
SGD momentum 0.9, batch 128, warm-up 100 steps then constant lr, 3000 steps. Augmentations:
random affine (shift/rotation/scale), random erasing, pixel dropout, Gaussian noise; three
strengths. Grid (configs.MAIN, 29 configurations, fixed after the pilot): SimSiam (lr, wd,
no predictor, no stop-grad, aug strength, emb 16, no projector BN), BYOL (EMA, no predictor,
strong aug), VICReg (lr, cov=0, var=0, var=5, cov=10, weak aug), SimCLR (temperature,
aug strength). Pilot (seed 99, 4000 steps): final linear-probe accuracy 0.12-0.97, 5/29
runs at or below random-init features, RankMe(z) from 1 to 61 -> healthy, dimensionally
collapsed and completely collapsed runs are all present.

Seeds. Calibration: 0, 1. Test: 10, 11, 12. Pilot seed 99 is not used in any number.

Logs (per step, scalars): loss (SSL mini-batch loss), param_norm, grad_norm, probe_z_norm,
probe_h_norm (norms of the embedding / representation of ONE fixed training image, eval mode).
Ground truth: linear-probe accuracy on h (logistic regression C=0.1 on standardised h of
5000 labelled training images, scored on 5000 test images) at steps 0, 500, ..., 3000.
Secondary truth: kNN-20 (cosine) accuracy.

Selectors (every one computed from information available at the scored step t):
  MG (primary): E<=20, tau=1, k=20, Theiler=embedding span, on the window of the last
     W=1000 steps of one log. MG_t4k50 (tau=4, k=50) reported as a secondary variant.
  Scalar competitors on the same window and log (baselines.py, tau=1 for delay ones):
     spectral_entropy, self_repeat, roughness, perm_entropy, recurrence_rate, corr_dim,
     twonn, linear_pr, cifar_events.simple (crossings, lag1, det_std); level statistics
     mean (on the loss log = the "SSL loss" rule), rel_std, rel_change.
  Domain label-free proxies (need embeddings of N images): RankMe(h) [N=2048],
     RankMe(z), alpha-ReQ(h) (power-law fit on eigenvalues 1..128) and -|alpha-1|,
     LiDAR(z) (256 images x 8 augmentations), SimSiam output std of normalised z.
  Trivial: random choice, the default configuration (ss_base), oracle (regret 0).
Calibration rule (identical for every selector): on the calibration seeds choose the log
(5 options; proxies have one) and the sign (+/-) that maximise the objective of the task;
apply unchanged to the test seeds. A NaN/undefined score (e.g. constant log) is ranked as
the worst score for every selector.

Tasks and metrics (test seeds; per seed across the 29 configurations, then mean):
  T1 configuration ranking at t=3000: Spearman rho(score, final lin acc) [calibration
     objective], top-1 and top-3 regret (accuracy points), mean within-family rho; paired
     bootstrap over configurations of rho(MG) - rho(other).
  T2 failed-run detection at t=3000: run is "bad" if its final lin acc < (mean random-init
     lin acc of the seed) + 0.02. AUC [calibration objective = AUC].
  T3 early warning: score from the window ending at t=1500 (proxies at step 1500) predicts
     final "bad"; AUC.
  T4 within-run checkpoint choice among t = 1000, 1500, ..., 3000: mean regret against the
     best checkpoint (lin acc at that checkpoint); trivial rule = last checkpoint.
Costs: wall-clock per window for every scalar statistic; RankMe/alpha/LiDAR per evaluation
including the forward passes; training ms/step.

Predictions (before running): RankMe/LiDAR are best or near best on T1 (they look at the
representation itself; MNIST probe accuracy correlates with rank). Complete collapse makes
the logs nearly constant/noise-like, so many scalar statistics (MG, roughness, self_repeat,
rel_std) separate failed runs (T2 AUC high) and MG will not be uniquely best there. On T1
inside the healthy range MG is expected to tie or lose to RankMe; whether it beats the
scalar competitors is open. T4: last checkpoint is hard to beat.

Run order: run_grid.py runs_main MAIN 0 1 10 11 12 ; features.py runs_main feats_main ;
analyze.py runs_main feats_main results_main.
"""
