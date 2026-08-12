# Known-active-dimension calibration (k=1..20)

## Design

A frozen nonlinear MLP backbone is followed by a trainable adapter with 20 fixed orthonormal parameter directions. The dynamics excite r directions, r=1,...,20. The true active dimension is measured from the trajectory covariance (participation ratio), not equated to r. Functional rank is measured from the held-out-logit Jacobian.

The primary arm is deterministic and recurrent (incommensurate sinusoidal forcing). Additional arms are a slow recurrent torus, rank-r stochastic forcing, projected mini-batch noise, and full-batch transient descent. Fixed-r controls test coordinate rotation and constant observer scaling.

## Frozen estimator

The estimator configuration was selected only on r=[2, 6, 10, 14, 18] and seeds=[0, 1, 2]; test ranks are the complementary values. Configuration: `{'max_E': 40, 'tau': 16, 'k_neighbors': 20, 'theiler': 'autocorr', 'window': 8000, 'stride': 4000, 'dither': 1e-09}`.

## Held-out recurrent test

```
 observer  rho_raw  mae_raw  rho_cal  mae_cal  max_error_cal  degenerate
   c_norm    0.948    2.322    0.948    1.406          5.160         0.0
    w_fro    0.948    2.322    0.948    1.406          5.160         0.0
    g_fro    0.950    4.467    0.950    1.532          5.919         0.0
   g_proj    0.939    1.937    0.939    1.533          6.566         0.0
loss_full    0.904    5.007    0.904    1.926          8.152         0.0
  c_proj1    0.909    2.558    0.909    1.950          6.599         0.0
 fn_proj1    0.918    2.798    0.918    2.019         11.154         0.0
   fn_fro    0.902    2.194    0.902    2.136          5.855         0.0
```

## Interpretation

For the recurrent arm, a successful estimator should increase monotonically with measured trajectory PR and remain stable across observers and seeds. The stochastic and transient arms are deliberately not expected to have an r-dimensional deterministic attractor; they test whether MG spuriously reports the injected rank.

The decisive comparison is MG versus the directly measured `traj_pr`/`update_pr` and versus linear `PRdelay`. A good result supports equality only for the identifiable recurrent regime, not for arbitrary training trajectories.
