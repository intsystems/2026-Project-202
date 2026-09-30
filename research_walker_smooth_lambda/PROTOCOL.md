# Walker2d: factorial sweep of temporal policy smoothing

This is a new follow-up to the completed smooth-policy experiment. Earlier
folders and reports remain unchanged. The purpose is to test whether the
observed simplification of action logs depends monotonically on the coefficient
of the temporal policy regularizer.

The policy is initialized from the same frozen repair anchor for every arm.
Each run uses the same PPO budget and implementation as `research_walker_smooth`:
8 environments, 512 steps per environment, batch 256, 5 epochs, learning rate
3e-5 to 3e-6, clip 0.1, target KL 0.01, gamma 0.99, GAE 0.95, entropy 0,
value coefficient 0.5, gradient clipping 0.5, and 1048576 transitions.

The only varied quantity is the temporal regularization coefficient:
lambda in {0, 0.25, 1, 4}. The added loss is the mean squared difference of
clipped deterministic policy means at consecutive nonterminal rollout states.
The environment reward and observation normalization are unchanged. Exploration
noise is not penalized. The same training seed 230 is used for the pilot so that
the comparison is a factorial coefficient sweep from a common anchor.

Validation uses the final three checkpoints and resets 61001--61005. The final
checkpoint is not selected by MG. We require at least four healthy validation
episodes, at least four common eligible records with the lambda=0 arm, at least
90% of the control original padded return, and no more than one lost healthy
episode. J1 and J2 are action smoothness measures; R and D are full-state
repeatability measures. MG is reported after the validation decision and does
not select an arm.

If the pilot remains healthy and at least one nonzero coefficient reduces J1
and J2 without violating the return gate, run all four coefficients on five
confirmation seeds 231--235. Held-out test resets are 62001--62010. No
coefficient is replaced after seeing MG. If no arm passes the pilot gate, keep
the pilot as a negative factorial result and do not spend confirmation budget.

The exact active dimension of the full Walker2d system is not claimed. The
primary target is temporal complexity of the chosen action log; whole-state
metrics are independent checks and may disagree.
