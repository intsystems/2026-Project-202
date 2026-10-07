# Protected latent-code control, 2026-09-29

Question: does MG distinguish loss of latent information from a change of KL
regularization when information is retained? Existing base and regularized
branches are reused exactly, never retrained or altered. Every extra branch
starts from the archived step1024 model/Adam/main-RNG checkpoint and continues
to3072. beta changes .01->1 at1024 in BOTH regularized and protected branches.
Dataset, vocabulary, fixed probe/noise, reference diagnostics and MG unchanged.

First protection: encoder5. Five additional encoder-only Adam updates before
each ordinary joint update at beta1. Encoder-only parameters: encoder GRU,mu,lv.
Shared embedding belongs to the decoder too, so it is FROZEN for inner updates,
along with init/decoder/output. Autograd still passes through the frozen decoder
to latent z. Restore all parameters for the ordinary joint update. Same Adam
state is retained; parameter-specific Adam steps advance only when gradients
exist. Separate RNG seed+90191 supplies inner batches/noise, preserving the
original outer batch/noise stream. This is a fixed-five-update adaptation of
He et al.(ICLR2019), not a reproduction of their adaptive aggressive algorithm.
MG logs once per outer update, not per inner update; additional compute is
reported and no speed comparison is inferred from this condition.

Select protection using PILOT0 REFERENCE ONLY, before computing its MG.
Meaningful retained-information criterion on median late checkpoints2048..3072:
MIprotected >= .5 * MIpre (median512..1024), and prediction-shuffle symmetric KL
protected >= .5 * shufflepre. Both also must be >=2 times regularized late value.
These thresholds are operational, not a theorem or claim of complete retention.
If encoder5 fails, retain it as a failed protection attempt, not a failed MG
specificity test. One declared fallback: freebits with lambda=.5 nats per latent
coordinate, batch-mean KL per coordinate then max(lambda,meanKL_j), summed over16
coordinates. Same beta1, one joint update, no extra encoder updates. This changes
the loss shape and is therefore a weaker isolation of the beta-change mechanism
than the unchanged-objective encoder control. Do not hide this distinction.
No other protection hyperparameter sweep. If neither protection meets reference
criteria, report the control as unestablished, without concluding MG nonspecific.

If a protection meets the reference criterion, freeze it and run it on ALL
previously planned confirmation seeds1..9, independently of its MG result.
These are prospective third branches on the same nine archived training states,
not nine new initializations. All successes/failures kept. No replacement seeds.

Only then compute MG in protected arms. PrimaryW512,E20,tau1,k20,Theiler39;
E40 ratio, floor flags; W256/W1024/tau4 retained. Before/late windows and scalar
channels identical to original protocol. Main continuous contrast:
R=lateMGprotected/lateMGregularized; desired R>1 when protection is independently
confirmed, together with protected/base q closer to1. Report all q and raw
changes even when protection only partially retains information.
An unchanged MG drop despite reference retention weakens the specificity claim.
This control does not establish causality for every possible confound: extra
optimization or a changed loss can themselves change scalar dynamics.

Reference sources: He et al., Lagging Inference Networks and Posterior Collapse,
ICLR2019, arXiv1901.05534; Kingma et al., Improved Variational Inference with
Inverse Autoregressive Flow, NIPS2016, arXiv1606.04934, AppendixC free bits.
