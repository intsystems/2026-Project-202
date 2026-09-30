# MNIST optimizer-dynamics monitoring, 2026-09-29

Purpose: test whether scalar MG tracks simplification of the WEIGHT TRAJECTORY
during actual image classification, not class coverage or representation rank.
No guarantee that LR reduction simplifies geometry; independently measure it.
Covariance participation ratio is an independent empirical descriptor, NOT true
active dimension or a theorem about the scalar estimator.

Data: MNIST real images, average pool28x28 to14x14, input[0,1]. Balanced2040 train
(204 per class) from official training split, all10000 official test images.
Fixed data seed260929. MLP196-64-32-10,tanh,tanh,all15018 parameters trained,
crossentropy, no augmentation, no label/rank alteration, no weight decay.
Pilotseed0: fullbatch GD lr1/3 and minibatch64 SGD lr.1 momentum.9. Limited to
these3 regimes, inspect independently before MG. Steps4096,switch2048,checkpoint256.
Each regime runs unchanged plus lr/10 continuation of exact same trained state
and exact data RNG. Training log loss, gradient norm and one fixed Rademacher
projection of full weights are recorded each step. Primary actual training loss;
secondary projection and gradient norm; report all, no post-hoc channel victory.
No extra probe-forward required. Save full theta BEFORE corresponding loss/update.

Primary reference window512,end every256: detrend each parameter by removing
its mean and least-squares linear time trend; covariance PR=tr(G)^2/||G||_F^2,
G=X X^T. This can be computed EXACTLY without an eigendecomposition: include
this fast exact baseline instead of inflating reference cost with unnecessary SVD.
Also save raw-centered PR, squared detrended movement, eigenspectrum at selected
windows, update covariance PR, accuracy and loss. Tiny residuals flagged.
Cheap approximation: fixed128 uniformly sampled parameter coordinates, with
same detrending. Include in timing/accuracy comparisons. Independent setup/no MG.

MG: W512,E20,tau1,k20,Theiler39; E40 at same exclusion for diagnostics. Report
degenerate windows. SensitivityW256/1024,tau4 with Theiler156. No range clamping.
Cheap scalar metrics mean/std,FFT entropy,lag1 autocorrelation,trend crossings.
Amplitude-scale and observer-smoothing controls; IAAFT comparisons on selected
windows. Interpret surrogate differences only as that test, not proof of theory.
All controls retain actual unmodified training/reference trajectories.

Pilot comparison before: end1024/1280/1536/1792/2048; after:end3072..4096 step256.
Independent task/geometry acceptance: trainacc>.9,testacc>.8; median PR ratio
after/before<.8 and lower ratio than unchanged arm; detect actual increase or
no change instead of assuming. No selection by MG during reference screening.
If no regime passes reference gate, report it; any extension declared exploratory.
If a regime passes, inspect all scalar channels and sensitivity, then explicitly
freeze a confirmation protocol on NEW seeds1..5. No filtering heldout seeds.
Pilot findings are exploratory even when they support a mechanism. Confirmation
must use primary channel chosen before viewing confirmation results and compare
cheap metrics including subsampled trajectory reference.

CPU-only available. Store full raw float32 trajectories (window work float64).
Sequential timing, warm calls, record acquisition overhead and input data volume.
Do not claim runtime win over practical alternatives merely by comparing to full
covariance/SVD. No large-model,GPU,CIFAR,LLM or automatic-LR-policy claim.
Context: Cohen et al. ICLR2021, arXiv2103.00065 (nonmonotone fullbatch GD);
this experiment does not claim edge-of-stability without measuring sharpness.

## Exploratory observer correction after pilot results, before new seeds
The working SGD pilot passes independent reference (PR ratio .639 vs .852 base),
but original batch-loss MG does NOT track it (ratio1.034). Preserve this negative
result. Fullbatch lr1 is already approximately PR1; lr3 fails image accuracy.
Re-evaluate archived SGD weights on fixed100 balanced TRAIN images and fixed100
TEST images (selection seed10299), no retraining, no optimizer use of test data.
Measure raw and linearly detrended windows for these two loss observations and
for the existing random parameter projection. Detrending is a change of observer
preprocessing and is exploratory; it can change MG, not an invariance theorem.
No primary-observer substitution in the original pilot. A new confirmation on
seeds1..5 can adopt a corrected observer only with this pilot selection disclosed.

## Frozen confirmation after corrected-observer pilot, before seeds1..5
Pilot raw fixed held-out100 loss MG: ratio .747 after LRdrop, .836 unchanged;
this is only a modest differential (ratio-of-ratios .894), not a strong success.
Raw fixed TRAIN probe gives .798/.828, weaker contrast. All comparisons retained.
Choose raw fixed HELDOUT100 loss as corrected primary, no detrending; freeze all
other settings. Those100 official-test images are now a MONITORING validation
set, not unseen test data. Report final accuracy on the remaining9900 separately.
The same fixed data/probe split is used for seeds1..5; these are independent
initializations, not independent datasets. No heldout seed excluded or tuned.
Full trajectory PR must decrease more in LRdrop than matched unchanged arm.
Scalar success likewise requires differential decline, not just a lower value
than early training. Report sign agreement across5 paired seeds, ratios, failures,
raw/trend sensitivity and all original channels. Cheap metrics included.
Do not infer the intervention caused a lower intrinsic dimension from PR alone.
No inference about automatic LR control: LR change is an externally scheduled
intervention, the method's task here is diagnostic monitoring of ensuing dynamics.

## Final interpretation and timing audit
Keep the distinction: only detrended weight covariance PR gives the main five-pair
agreement. Centered raw covariance PR does not, and update covariance PR gives3/5.
MG main protocol gives4/5, weak median differential; W1024 gives5/5 only in a
sensitivity analysis of the same runs. No further selection/retraining was done.
The 9900 accuracy subset excludes the fixed probe and gradients, but aggregate
accuracy on all10000 was logged during training, including the pilot. Thus the
9900 accuracy is a separate evaluation, not a blind untouched final test.
Benchmark explicitly loads lazy numerical libraries before restricting all thread
pools to1; audit.json verifies their thread counts. Acquisition+analysis totals
are projections from measured components, not end-to-end online timings.
