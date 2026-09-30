# Neural collapse and scalar-log MG — protocol, 2026-09-29

## Question and scope
Does a scalar-log MG monitor follow the emergence of a simpler class geometry
during image-classifier training? Exact recovery of active dimension is NOT the
practical endpoint. Neural collapse is measured independently from labelled
penultimate-layer features. Correlation with training time alone is insufficient.

This local study is a CPU pilot on a fixed, class-balanced subset of real CIFAR-10,
not a reproduction of a full-data ResNet-18 benchmark. No inference about the
full-data benchmark is made. We report failure as well as success.

## Frozen initial design (before inspecting MG)
- Dataset: official CIFAR-10 Python batches already present locally; 200 images
  per class for training (2,000 total), 100/class from official test (1,000 total).
  Split seed 20260929; same indices for all seeds. Images remain 32x32 RGB.
- Fixed CIFAR normalization; no augmentation, no label corruption, no collapse
  penalty and no artificial low-rank constraint.
- SmallResNet: stem width 16; three two-convolution residual blocks with widths
  16, 32, 64; global average pooling; 64-D features; bias-free 10-class head.
  BatchNorm and ReLU. All model parameters trained in the main arm.
- SGD, batch 64 sampled uniformly with replacement, momentum .9, weight decay
  .0005, constant LR .03. 4,096 optimizer steps. Seeds 0, 1, 2.
- Every step: training loss before the update; fixed balanced training-probe
  loss and test-probe loss after the update, 10 images per class in each probe.
  Probes run in eval mode and do not update BatchNorm or gradients.
- Every 128 steps and at initialization: full training/test feature extraction,
  train/test CE and accuracy; save training features and head weights.
- NC1: tr(Sigma_W pinv(Sigma_B))/K with float64 pseudoinverse rcond 1e-10.
  Companion: tr(Sigma_W)/tr(Sigma_B), avoiding hidden conditioning effects.
  NC2: distance of normalized centered class-mean Gram matrix from simplex ETF;
  class-mean norm coefficient of variation; NC3 normalized head/mean mismatch;
  NC4 classifier/nearest-training-class-mean disagreement (train and test).
- A fixed stratified 100-image training subset supplies a CHEAP reference NC
  baseline too; its features reuse the already measured fixed training probe.
- Negative control: seed 0, only classifier head trained; the feature extractor
  including BatchNorm statistics is frozen. Geometry should remain identical.
- MG primary: repository estimate(), E20, tau1, k20, Theiler39, W512, stride128;
  E40 with the SAME tau/exclusion for identifiability. No threshold tuning to NC.
  Sensitivity: W256/1024 at tau1, W512 at tau4; Theiler=39*tau.
- Cheap scalar baselines: mean loss, linear slope, raw/detrended standard
  deviation, lag-one correlation, detrended spectral entropy, trend crossings.
- Primary early/late comparison: windows ending at steps 512..1024 vs final
  1,024 steps, using seed-level summaries (overlapping windows not independent).
  Also compare first zero-error checkpoint with final geometry; Spearman before
  and after zero error, first-difference correlations and frozen-control response.
- Timing: single sequential job, recorded CPU/thread settings; distinguish
  existing-training-log MG, MG with validity checks, and probe acquisition cost.
  Compare against full-feature NC analysis AND cheap scalar/small-probe baselines.
  Baseline is efficient covariance/Gram computation, not an intentionally costly
  full state SVD. Warm repeated timings; no claim of superiority to all baselines.

## Provenance
Papyan, Han and Donoho (2020), Prevalence of Neural Collapse During the Terminal
Phase of Deep Learning Training, DOI 10.1073/pnas.2015509117.
CIFAR-10: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images
(2009), official dataset from the University of Toronto.

Model speed check before training: about .025 s/64-image update on this CPU.
No training results or MG results were available when this protocol was written.
