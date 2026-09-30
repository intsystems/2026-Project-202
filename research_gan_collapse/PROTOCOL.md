# Stacked-MNIST GAN pilot and conditional confirmation, 2026-09-29

This is a genuine image GAN training experiment, not training on its own logs.
Three independently sampled MNIST training digits form RGB channels (1000 label
combinations). No labels enter GAN training. All 60k training digits, uniform
image sampling with replacement. Test MNIST is reserved for evaluator testing.

Independent evaluator: two-convolution MNIST classifier, 5 epochs Adam .001,
batch128, seed20260929; never train on generated images. Accept a generated triple
only when ALL three softmax maxima >=.9. Report both valid fraction and coverage
and effective mode count among valid triples. Confidence is a heuristic, not an
OOD-quality guarantee; inspect image sheets. Count all covered modes and those
with >=5 accepted examples. Save per-image labels and confidences.

GAN: Generator z64 -> linear64x7x7 -> BN/ReLU -> transpose-conv32 14x14 ->
BN/ReLU -> transpose-conv3 28x28 -> tanh. Discriminator: conv32 stride2/leaky(.2)
-> conv64 stride2/leaky(.2) -> linear scalar. Kernel4,padding1 for all stride2
convs. Gaussian weight init std.02. No discriminator BN. Adam (.5,.999), lr .0002
both, batch64, one D update then one non-saturating G update. Targets [−1,1].
Separate rng seed+10000 for data/noise; evaluator latent codes fixed seed7711.
No random dropout. All checkpoints and failures retained. CPU, 4 Torch threads,
BLAS1; no simultaneous computational experiments during timing.

Pilot seed0: balanced training6144 steps, possible stress begins3072. Candidate
arms branch from EXACT same checkpoint3072: G lr x10 (`g_fast`), or D lr /10
(`d_slow`). Limited to these candidates; choose by independent reference before
ever computing MG. No rank, latent distribution, dataset class or output-shape
restriction is introduced. Stress settings represent optimizer imbalance, not
an assertion that spontaneous collapse always occurs.

Evaluate every256 steps on 10000 FIXED latent codes for paired changes; report
the first512 as small-budget baseline. Fixed common random numbers do not imply
statistical independence across checkpoints. Additional fresh-code repeats at
key checkpoints will check coverage-sampling variability if an event occurs.

Provisional event: previously valid_fraction>=.25 and effective_modes>=100;
afterward both effective_modes and coverage decrease >=40% relative to the median
of the last three pre-switch checkpoints, for two consecutive checkpoints, while
valid_fraction>=.25. Record quality deterioration separately. This cutoff is
declared before pilot reference results; endpoint ratios also reported without
dichotomization. If no candidate has such a transition, label pilot inconclusive
and do not pretend that growing diversity during initial learning is collapse.

If a candidate qualifies, freeze it and run new seeds1..5 with paired base arms.
All runs retained regardless of actual collapse. Add frozen-G control (seed1):
freeze generator including BN, continue D training. Its image distribution is
identical on common latent codes although GAN losses can change. This tests
whether a loss alarm is specific to output diversity.

MG plan frozen before GAN reference inspection: primary g_loss per G update;
secondary d_loss; W512, E20, tau1, k20, Theiler39, repeat E40 same exclusion.
Compute on trailing windows every256 steps. A window includes only data already
observed; label by RIGHT endpoint. Sensitivity W256/1024 and tau4 (Theiler156).
Early/late summary uses fully pre/post-switch windows, excluding straddling.
Negative controls: multiply log by10 and smooth by8 at fixed weights/reference.
Surrogates: IAAFT, same marginal values and approximate power spectrum. Ratios to
surrogate do not prove dimension/recurrence/nonlinearity.

Cheap scalar competitors: mean/std, autocorrelation, trend crossing count and
FFT entropy on identical windows. Practical metric alarms and false alarms will
be calibrated on pilot only; no tuning on held-out seeds or switching primary
observable after results. If reference transition is absent or MG does not
respond in pilot, do not spend the large confirmation budget on a claimed
positive example. A changed/extended pilot must be labelled exploratory.

Measure training, generator sampling, classifier, MG, E40 checks and small-budget
evaluation. Full pipeline speed must include any extra forward passes. Expensive
reference is selected for measuring mode coverage, not merely for large runtime.
Reference event time is interval-censored by checkpoint spacing; no unjustified
lead-time claim. No CIFAR-10 experiment or deployment result is implied by this
pilot; scale to CIFAR only after evidence supports the scalar-monitor hypothesis.

## Exploratory amendment before any MG computation
The full initial Stacked-MNIST pilot (seed0,6144 steps) failed the independent
quality gate: final valid triples .0605, only 228/1000 observed accepted modes.
No collapse event is claimed. Retain all results. Do not run an expensive heldout
series or claim CIFAR evidence on the basis of this undertrained generator.
Run a SINGLE-DIGIT MNIST calibration using exactly the same G/D architecture with
one output/input channel. This is a different, simpler task, not a successful
Stacked-MNIST run. Keep all other training/MG parameters; seed0,6144steps,switch3072.
Initial-quality gate for the 10-class task: valid_fraction>=.70, coverage>=8 and
effective_modes>=5 in the last three pre-switch checkpoints. The collapse event
requires >=40% fall in BOTH coverage and effective modes for two consecutive
checkpoints with valid_fraction>=.70. Same two fixed candidate interventions.
Frozen-generator negative control remains essential. None of these decisions
uses MG values; no MG computation has yet taken place.
Runtime clarification: threadpool_limits(1) limits OpenMP as well as BLAS; the
actual training/evaluation uses one effective Torch thread in this environment,
despite the outer configure(4). Record actual_torch_threads in metadata and
report the effective setting, not the nominal requested setting.

## Further reference-only warm-up extension (MG still not computed)
The single-digit6144-step pilot covers all10 classes with effective_modes9.32,
but valid_fraction .5715 is below the predeclared .70 quality gate. Extend the
SAME generator/discriminator/optimizers to12288 steps, no hyperparameter changes;
evaluate every512 to limit diagnostic overhead. This extension is exploratory
and based on quality only. Keep original6144 results separately. Stress tests,
if undertaken, branch from checkpoint12288 rather than pretending3072 qualified.
The runtime thread note above was a hypothesis; actual metadata currently says
4 Torch threads. Verify effective runtime threads in an independent benchmark
and report what was measured, not the unverified assumption about threadpoolctl.

## Exploratory branch audit, NOT a passed quality gate
After extending single-digit training to12288, coverage=10, effective_modes9.61,
valid_fraction=.6292. The .70 predeclared gate still FAILS. Do not relax it or
call this a successful confirmatory collapse experiment. For diagnostic value,
continue four paired branches from this exact checkpoint to16384: unchanged,
G-fast,D-slow,frozen-G. The resulting observations are exploratory and conditioned
on the classifier-accepted subset. All data retained. No five-seed confirmation
will be claimed from this seed0 pilot. Analyze MG only after these reference
branches complete. If evidence remains insufficient, prepare the negative pilot
report and a reproducible GPU bundle rather than claim an experiment on CIFAR.
