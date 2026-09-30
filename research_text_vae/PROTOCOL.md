# Text VAE: latent-information loss during NLP learning

Written before pilot, 2026-09-29. Separate direction from CV / learning-rate
changes. Goal: test scalar MG against independently established loss of sentence
information in a learned latent representation. This is NOT a claim that the
whole optimizer or autoregressive decoder becomes low dimensional.

Motivation: Bowman et al., Generating Sentences from a Continuous Space (CoNLL
2016, K16-1002), especially section3.1. He et al., Lagging Inference Networks and
Posterior Collapse in Variational Autoencoders (ICLR2019, arXiv1901.05534).
This is a small NLP experiment, not an LLM-scale result or an exact reproduction.

Data: preprocessed Penn Treebank word-level train/valid from tomsercu/lstm/data,
record SHA256. Complete lines8..24 words; no artificial templates. Random fixed
subset6000 train sentences,256 validation sentences; vocabulary top1996 words
from training split plus PAD/BOS/EOS/UNK (2000 maximum). Never fit vocabulary to
validation. Dataset selection seed20260929. Save exact selected indices and vocab.

Encoder GRU64, embedding48, Gaussian latent16; decoder GRU64, same embedding,
latent concatenated at each decoder step and projected to initial state.
Teacher forcing, no token dropout. Adam lr.001, clip gradient norm5, batch32.
Loss = reconstruction NLL summed over valid tokens per sentence + beta *
KL(q(z|x)||N(0,I)), then average sentences. All model parameters train.

Pilot seed0: shared1024 updates beta.01, then two continuations up to3072 updates:
base beta.01 vs regularized beta1. Same weights/Adam state/batches/latent noise
at branch. No parameter pruning or manually removing latent coordinates.
Changing beta is an externally scheduled regularization intervention, not proof
of collapse by itself. This tests annealing toward the ordinary VAE objective.

Primary MG input: mean per-token RECONSTRUCTION NLL on the same16 validation
sentences before every optimizer update. Fixed Gaussian epsilon seed829, so the
only evolving source is the model. No beta, KL, MI or other reference enters MG.
No trend removal or smoothing. Window512,stride128,E20,tau1,k20,Theiler39;
E40 at the same exclusion, floor flags. SensitivityW256/1024,tau4 always shown,
never selected to replace the primary result. Pre windows end512..1024 step128;
late windows end2048..3072 step128, all entirely after the intervention.

Independent references every128updates on256 fixed validation sentences:
1. Analytic average KL(q(z|x)||prior), nats/sentence. Cheap baseline, not a
   unique proof of useful information when positive.
2. Monte Carlo I_q(X;Z) on the empirical uniform256-sentence mixture: four
   fixed independent Gaussian draws per sentence, explicit all-pairs logq(z|x).
   Reference timing includes encoding. It is a finite-mixture MC diagnostic,
   not the unknown population MI. Keep any small negative MC estimates.
3. Shuffle encoder distributions across sentences, keeping epsilon matched;
   compare teacher-forced token distributions and NLL with the matched code.
   Report mean symmetric KL of predictions and NLL disadvantage of shuffling.
   If both information and dependence vanish, latent information is unused;
   this is not a claim about autonomous dynamical recurrence.
4. Active-unit variance threshold.01 as context only, not ground truth dimension.

Independent event acceptance: MI and prediction-change response must both fall
by at least50% relative to base, with nontrivial starting values (MI>.1 nats,
prediction symmetric KL>1e-4 nats/token). Otherwise report no established event.
MG success requires a stronger sustained decline than the unchanged arm, not a
dip in a window spanning the switch. All raw traces and reference failures kept.
One seed is only a pilot: if promising, confirm on new seeds with fixed settings.
If it fails, report the failure rather than relabeling another channel as primary.

Potential practical benefit is a cheap generic alert triggering occasional
expensive audits. However KL and cheap loss statistics are strong fast baselines.
Do not claim improvement over them without measuring detection and false alarms.
Do not promise MG will detect posterior collapse: that is the hypothesis tested.

## Confirmation frozen after seed0, before seeds1 and2
The pilot independent event passes both criteria: late/early MI ratio about.048
versus1.012 in base; prediction-shuffle response ratio.026 versus3.516 in base.
Primary MG late/early=.581 versus1.193 in base; paired ratio=.487. E20,tau1
works for W256/512/1024, but tau4 almost removes the contrast. This sensitivity
is retained, not tuned away. Standard deviation also separates the branches.
Run two new independent initialization/batch seeds1 and2 with identical settings,
same data/probes. No failed seed removed. Primary remains fixed-probe NLL,W512.
Additionally score actual minibatch NLL at W512 as an exploratory channel;
report it separately, never substitute it for the primary if it looks better.
Amplitude-scale and IAAFT controls are descriptive, not proofs of validity.
No detection-latency threshold is chosen on confirmation seeds.

## Expansion frozen on 2026-09-29, before seeds3 through9
User requested a larger series after inspecting the first three runs.
Add exactly seeds3,4,5,6,7,8,9 without changing model, data, observation,
optimizer, intervention, windows, metrics or acceptance criteria. Ten total
initializations: pilot0, initial confirmation1-2, expansion3-9. Never remove a
negative outcome or replace it with a different seed. Runtime failures are
reported and rerun with the same configuration, not treated as negative data.
Primary aggregate excludes pilot0 and uses confirmation1-9; also separately
report the seven newly requested seeds. Summaries operate on paired seeds,
never treating overlapping windows as independent observations. Show all ten
rows, median/range, bootstrap interval for the confirmation median paired MG
ratio (20000 resamples, RNG20260929), and window/delay sensitivity for every seed.
The interval concerns variation across initialization/batch seeds at fixed data
and probe, not transfer to other datasets or models. Workers may run in parallel;
their timings must not be used to claim a speed advantage.
