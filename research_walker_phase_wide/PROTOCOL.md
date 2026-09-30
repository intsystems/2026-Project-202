# Broad-band phase tracking: final bounded repair

This third experiment is motivated by the FAILED phase pilot: median post-step tracking error on its control final validation records is ~38 whereas sigma2=.1168 made the bounded penalty~.98, providing little separation between ordinary visited states. This correction is chosen before any MG of this new experiment. Preserve both preceding failures. No further reward/architecture changes in this task after this final repair.

Copy all physical/reset/phase/PPO/measurement rules from research_walker_phase/PROTOCOL.md EXCEPT:
1. Reward bandwidth sigma2 becomes median of the five MEAN squared tracking errors of seed260 lambda0 final validation73001..73005. Compute once, freeze in bandwidth.json, not from MG or new held-out outcomes. Both arms use same file; lambda0 unaffected.
2. Pilot seed270,lambda0 vs3; validation75001..75005; held-out76001..76010. If pilot gate passes run five pairs271..275 on new held-out resets. Same1048576-step budget. Else evaluate both failed pilot arms once on held-out states and stop. No replacements, extra budget or alternative bandwidths.
3. Baseline weights/norm/empty optimizer copied byte-for-byte from phase anchor; no newly optimized initialization. All other parameters unchanged, including continuous external phase, reference initialization, original reward, health termination and frozen normalization.
4. D_strobe baseline>=.001, R baseline>=.02, final validation >=4 healthy/common records, both ratios<=.8 and padded return>=.9control for expansion. Strong held-out event requires both<=.75,>=8common and preservation gates exactly as original phase protocol. Report all exploratory pilots as pilots, not independent replications.

Known limitations remain: externally phase-conditioned task, local starts, approximate reference, no proof of active dimension or convergence. Error reduction itself is not independent evidence for periodicity; full-state R,D and physical perturbations are required. MG fixedE20/W2048/tau8 primary, left and tau4/16 and known14cycle window secondary; all retained.
