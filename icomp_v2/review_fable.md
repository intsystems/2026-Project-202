# Review of *Counting the Active Degrees of Freedom of a Training Run*

Reviewer's note on how to use this document. It is split into nine sections that are meant to be
worked on **independently and in parallel**. Each states its own scope, so two people editing
sections 3 and 7 will not collide. Cross-references between sections are given by number where a
fix in one place forces a fix in another.

Everything below refers to `icomp_v2/report.tex` at commit `f9fe0ff` and to the code and result
files under `code/`. Section numbers are as compiled (§5.3 is `sec:digits`, §7.3 is
`sec:grok-diagnostics`, Table 3 is `tab:ladder`, and so on).

Every numerical claim in the paper was re-derived from the committed result files, and every
algorithmic claim traced to the implementation that produced the published numbers. Sections 3, 4
and 8 each end with a **"Verified correct"** subsection listing what passed, so nobody re-checks
ground that has already been covered. Where a number could not be reproduced I say what recipes were
tried; where a result file does not exist I say that too, since a missing file is itself a finding.

**Verdict in one paragraph.** The paper has a real thesis, an unusually disciplined protocol, and a
genuinely interesting negative result, and it is currently held back by three things: the story is
told in the wrong order for the claim it wants to make, a number of load-bearing figures do not
survive contact with the data that produced them, and the body has one figure in nine pages while
the best result sits on page 19. The presentation problems are all fixable in a fortnight. The
numerical ones are not all cosmetic: the great majority of the paper's tables reproduce cell by cell
from the committed results — which is more than most submissions could say — but §7.1's headline
collapse numbers do not reproduce under any definition tried, and the committed data is materially
less favourable than the published range. **Fix section 4.1 first.** Everything else in this review
can wait behind it.

---

## What has been fixed so far

This file is the **live** review: resolved items are deleted from it as they are applied to
`report.tex`. The complete original is preserved in `review_fable_archive.md`.

**Installment 1 — attributions, the Theiler cap, and the protocol column** (applied).
Removed from this file: 2.3, 3.1–3.6, 8.1–8.5, 8.8, 8.9, and the Table 1 transient range.
What changed in the paper: the transformer setting is now attributed to Nanda et al. with the
mini-batch, $S_5$ and weight-decay departures stated; Doshi et al. is no longer called
unregularised; Nanda and Varma are described as dynamical rather than end-of-training accounts;
the Theiler 1986 bias direction is corrected and the Stark critique rewritten around the
fibre-wise embedding; Appendix L's cap condition is inverted to what the code does and the
"uncapped" claim in Appendix B is replaced by the truncation that actually occurs on the training
logs; the window and stride override on the grokking logs is disclosed in §7.3; Table 3's
protocol column now names the requirements each row fails, with an appendix note on the two rows
that predate the canonical estimator; the unverifiable $(m-1)/(m-2)$ and $0.003$ claims are gone,
as is "within half a component up to ten directions", replaced by the per-rank errors that
foreshadow the ceiling. Bibliography: five entries corrected, five added
(Eckmann–Ruelle, Casdagli et al., Clarkson–Woodruff, Roy–Vetterli, Merrill et al.), nine leftover
CCM entries deleted, six previously uncited entries brought into the text.

**Installment 2 — the numbers** (applied). Removed from this file: 3.7–3.12, most of 3.13, 4.1,
4.2, 4.5, 4.10 and 4.11. §7.1's collapse is now reported as the committed data has it — a factor
of $5.2$ to $12.2$, median $7.0$ — with the controls given both ways, $1.6$ and $1.3$ in the
aligned window and $3.2$ and $7.1$ over their whole budget, and the honest conclusion that the
groups are separated by timing and shape rather than depth. A new appendix table gives every run.
§7.2's peak values, norm ratios and "falls by more than half" are corrected. Table 7's
twenty-direction column is regenerated from `scores_frozen.csv` under the recipe that reproduces
its neighbour column. Table 12 no longer prints `n/a` for values that were computed. The
Appendix I memorisation step is $1\,800$, not $2\,250$, and its fall is a delay from memorisation
rather than an absolute step. Four missing runs are added to the inventory. Appendix A's
window-dropping convention now says what the code does, and the degeneracy tally states its
counting rule. A note in Appendix C reconciles $0.87$, $0.47$ and $0.457$ as three aggregations of
one measurement.

Also fixed in the repository: `code/active_rank/dip.py` now defaults to `results_fine` and reports
the statistics the paper uses, so it regenerates the committed `rank_dip.csv` bit for bit; it also
writes the aligned-window control measurement the paper now quotes.

**Installment 3 — structure, compression and figures** (applied). Removed from this file: 1.3,
1.4, 1.5, 1.7, 1.9, 1.11, 2.1, 2.2, 4.3, 4.6, 4.7, 4.10, 4.11, 5.1, 5.2, 5.5 and all of section 6.
Section 5 is split into *the ceiling* and *the silence control*; section 7 is reordered so the
diagnostics come first and the direct measurement second, with a roadmap sentence; the estimand is
defined at a resolution, which turns section 6.4 into a confirmation; the twelve observers are
defined in a new appendix; the protocol requirements are named; Table 3 and Figure 3 move into the
body; related work is compressed from 68 lines to 44; the introduction gains a motivation
paragraph; the conclusion becomes *what this does not establish* plus *what we would do next*; and
the three figure captions are replaced against regenerated figures that show uncertainty and
multiplicity.

**The body is back inside nine pages**, with no undefined references, no citation warnings and no
overfull boxes.

**Installment 4 — the missing experiment, the tables, the prose** (applied). Removed from this
file: 1.6, 1.8, 2.4, 2.5, 2.6, the rest of 3.13, 4.4, 4.8, 4.9, 5.3, 5.4 and 7.4. The
sliding-window estimate was run on the 281 committed windows and returns a resolution limit: the
frozen window spans $39\,990$ optimiser steps, one of twenty-seven centres falls near a
generalisation step, and the largest excursion in the set is in a run that never generalises. That
is now a new appendix and figure, and §7.1 states the claim as untested at this resolution instead
of leaving it hanging. Figure 2 gains an uncertainty band, so every figure now has one. Sections 3
and 6 are rewritten in the plainer register: the functional dimension is demoted to the bound it
is, the terminology is made consistent, the degeneracy flag is described as the conditioning guard
it is, and §6.2 now states that $
ho_{\mathrm{ident}}$ doubles the delay span along with
$E_{\max}$, so a large value cannot be told from a mismatched lag. Eighteen tables were audited;
the observer table's roughness column is now a flag, correlations are signed and two-decimal
throughout, three derivable columns are gone, and the anisotropy decay factor is renamed $q$.

---

## Contents

Section numbering follows the original review, so gaps are items that have been applied and
deleted. Section 6 is gone entirely; sections 2, 3 and 4 retain only their "verified correct"
records, which exist so nobody re-checks ground already covered.

1. [The story](#1-the-story) — one open item: is the admissible regime ever realised?
2. [Definitions, theory and formulas](#2-definitions-theory-and-formulas) — closed
3. [Claims that the code and data do not support](#3-claims-that-the-code-and-data-do-not-support) — closed
4. [The grokking section](#4-the-grokking-section) — closed
5. [Figures and tables](#5-figures-and-tables) — figures that could still be added
6. [Writing](#7-writing) — the title
7. [Citations and related work](#8-citations-and-related-work) — optional additions
8. [Reproducibility and the artifact](#9-reproducibility-and-the-artifact)

Appendix: [what remains](#appendix-priority-list).

---

## 1. The story

*Scope: the thesis, the arc, the contributions list, what to lead with, what is missing from the
argument. Owner: whoever rewrites the spine. Nothing in this section is a line edit.*

### 1.10 The unanswered question: is the admissible regime ever realised in real training?

This is the criticism I would expect from the strongest referee, and the paper currently raises it
and walks away.

Every one of the six validation systems is **externally driven**: a quasiperiodic drive with $r$
independent phases is imposed in order to manufacture a torus. Real training has no such drive. So
what the ladder establishes is conditional — *if* a run were quasiperiodically driven, a scalar log
would count the phases. The paper then examines two real settings and finds neither is. The
question a reader is left holding is whether the admissible regime is ever occupied by anything
anyone trains, and if it is not, the instrument is a well-characterised tool with an empty domain.

The paper knows this. §2 says: *"whether the oscillations of full-batch descent at the edge of
stability \citep{cohen2021gradient} amount to the recurrence the first regime needs is a question
we raise rather than settle."* That sentence identifies the single most valuable remaining
experiment in the project and then declines it.

I would push hard here, because the cost is low and the payoff is the difference between "a careful
negative result" and "a validated instrument plus its first real application":

- You already have a full-batch deterministic setting (the quadratic perceptron). Run it above the
  stability threshold, $\eta > 2/\lambda_{\max}$, where the literature says the loss oscillates
  non-monotonically on short scales while descending on long ones. That is a *deterministic*
  system, so the second condition of §3.3 (an invariant set exists) is at least arguable, and the
  oscillation supplies the recurrence the neighbour statistic needs.
- Then apply the diagnostics. If a real, undriven training run lands inside the admissible
  rectangle, the paper has its headline: the regime is not hypothetical, here is a run in it, and
  here is what the log says. If it does not, you have converted a raised question into a measured
  negative, and §8's limitation becomes "we tested the one candidate regime the literature offers
  and it fails, for this measured reason".

Either outcome is worth more than the current sentence. If neither is feasible before the deadline,
the framing must at least be honest and explicit in §8: the admissible regime has not been
exhibited in an undriven training run, and whether it occurs is open.

## 2. Definitions, theory and formulas

*Nothing open. The estimand is now defined at a resolution, the taxonomy names the three
quantities that matter plus the bound that is only reported, the $E > 2d$ violation at twenty
directions is stated in §5.2, the $(m-1)/(m-2)$ claim is gone, and §6.2 records the confound
between a large identifiability ratio and a mismatched lag. See `review_fable_archive.md` for what
these items said.*

## 3. Claims that the code and data do not support

*Scope: §§4–6 and Appendices A–E, L, audited against the implementation and the committed result
files. Owner: whoever holds the experiments. These are correctness issues, not presentation; several
must be fixed before submission whatever else changes.*

Two independent audits were run: one tracing the estimator implementation, one re-deriving numbers
from the result files. The canonical estimator is
`code/active_dimension/mg.py::all_estimators`, importing neighbour and Theiler machinery from
`code/grokking_analysis/edm/`. **Three rows of Table 3 were produced by a different, older
estimator** (`code/dimension_recovery/estimators.py`) with different defaults; see 3.4.

### 3.14 Verified correct

This is a long list and it is the most reassuring thing in this review: **every table in §§5–6 and
Appendices D, E and L reproduces cell by cell from the committed result files**, with the single
exception of Table 7's twenty-direction column (3.9). Specifically verified: all eight rows of
Table 3; every cell of Table 5, including the degeneracy claim (probe accuracy at $1.0$, every other
observer at exactly $0.0$ in every arm); the eight-direction column of Table 7; all of Table 6,
including the across-seed spread row; all thirty cells of Table 9; all seven factors and all seven
percentages of Table 10; the whole change-detection experiment (true fall $3.994$, observed $3.954$,
stochastic $0.151$, all three level triples, level errors $0.895$ and $0.222$, lags $999$ and
$1999$ in essentially every cell, recovery to $0.0088$, detection by all eleven usable observers in
both directions); all thirty-six cells of Table 11 plus the closed-form agreement to
$1.1\times10^{-4}$ and the $1.02$-against-$1.67$ verdict; all forty-six cells of Table 10's ground
truth, plus the functional rank of $10$ everywhere with participation ratio in $[8.40, 9.03]$ and
the drive condition numbers $1.000$ and $13.274$; all twenty-one cells of Table 17 and both median
errors. The $0.87$, $0.47$, $0.92$ and perfect rank correlations of §5.3 all reproduce, and the
four calibration observers are confirmed as `c_norm, c_proj1, g_fro, w_fro`, so "out-of-sample in
observer for six of the ten" is correct. The stochastic-regime numbers of §6.1 reproduce, including
the flatness in rank ($15.06$–$15.17$) and the tenfold error increase under weak added noise with
Spearman $0.964$. The LB/MG bias figures were located in `dimension_recovery/report_0808.md` §3.1
and match — from the $k \le 6$ experiment, which is the provenance problem of 2.3, not an arithmetic
one.

On the estimator itself: equations (3) and (4) match the
implementation exactly, including the $m-1$ ratio terms, the use of $r_m$, LB as the mean of local
estimates and MG as the pooled form with the $-1$ finite-sample correction; the floors
($10^{-8}$, $10^{-5}$), the dither ($10^{-9}$) and the 1 % degeneracy threshold; the Theiler rule
and cap constants; the KD-tree neighbour search (candidate budget, stable ordering, feasibility
guard) and the delay-embedding indices, with no off-by-one; roughness, $\PR_{\mathrm{delay}}$ and
the spectral participation ratio, including the DC drop and the 256/1024/native binning; the
participation-ratio formula throughout; the trend-crossing count in both copies; TwoNN; the frozen
configuration files against Table 2, including grid sizes, selection seeds and ranks, and the
$0.324$–$1.499$ selection-error range; the claim that the Theiler axis is inert on the calibration
data (configurations 45 and 47 return bit-identical scores); the absence of clamping in the
published grokking numbers ($28.08 > E_{\max} = 20$ reported as such); every cell of Table 12
reproducing from the committed CSVs; the label-matched separation with no overlap; the cap-lift in
the Appendix-L re-scoring being correctly applied inside each worker process; and the construction
code — drive whitening by $\mathrm{pinv}(\Phi)Q$, per-mode gain division, the $\eta = 0$ control
retaining the drive, ground truth measured as the trajectory covariance participation ratio, and
incommensurate frequencies in a fixed octave.

---

## 4. The grokking section

*Scope: §7 and Appendices F–K. Owner: whoever holds the application. Read §1.3 and §1.6 first —
they are about the same section and are not repeated here.*

### 4.12 Verified correct

The damage above is concentrated in §7.1, §7.2 and Appendix I. Everything else in the application
section holds up, and much of it holds up exactly.

**Table 12**: all thirty-five printed numbers reproduce from the committed summaries.
**§7.3's regime signatures**: the identifiability band $1.29$–$1.60$ (over the five regularised runs
— see 3.13), and for the full-batch setting a ratio of unity ($0.9975$–$0.9997$), roughness of one
part in a thousand ($0.0004$–$0.0010$), two trend crossings everywhere, and a delay participation
ratio of $1.0000$–$1.0001$ across all nineteen rows.
**§6.4's degeneracy count**: exactly $7 \times 5 = 35$ cells, of which eleven have a majority of
windows flagged (ten at $100\,\%$, fourteen with at least one).
**The label-matched pairs**: separations $+3.16$, $+2.98$, $+3.54$, $+2.88$, no overlap.
**Appendix G**: parameter counts $226\,816$, $228\,608$ and $145\,500$; every cell of Table 13
against `verify_sketch.log`; the exact-versus-sketched agreement to three decimals on the raw
$145\,500$ parameters; the non-invasiveness check at zero maximum absolute difference over 400 steps;
and the window spans of 590 and 295 logged steps. Two small things: the two-hash disagreement figures
come from different columns — $8.4\,\%$ is the worst on `fn_PR_pos_det`, while the $1.1\,\%$ median is
`PR_pos_det` alone ($0.94\,\%$ pooled over the two published columns, $0.68\,\%$ over all eight) — and
the report says forty logged rows where the log says forty-one.
**Appendix H**: everything, as described in 4.9.
**Appendix J**: every checkable number — the mode count, the analytic order parameter and its
$0.0407$ initialisation, the $148.84$ analytic effective rank on a $500 \times 194$ matrix, the
$139.06$ at initialisation and the $132.6$–$150.9$ span, the whole four-point order-parameter trace,
and every row of Table 15 including the two non-trivial own-references ($0.052$ and $0.062$) and
`g_p3x`'s $73.1\,\%$ against a $1.34\,\%$ majority baseline.
**Table 16**: every row except `p211_wd0`'s memorisation step, including all six sketch rows, all
seven extended rows, all nine perceptron rows, the $S_5$ weight decay of $0.2$, the double
`optimizer.step()` on exactly the two $S_5$ runs, the logging strides, and the batch, learning rate,
betas, validation subset size and double precision.

Two provenance problems for the artifact, which belong in section 9 as well: `active_rank/dip.py`
does not regenerate the `rank_dip.csv` that §7.1 depends on (4.1), and
`code/grokking_train/grokking_logs/` is empty and git-ignored, so Appendix K's pointer to it is
misleading — the logs actually used live in `code/dimension_recovery/results/extended/` and
`code/Grokking/`.

---

## 5. Figures and tables

*Scope: the four figures, their captions, the sixteen tables, and what should be added. Owner:
whoever holds the plots. Findings are from a dedicated audit against `make_figures.py` and the
source CSVs.*

### 5.6 Figures — what was added, and what is still missing

Applied. The paper now has nine figures, four of them added in this pass: the anisotropy
separation (appendix E), the delay-lag sweep (appendix E), per-observer error with across-seed
spread (appendix D), and the participation ratio against window length (appendix J). Each sits
beside the table it visualises, so the table keeps the exact values and the figure carries the
shape. The palette was replaced — the old one separated by only 8.8 ΔE under simulated
deuteranopia despite the build notes claiming otherwise — and every explanatory sentence moved
from inside the axes into the captions.

Still not drawn, and both need computation rather than plotting:

- **The $E_{\max}$ ceiling.** The figure that would settle §5.2's open question needs the sweep
  described there; there is nothing to plot until it is run.
- **Raw scalar-log traces, one per regime.** For the synthetic arms only windowed summaries are
  committed, so the raw series would have to be regenerated. The real grokking logs do exist
  (`active_rank/results_fine/*_train.csv`), and the accuracy, loss and norm traces for all four
  transformer runs were plotted in the previous draft
  (`icomp_article/images/{mod,s5}_wd{0,1}_{acc,loss,norm}.png`), so the real half of this figure
  remains a restyling job. It is the one figure a reader of a paper about reading time series
  might still miss.

## 7. Writing

*Scope: title, abstract, headings, sentences, terminology, numbers in prose. Owner: the copy pass.
Per your instruction the register moves toward standard A\*-venue prose: shorter declarative
sentences, explicit signposting, numbers in tables. The precision and the hedged claims stay.*

### 7.1 The title

Nineteen words, a colon, and manual line breaks. The right-hand side ("What a Single Scalar Log Can
and Cannot Measure") is the honest promise of the paper and should survive; the left-hand side is
long and abstract. Options:

- *Counting the Active Dimension of a Training Run: What a Single Scalar Log Can and Cannot Measure*
- *When a Training Log Counts: the Active Dimension of an Optimiser from One Scalar Series*
- *The Active Dimension of a Training Window, and What a Scalar Log Can Measure of It*

Remove the `\\` breaks from the canonical title; they leak into metadata and other renderings.

## 8. Citations and related work

*Scope: `references.bib` and every citation use, audited against the sources. Owner: the references
pass. This section is longer than expected because the audit found four attribution errors in the
experimental description, one of which changes what the paper is claiming to reproduce.*

### 8.6 The Takens statement is correct — my §2.2 concern was unfounded

For the record, since section 2 of this review raised it: $E > 2d$ is an exact restatement of Takens
for an integer $d$, and Sauer et al.'s prevalence theorem is *literally stated* as $n > 2d$ with $d$
the box-counting dimension, plus the periodic-orbit conditions. The shared phrasing is correct for
both forms and the sentence does not conflate them. Only two small things remain: Sauer gives
one-to-one on $A$ plus immersion on smooth submanifolds rather than "embedding" in the differentiable
sense on a fractal (an optional footnote), and the twenty-direction configuration sits at $E = 2d$
exactly, which is one short of the condition the paper itself states — see §1.5 and §2.2 of this
review, which stand.

### 8.7 Works that should be cited and are not

High value, in order:

1. **`nanda2023progress` for the experimental setting** (8.1). Already in the bibliography; free to fix.
2. **Eckmann & Ruelle 1992**, *Fundamental limitations for estimating dimensions and Lyapunov
   exponents in dynamical systems*, Physica D 56:185, and/or **Ruelle 1990** with the
   $D \lesssim 2\log_{10} N$ rule. This is the canonical statement that a finite record bounds the
   measurable dimension — which is the paper's own repeated claim, in its own words ("set by
   $E_{\max}$, the correlation time and the record length"). Its absence is the single largest gap
   in §2, and it bears directly on the ceiling story: at $N = 8000$ the Ruelle bound gives $7.8$,
   which is an alternative explanation for the eight-direction ceiling that the $E_{\max}/2$
   hypothesis of §1.5 must be tested against. See the revised §1.5.
3. **`notsawo2023predicting`, for its trajectory-PCA result.** They report that **more than 98 % of
   the optimisation trajectory's variance lies in the first two PCA components** on grokking runs.
   That is a trajectory effective-rank measurement on grokking runs — the same quantity as §7.1 —
   and the paper currently cites Notsawo only for early-curve prediction while claiming novelty for
   the trajectory measurement. This must be engaged with directly; see §4.9 of this review.
4. **Clarkson & Woodruff 2013**, or Nelson & Nguyễn 2013 / Meng & Mahoney 2013. CountSketch *is* an
   oblivious subspace embedding, which is a spectral guarantee. Appendix G's disclaimer ("We claim
   no spectral guarantee for this") is stronger than the literature warrants; your windows are
   60 samples and therefore low rank, where the relevant bound is $O(k^2/\varepsilon^2)$ rows.
   Cite the subspace-embedding line before declining it — the honest position is probably "a
   guarantee exists but at a rank we do not verify", not "no guarantee exists".
5. **Merrill, Tsilivis & Shukla 2023**, *A Tale of Two Circuits* — measures sparsity and rank
   structure across the grokking transition; the nearest prior work to §7.1.
6. **Roy & Vetterli 2007**, *The effective rank*. You use "effective rank" throughout for the
   participation ratio, but that term already denotes $\exp(\text{spectral entropy})$ in the
   literature. Either cite and disambiguate in one sentence, or rename. Given how central the
   quantity is, I would disambiguate.
7. **Casdagli et al. 1991**, *State space reconstruction in the presence of noise* — the standard
   treatment of reconstruction under noise and of the embedding window $(E-1)\tau$, which is §6.2's
   entire subject.

Already in your bibliography and worth promoting to cited: `kantz2004nonlinear` (the standard
reference for the Theiler window and embedding practice), `sagun2018empirical` (Hessian spectrum,
supports the Gur-Ari claim), `aghajanyan2021intrinsic` (belongs beside `li2018measuring`),
`thilak2022slingshot` (the slingshot mechanism produces the oscillations Notsawo exploits and is the
obvious alternative explanation for the AdamW roughness and trend-crossing signatures in Table 12),
`barak2022hidden` (belongs in the progress-measures list), `camastra2016intrinsic` (supports "the
standard alternatives").

Optional: Theiler 1991 on the correlation dimension of $1/f^\alpha$ noise, which qualifies
`osborne1989finite`; Sauer & Yorke 1993, *How many delay coordinates do you need?*; and Provenzale
et al. 1992's space–time separation plot, which is the principled way to set $W_T$ and is directly
relevant to Appendix L.

### 8.10 Verified correct

Takens, Sauer et al., Levina–Bickel, MacKay–Ghahramani (substance), Osborne & Provenzale, the
Theiler citation in §3.2, Theiler 1992 with Schreiber–Schmitz, Li et al., Gur-Ari et al. (modulo
8.8), Gao et al., McCandlish et al., Mandt et al., Cohen et al., Nanda's cleanup phase as
paraphrased in §7.1, Varma's efficiency argument, Notsawo's Hjorth features ("spectral features of
its early learning curve" is exact), the Gromov/Doshi division of labour including the mode count
$(p+1)/2 = 49$ at $p = 97$ and the $500 \times 194$ first layer with 145 500 parameters, the
CountSketch/feature-hashing co-citation, and the venues, years, volumes and pages of everything not
listed in 8.9.

---

## 9. Reproducibility and the artifact

*Scope: what a reader would need to reproduce this, and what the code audit found about the state of
the repository. Owner: whoever prepares the release. Short section, mostly actionable.*

0. **A committed analysis script does not regenerate the committed file the paper depends on.**
   `active_rank/dip.py` computes `fn_PR_step5, fn_PR_step20, PR_step20, PR_pos_det`; the
   `results_fine/rank_dip.csv` that §7.1 is built on contains `fn_PR_pos_det` and `PR_step5`. This
   is almost certainly why §7.1's numbers cannot be reproduced (4.1), and it is the first thing to
   repair — not for the paper's sake but so that the next person to touch these numbers can check
   them.
1. **Three tables' worth of results come from a different estimator than Algorithm 1** (3.4). If the
   experiments are not re-run, the paper must say which rows used which implementation, and the
   repository should carry a single canonical estimator with the older one deleted or clearly marked
   superseded. Right now `dimension_recovery/estimators.py` and `active_dimension/mg.py` differ in
   their defaults for Theiler exclusion and clamping — the two settings the paper argues most about.
2. **The observers are undefined in the paper** (3.13). One appendix table, one formula per observer.
3. **The pipeline overrides the frozen configuration in three places** without the paper saying so
   (3.3). Either make the frozen configuration authoritative in code, or document the overrides in
   Appendix B beside the configuration they modify.
4. **No compute cost is reported.** Total GPU or CPU hours, and the cost of the trajectory sketch
   relative to training, belong in an appendix; the sketch's cheapness is one of the paper's
   practical selling points and is never quantified.
5. **No code-availability statement.** For an anonymous submission, a sentence promising release on
   acceptance, naming the modules, is standard and costs one line.
6. **`icomp_v2/README.md` documents two known defects in upstream result files** (the
   `invariance_controls.csv` deltas that are really seed scatter, and the never-written-up $k = 20$
   calibration). Those fixes exist only in the paper. Push them back into the source repository, or
   the next person to read `active_dimension/results/` will re-derive the wrong numbers.
7. **Appendix K points at an empty directory.** `code/grokking_train/grokking_logs/` is git-ignored
   and empty; the logs the paper actually uses are in `code/dimension_recovery/results/extended/` and
   `code/Grokking/`. Fix the pointer or ship the logs.
8. **`OmnigrokTransformer` is Nanda's architecture** (8.1). Rename the class, or the misattribution
   propagates into the next paper from this repository.
9. **Table 7's twenty-direction column is not regenerable** from any recipe that reproduces its
   neighbour column (3.9). Commit the script that produced it.
10. The `X.log` and `texput.log` files at the repository root, and the untracked build artifacts in
    `icomp_v2/`, should not be in a release. `exp7_inventory.csv` records a logging stride of 49 for
    the $p = 211$ run where the log has 50 — harmless, but it is a file a reviewer would open.

---

## Appendix: priority list

Four items remain, and two of them need compute rather than editing.

1. **The edge-of-stability run.** The conclusion names it as future work, which is honest but
   weaker than doing it. It is the one experiment that could show the admissible regime is ever
   occupied by an undriven training run, and until it is run the instrument has no demonstrated
   domain outside constructed systems. *(1.10)*
2. ~~**Re-run the log estimate at a window matched to the transition.**~~ **Done** in commit
   `b5a4a7c`, which added `active_dimension/e9_matched_window.py`, `e9_analyse.py` and
   `e10_surrogate.py`, and rewrote section 7 around the result: at a $600$-step window the estimate
   falls at the generalisation step in all four generalising runs and in neither control, and
   survives a surrogate control on the observer's shape. This list was not updated at the time, so
   the item sat here stale for two commits.
3. **The $E_{\max}$ and record-length sweeps**, which would separate the two explanations of the
   ceiling that §5.2 currently leaves open.
4. **The title.** Nineteen words and a colon; three alternatives are in the archived review. This
   is the last purely editorial item. *(7.1)*

Still worth adding if space is ever found: the anisotropy plot, the $\tau$-sweep plot, and raw
scalar-log traces per regime *(5.6)*.
