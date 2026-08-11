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

**Known regression:** the body is now ten pages against a nine-page limit. Installment 3 is the
restructuring pass that pays it back; installment 2 is net-negative on length.

---

## Contents

1. [The story: what this paper is about and in what order to say it](#1-the-story)
2. [Definitions, theory and formulas](#2-definitions-theory-and-formulas)
3. [Claims that the code and data do not support](#3-claims-that-the-code-and-data-do-not-support)
4. [The grokking section: evidence, logic and what is missing](#4-the-grokking-section)
5. [Figures and tables](#5-figures-and-tables)
6. [Structure, placement and the page budget](#6-structure-placement-and-the-page-budget)
7. [Writing: title, abstract, headings, sentences](#7-writing)
8. [Citations and related work](#8-citations-and-related-work)
9. [Reproducibility and the artifact](#9-reproducibility-and-the-artifact)

Appendix: [the priority list](#appendix-priority-list), if you only fix fifteen things.

---

## 1. The story

*Scope: the thesis, the arc, the contributions list, what to lead with, what is missing from the
argument. Owner: whoever rewrites the spine. Nothing in this section is a line edit.*

### 1.1 The thesis is good and the paper does not say it plainly

The best sentence in the draft is in the abstract: *"The estimator is standard [...]; the
contribution is knowing when its output is a count."* That is a real contribution and an unusual
one — a paper about the conditions under which an existing instrument may be read. Everything else
should be arranged to serve it.

The frame I would recommend, stated once, early, and then never re-argued:

> A scalar log is free and a stored trajectory is expensive. The question is what survives the
> projection. Under conditions that can be checked from the series itself, almost nothing is lost
> up to about eight components. In the two settings in which grokking is actually studied,
> everything is lost — and the diagnostics say so before the number is believed. When one does pay
> for the trajectory, there is something to see.

That framing makes the negative result *the point* rather than the disappointment. As written, the
reader meets the negative result as the last in a series of retreats, and a tired reviewer will
summarise the paper as "the method does not work on the thing they wanted to use it for."

### 1.2 The reader is never told why they should want this number

There is no motivation paragraph anywhere. §1 opens with an $r$-torus. The only stated reason to
want the active dimension is that a scalar log is cheap — which is a reason to prefer *this
estimator*, not a reason to want *the quantity*. Before the definition, the paper needs one
paragraph answering: who measures this, and what do they do differently once they know it?
Candidate answers already latent in the draft — pick two, do not list five:

- detecting phase transitions in training cheaply, from logs that already exist;
- deciding whether an apparent simplification is a real reduction in degrees of freedom or an
  artefact of a shrinking scale (the paper's own §6.3 nuisance);
- adjudicating "the optimiser moves in a low-dimensional subspace" claims, which are made in the
  literature with four different quantities that the paper's §3.1 separates.

### 1.3 The arc of §7 fights the thesis

Given the methodology-first framing you have chosen, §7 currently runs backwards. It opens with a
direct trajectory measurement that does **not** use the instrument the previous four sections
built, and the instrument does not reappear until §7.3. A reader who has just spent five pages on
delay embeddings is entitled to ask, at the top of page 8, why they read them.

Two acceptable orders. Either:

- **(A) instrument-first**: §7.1 the diagnostics refuse both settings and say why (this is the
  instrument *working*); §7.2 so we pay for the trajectory, and here is the collapse; §7.3 what a
  log can still contribute, and the label-matched pairs. This matches the title and abstract.
- **(B) result-first**, which is what you have: then you need **one roadmap sentence** at the top
  of §7 that states the whole logic in advance — "the logs of both settings turn out to be
  inadmissible, which we establish in §7.3; we therefore begin with what can be measured when the
  trajectory itself is stored." Without it the section reads as a topic change.

I would take (A). It costs nothing and it is the only order in which the paper's own instrument is
doing work at the moment of the headline result.

### 1.4 The self-caught leakage is an asset, currently filed as an embarrassment

Requirement 4 — set the learning rate to zero and the observer must fall silent — caught two of
your own systems reproducing their reported result with no training taking place, at correlation
above 0.98. It also caught the instantaneous mini-batch loss, an observer that scored in the best
three. That is the most quotable methodological result in the paper: **a control that most
published dimension-on-training-logs analyses do not run, which invalidates two of six of our own
systems.**

Right now it is delivered in a subordinate clause inside a subsection called "Adding an optimiser,
and then a network". Give it a name and a heading of its own ("The silence control", §5.2 or a
short §4.1), state it as a finding, and let the two rejected systems be the evidence rather than
the subject.

### 1.5 The saturation ceiling is a result, not a limitation — and you can probably explain it

This is the most substantive scientific suggestion in this review, and it addresses the concern
you raised directly: the intermediate systems may not be "built wrong", they may be showing a real
property of the instrument.

The draft reports three saturation points and never connects them:

| where | $E_{\max}$ | saturates above about |
| --- | --- | --- |
| §5.1, oscillating matrix extended to twenty | 25 (`exp9`, `best_config.json`) | 9 |
| §5.2, linear and logistic regression to twenty | not stated | 12 |
| §5.3, image data at twenty available directions | 40 | 6, reaching 15.1 at $r=20$ |

Takens' condition as the paper itself states it in §3.2 is $E > 2d$. That gives a **predicted
ceiling of $E_{\max}/2$**: 10 for $E_{\max} = 20$, 20 for $E_{\max} = 40$. The observed ceiling at
the eight-direction configuration is "above about nine", which is $E_{\max}/2$ to within the
resolution of the experiment. And the twenty-direction configuration uses $E_{\max} = 40$ for
$d = 20$, i.e. $E = 2d$ exactly, which **fails** the paper's own sufficient condition — so the
observed 15.1 against a true 20.0 is not a mysterious shortfall, it is what the design predicts.

Your own calibration grid already contains supporting evidence: at $E_{\max} = 10$ the selection
error over ranks $\{2,4,6\}$ is $0.500$, against $0.324$ at $E_{\max} = 20$
(`active_dimension/results/e1_calibration/config_ranking.csv`), and the difference must come from
$r = 6$, which exceeds $E_{\max}/2 = 5$.

Two consequences:

- **Reframe.** Replace the scattered saturation remarks with one named paragraph — "the ceiling" —
  that states the $E_{\max}/2$ prediction, shows it, and thereby converts the paper's most-noticed
  weakness into a confirmed property of the instrument. It also disposes of the awkward "the
  twenty-direction result is exploratory" framing: the honest statement is that $E_{\max} = 40$
  cannot recover twenty components *by construction*, and the experiment confirms it.
- **One cheap experiment closes it, and there is a rival hypothesis it must beat.** The competing
  explanation is the classical finite-record bound: Eckmann and Ruelle's rule of thumb is
  $D \lesssim 2\log_{10} N$, which at your window of $N = 8000$ gives $7.8$ — uncomfortably close to
  the observed eight-direction ceiling. The two hypotheses are cleanly separated by two sweeps:
  vary $E_{\max} \in \{10, 20, 40, 80\}$ at fixed record length, and vary the record length at fixed
  $E_{\max}$. The $E_{\max}/2$ hypothesis predicts movement in the first and none in the second; the
  record-length hypothesis predicts the reverse. Note that the record-length bound does **not**
  explain the twenty-direction ceiling of $\approx 15$ at the same $N = 8000$, while $E_{\max}/2$
  does, so the evidence you already have favours the embedding explanation — which is why this is
  worth running rather than merely conceding. Either outcome is publishable and either beats the
  present silence, and the identifiability ratio already computes a $2E_{\max}$ arm, so half the
  machinery exists. Eckmann and Ruelle must be cited whichever way it comes out; see §8.7.

Per your instruction, keep the intermediate systems in the story but do not leave them mixed in one
subsection. Split §5.2 along the two threads that are currently tangled: **the ceiling** (regression
systems, matrix extension, $k=20$) and **the silence control** (decoder, perceptron, instantaneous
loss). Constructions and per-system tables go to the appendix; the body keeps a paragraph each.

### 1.6 §7.3's constructive conclusion has no supporting experiment

§7.3 ends with the paper's forward-looking claim: the *level* of the estimate is uninterpretable,
but a *change* within a run is read against a fixed configuration, so "a change that is fast,
localised and larger than that floor carries information the level does not."

No experiment in the paper demonstrates such a change in a training log. The only fall actually
detected on a real log is the $p = 211$ run of Appendix I, which is slow, in an inadmissible
regime, and attributed to the growing norm. So the section's positive half is a hypothesis
presented in the grammar of a conclusion.

Two ways out, and the first is much better:

1. **Run it.** `active_dimension/results/e5_real_logs/real_logs_windows.csv` already holds 281
   sliding windows of MG and $\rho_{\mathrm{ident}}$ per run and per column. Plot the windowed
   estimate against step for the regularised runs, aligned on $t_{\mathrm{gen}}$, against the
   controls. Whatever it shows is a result: if the estimate moves at $t_{\mathrm{gen}}$ you have
   closed the loop from a free log to the transition; if it does not, you have the strongest
   possible statement of the paper's thesis, because the direct measurement in the same runs *does*
   move. Either way this is the experiment the paper is set up to run and does not.
2. Reframe §7.3's second half explicitly as what would be required, not as what follows.

### 1.7 There is no "so what should I do" for the reader

Three actionable rules are derivable from the paper and are currently scattered across five
sections. Collect them, in the body, as a short numbered block — reviewers reward this and it costs
eight lines:

- do not read a delay-embedding dimension off a training log without the two diagnostics; both
  standard grokking settings fail them;
- if you want the trajectory rank, a CountSketch to $\mathbb{R}^{1024}$ is enough and the
  validation is in Appendix G — this is a cheap, reusable recipe and the paper undersells it;
- a trajectory participation ratio must be controlled for window length and for batch size before
  anything is attributed to learning (Appendix H proves this on your own data and it is currently
  buried).

### 1.8 The hedging, in aggregate, argues against the paper

Individually every qualification in the draft is correct and I would not remove one of them. But
there are, by my count, more than twenty, and several results are hedged twice in the same
sentence ("which is a match and not a guarantee on an arbitrary manifold"). The cumulative effect
on a reader skimming for a verdict is that nothing was established.

The fix is structural, not a matter of deleting hedges: **collect them**. Give §6 or §8 an explicit
"What this does not establish" block that carries the scope limits in one place, and let the result
sentences be declarative. You lose no honesty and you gain the ability to state a finding without
apologising inside the same clause.

### 1.9 The contributions list overclaims relative to the body

- Contribution 2 says the estimator "orders it to twenty". The body marks the twenty-direction
  result exploratory (no seed split; Theiler window smaller than the embedding span). A
  contribution bullet must not claim what §5.3 disclaims. Either promote the result — which
  §1.5 above gives you a principled way to do — or say "ordinally to twenty in an exploratory
  configuration".
- The abstract says "On six systems [...] recovery is accurate to under one component". Only two
  rows of Table 3 carry a recovery number the protocol accepts, and the headline $0.87$ comes from
  **one** system. The sentence as written implies six-system agreement. Say: evaluated on six
  systems, of which the two that pass every control give $0.87$ and $0.50$.
- "to $0.87$ components" reads as a bound; it is a mean absolute error. Same for "to under one
  component" throughout.
- §8 says "Four limits bound this" and then gives three sentences, the second containing four
  clauses. Either itemise or fix the count.

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

### 1.11 Limitations that are missing

Currently absent from §8 and needed:

- **Scale.** Every system is tiny: at most twenty active directions, an 8×8-pixel dataset, a
  one-layer transformer. Nothing here is tested at a scale anyone trains at. Say so; it is not
  disqualifying and the reviewer will otherwise say it for you.
- **The configuration is a choice.** Appendix B reports the selection error spanning a factor of
  five across the grid on identical data, and both configurations sitting at a grid boundary. That
  belongs in the limitations, not only in an appendix.
- **The diagnostics' calibration set is five arms of one system.** The admissible region in the
  figure is two thresholds fitted on a handful of points, with no held-out validation. The paper
  says "calibrated against no wider family" once, in passing; it should be a stated limit.

---

## 2. Definitions, theory and formulas

*Scope: §3 and the mathematical statements throughout. Owner: whoever holds the theory. Several of
these are cheap fixes that materially raise the paper's rigour.*

### 2.1 CRITICAL — the definition of $d_{\mathrm{act}}$ is not well-posed as written

Equation (1) defines $d_{\mathrm{act}}(\gW) = \dim \gM_{\gW}$ where $\gM_{\gW}$ is "the set of
states the trajectory recurrently visits inside the window, the local part of the invariant set or
the closure of the orbit."

Over a *finite window*, the closure of $\{\vc_t : t \in \gW\}$ is a compact set of dimension one —
a curve segment. The $r$-torus is the closure of the **infinite** orbit, or equivalently the
support of the invariant measure. So the definition, read literally, gives $d_{\mathrm{act}} = 1$
for every $r$, which is exactly the reading the following sentence tries to exclude ("It is not the
dimension of the time-parameterised curve $t \mapsto \vc_t$, which is one for every $r$, but of the
set the curve fills"). A finite orbit segment does not fill a torus. As it stands the definition
and its disclaimer contradict each other, and a referee from dynamical systems will open here.

**The fix also solves a second problem, which is why I would take it.** Define the estimand *at a
resolution*:

$$d_{\mathrm{act}}^{\varepsilon}(\gW) \;=\; \dim_{\varepsilon} \overline{\{\vc_t : t \in \gW\}},$$

the dimension of the $\varepsilon$-neighbourhood of the visited set — equivalently the dimension of
the support of the empirical measure resolved at scale $\varepsilon$ — with the window taken long
compared to the recurrence time so that the empirical measure approximates the invariant one. Then:

- the $r$-torus statement becomes true rather than aspirational, for $\varepsilon$ smaller than the
  smallest mode amplitude and a window covering several recurrence times;
- the Levina–Bickel estimator, which is intrinsically a finite-neighbourhood-scale statistic,
  becomes an estimator **of the thing you defined** rather than of an idealisation it can only
  approach;
- and §6.4's result — "It counts the components it can *resolve*" — stops being a caveat about the
  estimator and becomes a correct measurement of $d_{\mathrm{act}}^{\varepsilon}$. The anisotropy
  experiment then *confirms* the estimator rather than qualifying it, which is a strictly stronger
  claim from the same data.

You should also state the two idealisations explicitly, because the honest version is short: over
the window the dynamics is treated as autonomous (quasi-static), and the window is assumed long
relative to the recurrence time. Both are true by construction in your systems and false on the
grokking logs, which is a second, independent way of saying why those logs are inadmissible.

### 2.2 MAJOR — $E > 2d$ is violated by the twenty-direction configuration

§3.2 states the embedding condition as $E > 2d$. The twenty-direction configuration uses
$E_{\max} = 40$ and targets $d = 20$: $40 > 40$ is false. The design cannot satisfy its own stated
sufficient condition at the top of its range. See §1.5 — this is worth stating as a prediction that
the data confirms, not hiding. Minimum fix: one sentence in §5.3 or Appendix B.

Also, the sentence conflates two theorems in sequence: the manifold form (integer $d$, $E \ge 2d+1$,
equivalently $E > 2d$) and the prevalence form (box-counting $d$, possibly fractional, $E > 2d$).
Written in one breath, the reader cannot tell which $d$ the paper's own $E > 2d$ refers to. Split
into two sentences and say which one licenses your choice of $E$.

### 2.4 MAJOR — $\rho_{\mathrm{ident}}$ conflates dependence on $E$ with dependence on the delay span

$\rho_{\mathrm{ident}} = \hat d_{\mathrm{MG}}(2E_{\max}) / \hat d_{\mathrm{MG}}(E_{\max})$ is
computed with $\tau$ held fixed (confirmed in `runner.py:59`, `e5_real_logs.py:68`). Doubling $E$
at fixed $\tau$ **doubles the delay span** $(E-1)\tau$. §6.2 then demonstrates, at length, that the
estimate is violently span-dependent: on a correct system at $r = 8$, doubling the span from
$0.19$ to $0.38$ of a period moves the estimate from $4.29$ to $8.53$ — a ratio of 1.99.

So on a system that is perfectly admissible but whose lag is mismatched, $\rho_{\mathrm{ident}}$
would read about 2, and the diagnostic would reject it. The interpretation offered in §3.4 — the
estimate "is a property of the embedding space rather than of the data" — is only one of two
readings; the other is "the lag is wrong". The paper cannot currently distinguish them, and the
grokking logs are precisely the case where the lag is known to be wrong by an order of magnitude
(§6.2 says so).

Concrete fixes, in increasing order of effort:

- state the confound explicitly, and note that on the grokking logs a large
  $\rho_{\mathrm{ident}}$ is consistent with both a missing invariant set and a mismatched lag;
- add the **span-preserving variant**: double $E$ while halving $\tau$, so the delay window is
  unchanged and only the embedding dimension moves. This is the clean test of the property §3.4
  claims to measure, it is a two-line change, and running it on the five calibration arms plus the
  seventeen logs would either confirm the existing diagnostic or replace it with a better one.

### 2.5 The taxonomy: four quantities, a title that names three, and one that never works

§3.1 is titled "Available, functional and active dimension" and defines four things (the fourth,
the trajectory effective rank, arrives in the second paragraph). §1 says "we separate four
quantities". Fix the title.

More substantively: the **functional dimension** is defined with equal billing, given a formal
definition as the rank of $\partial\vf/\partial\vc$, and then does nothing. It appears once more, in
Appendix E, as a constant ($10$ at every rank, seed and regime). A four-way distinction in which one
member is inert weakens the taxonomy rather than strengthening it. Either demote it to a remark
inside the paragraph on available dimension, or give it a job — the obvious one is a system where
functional and active dimension differ, which would be a genuinely novel control and is the natural
companion to §6.4's separation of active dimension from effective rank.

### 2.6 Smaller mathematical and notational points

- §3.4: "it is what separates the transient regime, where $\rho_{\mathrm{ident}}$ is near unity for
  the wrong reason" — separates it *from what*? The sentence has no second term.
- The **degeneracy flag** is a numerical-conditioning check, not a dynamical-regime diagnostic, but
  it is introduced as the third of the "admissibility diagnostics". Say what it is: a guard against
  quantised or constant observers, which is how it is actually used (it fires on probe accuracy in
  100 % of windows and on nothing else in the calibration data).
- The abstract says "Two diagnostics"; §3.4 says "two statistics and one flag". Pick one.
- Notation drift, one canonical name each please: *trajectory effective rank* / *effective rank* /
  *participation ratio* / *trajectory participation ratio* / *measured effective rank* are used for
  the same object; $\PR^{\mathrm{pos}}$, $\PR^{\mathrm{det}}$ and $\PR^{\mathrm{upd}}$ are defined
  in §3.1 but the text then says "the effective rank" without saying which; *the estimate* / *MG* /
  *the estimator* alternate; $E$ and $E_{\max}$ alternate for the same quantity.
- The `definition` theorem environment is declared and never used. Equation (1) is the paper's
  central definition and would gain from being a numbered Definition.
- The $\todo$ macro is defined and unused; `nicefrac`, `subcaption` and (with `\todo` gone)
  `xcolor` are loaded and unused.

---

## 3. Claims that the code and data do not support

*Scope: §§4–6 and Appendices A–E, L, audited against the implementation and the committed result
files. Owner: whoever holds the experiments. These are correctness issues, not presentation; several
must be fixed before submission whatever else changes.*

Two independent audits were run: one tracing the estimator implementation, one re-deriving numbers
from the result files. The canonical estimator is
`code/active_dimension/mg.py::all_estimators`, importing neighbour and Theiler machinery from
`code/grokking_analysis/edm/`. **Three rows of Table 3 were produced by a different, older
estimator** (`code/dimension_recovery/estimators.py`) with different defaults; see 3.4.

### 3.7 MAJOR — requirement 5 is violated in the paper's own headline table

Requirement 5: *"The three statistics of §3.2 are reported on every call."* Table 12
(`tab:grok-diagnostics`) prints `n/a` in two places and both are computed and committed:

- $\PR_{\mathrm{delay}}$ on the transformer rows — the values are $1.0169$, $1.5182$, $3.9450$ and
  $1.0000$. The $3.94$ is not close to one, which is presumably why its absence is noticeable;
- the trend-crossing count on the perceptron rows — `osc = 2.0` for all of them, **and §7.3 quotes
  exactly that number** ("two trend crossings per window"). The table is hiding a value its own
  discussion relies on.

Print all four. Withholding computed companion statistics in the table that classifies the paper's
application, under a protocol that requires reporting them, is the kind of thing a hostile reviewer
finds and a friendly one resents.

### 3.8 MAJOR — the flagged-window convention described in Appendix A is not what the grokking pipeline does

Appendix A: *"windows are dropped only when a series is summarised over its sliding windows, which
is where every number in this paper is formed."* True of `mg.py::summarise` (lines 234–245), used by
the ladder pipelines. The grokking summaries do **not** call it: `e5_real_logs.py:98-103` and
`dimension_probe.py:124-129` take pandas medians over all windows, flagged or not. No published
cell is affected (all have zero degenerate fraction), so this is a documentation defect rather than
a numerical one — but "every number in this paper" is false precisely for the application section.

Relatedly, "the degeneracy flag fires on eleven of the thirty-five run-by-column cells" counts cells
in which *more than half* the windows flag; cells with at least one flagged window number **14**.
State the counting rule.

### 3.9 MAJOR — the twenty-direction column of Table 7 cannot be reproduced

Every cell of the *eight-direction* column of Table 7 reproduces exactly from
`e2_rank_sweep/sweep_raw.csv` under a clearly identifiable recipe (qp arm, withheld ranks, drop
`acc_probe` and `loss_step`, median over seeds and observers at each $r$, score against `traj_PR`).
Applying the same recipe to `k20_calibration/scores_frozen.csv` does **not** give the
twenty-direction column:

| statistic | paper | recomputed |
| --- | --- | --- |
| MG | 1.632 / +0.961 | 1.624 / +0.973 |
| LB | 1.358 / +0.979 | 1.358 / +0.983 |
| TwoNN | 3.296 / +0.864 | 3.484 / +0.889 |
| delay PR | 2.320 / +0.982 | 2.185 / +0.979 |
| spectral PR, 256 | 5.401 / +0.996 | 5.286 / +0.998 |
| spectral PR, 1024 | 4.547 / +0.996 | 4.471 / +0.998 |
| spectral PR, native | 2.760 / +0.968 | 2.784 / +0.934 |
| roughness | 10.069 / −0.482 | 9.895 / −0.568 |

Mean and median aggregation, both orderings, per-observer aggregation, leave-one-observer-out and
leave-one-rank-out were all tried; none reproduces the column. The differences are small
($\le 0.2$) and directionally consistent, so this is most likely a stale table computed from an
earlier version of the scores rather than an error of substance — but an unreproducible table
column is exactly what a reviewer requesting the artifact will find. Regenerate it from a script
that is committed alongside, and record the recipe in the caption.

Note also that Table 3's $+0.973$ for the same quantity **is** reproducible ($0.9729$ exactly),
while Table 7's $+0.961$ is not. The paper quotes both, in different places, for the same
measurement.

### 3.10 MAJOR — three claims are not supported by the tables that are supposed to support them

- **"At four tenths of a period the estimate follows the rank to within one component"**
  (Appendix D, describing the correct lag). Table 9's own $\tau = 8$ row gives $4.18$ at $r = 3$
  (error $1.18$) and $6.16$ at $r = 4$ (error $2.16$). The claim is false against the table printed
  three lines below it, and it is the sentence that establishes what a *correct* lag looks like.
- **"The estimate stays between $3.74$ and $4.06$" at four active directions** (§6.4). The $3.74$
  is the $\rho = 0.6$ row, and Table 11 prints only $\rho \in \{1.0, 0.7, 0.5\}$. The table cannot
  support its own sentence. Either print the full six-value $\rho$ grid or quote the range the
  table shows.
- **"The measured effective rank equals $r$ to within $0.008$ components"** (§5.3). From
  `sweep_raw.csv`, the largest deviation of the per-$r$ median from $r$ is $0.0034$ and the largest
  over individual runs is $0.0128$. Neither is $0.008$. Right order of magnitude, wrong number under
  both conventions — pick one and quote it.

And one claim that cannot be checked at all: **"the effective rank measures $r$ to within $0.003$"**
for the oscillating matrix (§5.1). Neither `exp9_frobenius_k10` nor `exp10_frobenius_k20` records any
participation ratio; there is no PR column in `calibration_grid.csv` or `stationary_validation.csv`
and neither report mentions one. The closest committed measurement of that system is the anisotropy
experiment at $\rho = 1.0$, where the deviation is below $1.1\times10^{-4}$. Under requirement 1
every system's realisation must be verified by measurement; for this system the verification is not
in the repository.

### 3.11 MAJOR — the aggregation convention changes between rows of the same table

Table 3's caption states one unit of analysis: "one (rank, seed, observer) run". The image-data
$1$–$8$ row ($0.87$) follows it. The image-data $1$–$20$ row ($1.62$) does not — it is computed
after aggregating over seeds and observers. Run by run, that row would read **2.94**. Conversely
the $1$–$8$ row under the aggregated convention reads $0.457$, which is what Table 7 prints.

So $0.87$, $0.47$ and $0.457$ are three aggregations of *the same cell of data*, and $1.62$ and
$2.94$ are two aggregations of another; the paper prints four of these five numbers in three
different places without saying that any of them describe the same measurement. This is the single
most confusing thing in the results, and it is fixed by one sentence defining the convention plus a
consistent application of it. If both conventions are wanted, print both columns.

### 3.12 MAJOR — $r = 7$ was never simulated

`e2_rank_sweep` runs $r \in \{1, 2, 3, 4, 5, 6, 8\}$. Every statement of the form "across
$r = 1, \dots, 8$" is therefore wrong, including the zero-learning-rate sequence in Appendix C whose
seven values were flagged above as a possible missing entry — nothing is missing, the range is
misstated. Write $r \in \{1, \dots, 6, 8\}$ wherever it appears, or say "seven ranks between one and
eight".

### 3.13 Smaller discrepancies

- **Table 2, "withheld seeds 0–4"** implies five seeds; the code uses `SEEDS = (0, 1, 2, 3)`
  (`e2_rank_sweep.py:71`) and the text elsewhere says four. Write 0–3.
- **Appendix D, "the estimate at $r=8$ moves by $61.5$ components across the sweep"** uses the raw
  per-seed extremes while Table 9 prints seed medians, under which the move is $60.75$. Same
  sentence, two conventions.
- **Appendix D, rotation "is scored on the three projection observers alone, since the remaining
  nine are invariant to it"**: `controls_scored.csv` has the rotation control on **one** observer,
  `c_proj1`. One and eleven, not three and nine.
- **§3.1, "the effective rank falls from $7.99$ to $6.43$ under stochastic forcing"** — the two rows
  are `noise` and `noise_nopre`, and Table 10 prints the first as $7.98$. The text contradicts its
  own table by $0.01$; $7.99$ is the `qp_slow`/`mixed` value.
- **The true value at twenty directions** is printed as $20.0$ in §5.3 and $19.99$ in Table 17; the
  measured value is $19.993$. Also, Table 6 has no truth row, so "saturates at $15.1$ against a
  true $20.0$" cannot be checked against the table it cites.
- **Appendix C's roughness caveat is generous to itself**: "for seven of the nine usable observers
  the roughness statistic carries little or no ordering" describes a column containing $+0.800$,
  $-0.800$ and $-0.800$. Three of the nine have $|\rho| = 0.8$ on four points. Either say so, or
  drop the column and make the point qualitatively — with $n = 4$ it cannot carry this weight.
- **Appendix L, "the absolute value moves down by $0.1$ to $0.5$ components"** — the data gives
  $0.11$ to $0.51$.
- **"a factor of four across seven runs"** (§7.3): across the seven transformer runs the MG values
  span $3.94$–$28.08$, a factor of $7.1$. The factor of $\approx 4$ and the quoted
  $\rho_{\mathrm{ident}}$ band $1.29$–$1.60$ both describe the **five** regularised runs only; the
  two zero-weight-decay runs sit at $0.999$ and are classified transient in the same paragraph.
  Write "five".
- **$\rho_{\mathrm{ident}}$ is aggregated by two different rules**: the ladder takes the ratio of
  window-medians, the grokking pipeline takes the median of per-window ratios. The paper describes
  one rule. They are not the same statistic.
- **Standardisation happens in different places**: per window in the grokking pipelines (as
  Algorithm 1 states), once over the whole series in the ladder pipelines (`runner.py:68`,
  `e1_calibration.py:74`, `score_k20_parallel.py:39`). Immaterial on stationary arms; on the
  non-stationary transient arm the dither and the numerical floors then sit at the wrong scale.
- **§5.3 and Appendix C disagree on the aggregation**: the body says the four seeds are "averaged
  first", the appendix says "the median over the four seeds [...] is taken before scoring". Pick
  one word.
- **The reported error changes across tables for the same quantity**: $0.87$ (§5.3, run by run),
  $0.47$ (§5.3, after seed aggregation), $0.457$ (Table 7, MG at eight directions), $1.62$
  (Table 3) against $1.632$ (Table 7). Some of these differ by aggregation and some by rounding;
  as printed the reader cannot tell which. One convention, stated once.
- **Appendix C's zero-learning-rate sequence** lists seven values "across $r = 1, \dots, 8$". All
  seven are correct; the range is wrong — see 3.12.
- **The observer-dependence paragraph's closing sentence** — "at a lag common to both observers the
  two factors become $6.6$ and $1.9$" — is misleading. $6.65$ is the equal-weight observer's factor
  at $\tau = 4$ and $1.91$ is the generic observer's factor at $\tau = 1$: two *different* common
  lags, presented as one. At $\tau = 4$ for both, the factors are $6.65$ and $4.69$; at $\tau = 1$
  for both, $3.38$ and $1.91$. The point survives either way, but as written it is not a
  common-lag comparison.
- **The constant-rescaling control never appears in Table 10**, and the corrected value ($0.000$ on
  all eight observers) is never printed anywhere; §6.3 refers to it only qualitatively. Given that
  the repository's own notes record this as a defect found and fixed, printing the corrected zero is
  worth two characters — it is the cleanest invariance result in the paper.
- **Table 10's percentages** are computed against a denominator of $3.95$, the observed fall of the
  change-detection experiment. Say so in the caption; "as % of a real four-component change" invites
  the reader to divide by 4 and get different numbers.
- **"Six excitation regimes"** (§5.3) against "five excitation regimes" (Figure 1 caption) against
  nine arms plus controls in Table 10, against eleven arm labels in the CSV (including a second
  $\eta = 0$ control, `noise_eta0`, that Table 10 omits). Enumerate the six in §5.3 and have the
  caption say "five of the nine arms of Table 10".
- **"Six systems"** (§5, abstract) against seven distinct systems in Table 3 if the function-space
  construction is counted separately, which §5.3 treats as separate ("A second construction in
  function space"). State the mapping from systems to table rows.
- **The twelve observers are never defined.** Table 5 names them in English ("margin", "probe
  loss", "function-space norm", "parameter norms (two)") and no definition appears anywhere in the
  paper. From `dynamics.py:86-95` they are `loss_step, loss_full, loss_probe, w_fro, c_norm,
  fn_fro, g_fro, g_proj, c_proj1, fn_proj1, margin, acc_probe`, and "parameter norms (two)" merges
  `w_fro` (norm of the full parameter vector) with `c_norm` (norm of the subspace coordinate) —
  two quite different quantities reported as one row and one MAE. An appendix table with a formula
  per observer is required for reproducibility, and the merge needs a justification, particularly
  since those two are the paper's only observers with a perfect roughness–rank correlation.

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

### 4.1 CRITICAL — the headline collapse numbers are not reproducible, and the committed data is less favourable

This is the most serious finding in the review. §7.1's central sentence reads: the function-space
effective rank collapses "by a factor between $2.9$ and $15.6$, median $6.6$".

The committed per-run values in `active_rank/results_fine/rank_dip.csv` are:

| run | plateau | dip | factor | bottom, steps after $t_{\mathrm{gen}}$ | parameter-space dip | lag |
| --- | --- | --- | --- | --- | --- | --- |
| `mod_wd1` | 23.12 | 4.42 | **5.23** | +595 | 1.086 | 500 |
| `mod_wd1_s43` | 33.06 | 2.71 | **12.21** | +235 | 1.148 | 700 |
| `mod_wd1_s44` | 28.26 | 3.65 | **7.74** | +515 | 1.133 | 800 |
| `s5_wd1` | 21.15 | 3.41 | **6.20** | +712.5 | 1.601 | 150 |

The range is $5.23$–$12.21$ with median $6.97$, not $2.9$–$15.6$ with median $6.6$. About two hundred
alternative plateau and dip definitions were tried — different pre-window lengths, mean against
median against max, dip measured near $t_{\mathrm{gen}}$ or after $t_{\mathrm{mem}}$ or globally, all
eight recorded participation-ratio columns — and none produces the published triple. No script in the
repository emits it.

What makes this worse rather than merely careless: `icomp_v2/README.md` records that these figures
"replace the earlier *factor of five to twelve*, which was read off the plotted mean rather than
computed per run". **Five to twelve is exactly the committed per-run range.** The revision replaced
correct numbers with unreproducible ones, in the direction of a wider and more impressive interval.

Three further claims in the same paragraph do not survive:

- **"Both then re-expanding above the level they started from."** True in parameter space for all
  four runs. In function space, by the committed `recovered` column the post-dip level is *below*
  the plateau in three of the four ($20.4 < 23.1$, $22.4 < 33.1$, $23.4 < 28.3$). Only if
  re-expansion is measured by the post-dip *maximum* does it exceed the plateau, and then only in
  three of four. The claim needs a stated definition and a weaker verb.
- **"The two runs without weight decay fall by only $1.5$ and $1.6$."** Not reproducible under any
  definition tried. `rank_dip_controls.csv` gives, for the deepest dip anywhere, `mod_wd0` at
  **3.15** in function space and `s5_wd0` at **7.06**. Aligned at the matched run's
  $t_{\mathrm{gen}}$, as Figure 3 does, they give $1.61$ and $1.33$. Your own
  `active_rank/report.md` states the honest version: "$1.17\times$–$3.15\times$ (with one
  $7.06\times$ outlier)".
- Consequently the contrast the paper rests on is **much narrower than reported**. A control falling
  by $7.06$ against treated runs falling by $5.23$ to $12.21$ is not "a median factor of six in four
  regularised runs and not in their controls". The defensible claim is about the *shape* — the
  treated runs collapse within a few hundred steps of a specific event and then recover, the
  controls decline once and stay down — which is exactly what §7.2 already argues and argues well.
  §7.1's magnitude framing should be replaced by §7.2's shape framing, and the abstract with it.

Also worth knowing before you re-derive anything: `active_rank/dip.py` as committed computes a
different set of statistics (`fn_PR_step5`, `fn_PR_step20`, `PR_step20`, `PR_pos_det`) than the ones
in the committed `rank_dip.csv` (`fn_PR_pos_det`, `PR_step5`). **The script in the repository does
not regenerate the file §7.1 depends on.** Fix that first, then re-derive every number in the
paragraph from it.

### 4.2 MAJOR — §7.2's numbers do not match the data either

- **"The function-space effective rank falls from $28.6$ to about $6$"** (modular). The final value
  is right ($6.64$; median over the last eight windows $5.97$), the $28.6$ is not in the data: the
  peak is $32.70$, the rolling-smoothed peak $29.2$, and the value at memorisation $5.15$.
- **"From $19.3$ to $8.1$"** ($S_5$). Final value $8.10$ correct; $19.3$ not reproducible — the peak
  is $33.43$ and the median over the thousand steps after memorisation is $19.65$.
- **"Saturating at $1.4$ times its initial value" / "growing steadily to $2.4$ times"**. These are
  ratios to the **first logged window**, not to the initial norm. Against the true step-0 norm they
  are $1.73\times$ and $2.61\times$. Say which baseline you mean.
- **"Where in every generalising run it peaks and then falls by more than half."** Final over maximum
  is $0.448$, $0.473$, $0.517$ and $0.626$. **Two of the four fall by less than half.** Write "by
  between a third and a half".

### 4.3 The strongest result is stated in the weakest position

The rank collapse is named in the abstract, named in the conclusion, and shown on page 19 of an
appendix titled "Figures for the grokking application". A reviewer who reads the body and skims the
appendices never sees the paper's best plot. Move it into §7.1 (see section 6 for what pays for the
space).

### 4.4 The direct measurement's controls are the weak point and the paper knows it

Four generalising runs against two controls, and the controls differ from the treated runs in the
one variable (weight decay) that also drives the parameter norm, which is the observable the
nuisance analysis of §6.3 identifies as the dangerous one. §8 says "two imperfectly matched
controls" and stops. The reader needs to know what a *well*-matched control would be and why it was
not run. Two candidates that would substantially strengthen the claim, in order of cost:

- the same weight decay with a training fraction below the grokking threshold, so the regulariser
  is present and generalisation is not — this separates "weight decay" from "generalisation" and it
  is the control the argument actually needs;
- the label-matched perceptron pairs, which *are* properly matched, and where Appendix H reports
  that the direct measurement finds nothing. That negative is currently presented as a resolution
  limit; after the window-length sweep it is a genuine negative, and the paper says so in Appendix
  H but not in §7 or §8. The honest headline is: the collapse is present in the regularised
  stochastic setting and absent in the properly controlled one. That is a weaker claim than the
  abstract makes and it is the one the data supports.

### 4.5 "At the onset of generalisation" overstates the timing, in the wrong direction

The abstract says the collapse occurs "at the onset of generalisation"; §8 says "at
generalisation"; §7.1 says the bottom is reached **235 to 712 steps after** the generalisation step
in function space and up to eight hundred steps later again in parameter space, with a $\pm 300$
step localisation from the window. The body's own word — "near" — is the defensible one, and §7.1
already explains why it cannot be sharpened. Make the abstract and conclusion agree with the body.
This matters more than it looks: "at the onset" invites the reading that the collapse is a
*cause* or a *predictor*, which §8 explicitly disclaims two sentences later.

### 4.6 §7.3's "fixed nuisance" argument has a logical slip in a load-bearing sentence

*"A change inside a run is therefore read against a fixed nuisance, which §6.3 measures at about a
third of a real four-component change for the largest of them."*

What is fixed within a run is the *configuration* — $E_{\max}$, $\tau$, the window, the observer —
and therefore the level offset. The nuisance §6.3 measures is precisely **not** fixed: a ramping
observer gain produces a *drift* of up to $1.16$ components, and a ramping state amplitude $1.32$.
The sentence says the nuisance is fixed and then quantifies how much it moves. Rewrite as: the
configuration is fixed, so a change cannot be attributed to it; the residual nuisance is the
observer's own drift, which §6.3 bounds at about $1.2$ components over a run.

There is a second-order version of the same problem worth checking: the Theiler window is
recomputed per window from the autocorrelation time, so it is only constant within a run because
the 150-sample cap happens to bind (see 3.2). If any window of a run has $t_{\mathrm{acf}} < 150$,
$W_T$ changes inside that run and the "fixed configuration" premise fails for it. Verify and state.

### 4.7 The observer used on real logs is the one the paper flags as least trustworthy

Both §6.3 and Appendix C establish that the two parameter norms are the only observers whose
roughness orders the withheld ranks perfectly, so "for the cheapest observer a run logs a smoothness
explanation cannot be excluded". The observer used for every transformer row of Table 12 and for
Figure 3 is the parameter norm. The paper makes this point in §6.3 in the abstract and never
connects it to §7, where it bites. One sentence in §7.3.

### 4.8 The label-matched pairs prove something narrower than the section implies

The pairs are the paper's best-controlled comparison and the result is real (about three units of
separation, no overlap, confirmed in the committed summaries). But the paper's own Figure 4(b)
shows the mechanism: the two members' training losses differ by two-to-three orders of magnitude,
so the estimator is reading a difference that is visible by eye in the standard log. §7.3 says this
("the mechanism is visible in the standard log"), which is right and honest — but it means the
result is *not* evidence that the estimator adds anything. Consider making that explicit as the
conclusion of the paragraph rather than leaving the reader to infer it, and consider adding the
obvious comparison: does any simple statistic of the loss curve (final value, slope, curvature)
separate the pairs equally well? If yes, say so; if no, that is a much stronger result than the one
currently claimed.

### 4.9 Appendix H is a better result than its placement suggests

The window-length sweep is, methodologically, one of the most useful things in the paper: it shows
that a trajectory participation ratio computed over a badly chosen window reports the curvature of
the trajectory rather than anything about learning, that the scale at which the statistic acquires
range is two orders of magnitude above the published window, and that at that scale the generalising
run and its matched control are not separated. It also establishes that introducing mini-batches
moves the statistic by a decade while learning is unchanged.

That is a warning the field needs and it is in Appendix H under the title "The full-batch setting".
At minimum it deserves a sentence in the body's practical-guidance block (§1.7) and a mention in §8.

Every one of Appendix H's twenty-eight table entries and all of its surrounding claims reproduce
exactly from `rank_fb_long/pr_vs_window.csv` — this appendix is the most solid piece of arithmetic in
the paper. One caveat to add: "the same statistic then ranges over more than a decade" under
mini-batching is true of `PR_step` ($46\times$), `fn_PR_pos_det` ($17\times$) and `fn_PR_step`
($19\times$), but only $7.3\times$ for `PR_pos_det`, which is the statistic Table 18 actually
tabulates. Name the column or soften to "by up to a factor of forty".

### 4.10 Appendix I's two anchor numbers are wrong

- **"which memorises at step $2\,250$"** for the $p = 211$ run. The log
  (`Grokking/modular_addition_grokking_colab/training_log_p_211_wd_0.csv`) first crosses 95 % training
  accuracy at step **1 800**, and every sustained-block variant gives 1 800 as well;
  `exp7_inventory.csv` records `t_mem = 1800`. The $2\,250$ traces to a legacy audit note using a
  different criterion than the one Appendix K defines. Table 16 carries the same $2\,250$.
- **"a fall is detected at step $49\,250$"**. In `exp7_transcript.txt` that figure is in the *delay*
  column, and the transcript's own budget column ($198\,200 = 200\,000 - 1\,800$) confirms delay is
  measured from memorisation. The absolute step is therefore about **$51\,050$**. Nothing in the
  argument turns on it, which is exactly why it should be right.
- The $3.6\,\%$ against Table 16's $0.04$ is **not** a contradiction — $0.0361$ rounds to $0.04$ —
  but printing the same quantity to different precisions in two places invites the reader to think
  it is.

### 4.11 The run inventory is incomplete, and Appendix K's memorisation claim is wrong

Table 16 claims to list "every training run the paper draws on". Missing:

- **`x_mix_quad`**, which is plotted in Figure 4 — `make_figures.py:207-211` loads the whole of
  `arith/dimension_probe_summary.csv` with no run filter, so it is one of the squares;
- **`a_sub`** ($n-m$) and **`a_sq_sum`** ($n^2+m^2$), which are two rows of Table 15.

Also present in the results and absent from the inventory, though not obviously used: `a_sum_sq`,
the two `f_p1*` Adam runs at weight decay 5, and six `g_*_p23` polynomial runs — the sentence "a
third pair repeats them at $p = 23$" covers only the sketch pair, not these.

Separately, Appendix K states that the two memorisation criteria "differ on one run in this table and
by $1\,080$ steps". The $1\,080$ is correct, but it is the difference in the **generalisation** step
of `lowdata15_s0`. The **memorisation** column differs under the two rules on **six of the seven**
extended runs (first-crossing $1\,380, 600, 640, 650, 800, 590, 620$ against the table's
$1\,450, 620, 630, 660, 800, 610, 630$). The differences are small and no claim turns on them, which
is what the sentence should say instead.

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

### 5.1 CRITICAL — Figure 1(b) is not computed on the data its caption claims

The caption reads "(a) [...] median over four seeds and ten observers. (b) The identifiability
ratio **on the same runs**." In `e2_rank_sweep/sweep_raw.csv`, `ident_ratio` is non-null in 54 of
2 240 filtered rows: **seed 0 only, three observers only** (`c_proj1`, `g_fro`, `w_fro`), **ranks 2
and 6 only**. Each row of panel (b) is therefore two points, each a median over three observers at
one seed. The caption's aggregation is false for that panel, and the reader has no way to know that
each regime is represented by two measurements.

Either recompute $\rho_{\mathrm{ident}}$ over the full sweep — which would also give the diagnostic
a defensible calibration, currently its weakest point (§1.10) — or state the restriction in the
caption. The first is much better: the diagnostic is one of the paper's four claimed contributions
and it is calibrated on 54 numbers.

### 5.2 Other caption-versus-data defects

- **Figure 3 caption says "Windows of 600 steps"** for all six runs; the $S_5$ runs use 295-step
  windows (Appendix G says so correctly). The $\pm 300$ localisation claim is $\pm 150$ for those
  runs, and the plotted mean averages curves computed at two different window lengths on one step
  axis — worth a clause.
- **Figure 4's x-axis is labelled "recurrences per window"** and plots the trend-crossing count.
  §3.4 insists the statistic "tests non-monotonicity of the series, **not** recurrence". The
  rename after the August audit did not reach the axis label (or the phrase "hence the recurrence
  count of §3.4" in Figure 1's caption).
- **Figure 4 promises seventeen runs and shows about seven marks.** All ten perceptron runs sit at
  (crossings 2, $\rho_{\mathrm{ident}} \approx 1.0$) and plot as one open square, inside which the
  two zero-weight-decay transformer points also land. The coincidence is arguably the message — they
  degenerate identically — but it must be said in the caption, or jittered, or annotated "×10".
- **Figure 4's caption and §7.3's text disagree about Table 12**: the text says the table is what
  the figure plots; the table has 4 of the 7 transformer runs and 3 of the 10 perceptron runs.
- **"Three orders of magnitude short"** (§7.3, Figure 4 caption) is $2.65$ orders for the pair
  actually plotted (final training loss $8.84\times10^{-6}$ against $3.96\times10^{-3}$). The other
  pairs give $3.19$ and $3.06$. Either soften, or plot a pair where it is true.
- **Figure 1's transient points pile up** at $x \approx 1.05$ (seven points, two visible diamonds)
  because the transient's measured rank is $\approx 1$ at every nominal $r$. That is the correct
  behaviour and the caption should say it, or a reader reads it as missing data.
- **Figure 3's two controls** are dashed and dotted and only one is in the legend; which is $S_5$
  is undiscoverable, and in panel (c) they differ by three decades, which invites a question the
  figure does not answer. Give each its own legend entry.

### 5.3 MAJOR — not one figure in the paper carries an uncertainty

For a paper whose subject is the reliability of an estimator, this is the first thing a referee will
write down. The data exists in every case:

- Figure 1(a): forty values behind each point (four seeds × ten observers) in `sweep_raw.csv` — an
  interquartile band is a one-line change;
- Figure 3: the two-hash-family disagreement is recorded per window in `*_sketchsd` columns, and
  Appendix G quotes it (1.1 % median, 8.4 % worst) — plot it as a band and the appendix's argument
  becomes visible;
- Figure 4(a): window-level values exist in `dimension_probe.csv`, so whiskers are available.

Figure 2 is per-run and can remain point-only.

### 5.4 Table hygiene

- **Aggregation unstated** in the captions of Tables 10 (`tab:controls`), 12
  (`tab:grok-diagnostics`) and 7 (`tab:alts`, which says "the same convention" without saying
  which). Table 3's caption is the model to copy — it states the unit of analysis explicitly.
- **Digits.** Spearman correlations are quoted to three decimals on four withheld ranks; the
  $\rho(\text{roughness})$ column of Table 5 is a sequence of $\pm 0.200$ and $\pm 0.800$ values,
  which is $n=4$ noise printed as precision. Two decimals at most, and consider replacing that
  column with a two-level flag, which is all the paper claims from it.
- **Cross-table rounding mismatches**: Table 3's $1.62$ against Table 7's $1.632$; §5.3's "median of
  $0.47$" against Table 7's $0.457$.
- **Redundant columns.** Table 15 (`tab:ipr`): the "generalises" column is derivable from the
  "grokking step" column (a dash means no). Table 16 (`tab:runs`): the `arch` column takes two
  values already given by the block headers, and `wd` is zero for the entire perceptron block, also
  stated in its header. Table 5's `family` column is nearly redundant with the observer names.
  Table 2 has three rows identical in both columns ($m$, Theiler rule, dither) — fold them into a
  "common to both" note.
- **Overfull boxes** on four tables, worst at $+18.2$ pt for Table 16.
- **Tables that want to be figures**: the $\tau$ sweep (a divergence spanning $2.6$ to $63.4$),
  the twenty-direction saturation curve, the anisotropy table (the paper's central conceptual
  separation, currently twelve rows of digits), and the window-length sweep. See 5.6.
- **Table 3's `truth` column** is a binary "constructed / measured", which contradicts requirement 1,
  under which every system is both constructed *and* verified. Recast as "how the realisation was
  verified". The `protocol` column's values ("ordering only", "rank split only") need a legend.

### 5.5 LaTeX

- **Paragraph labels resolve to misleading numbers.** `sec:pairs`, `sec:whatfalls` and
  `sec:grok-diagnostics` all print "section 7.3"; `sec:diagnostics-validated` prints "section 6.4",
  whose title is about something else. Table 16's "used in" column shows "7.3" for rows pointing at
  three different paragraphs. Promote the two `\paragraph`s in §7 to subsections.
- **Appendix references are inconsistent**: three `\cref{app:...}` print "section A / B / K" while
  about twenty hand-written `Appendix~\ref{...}` print "Appendix". Add
  `\crefname{appendix}{appendix}{appendices}` and use `\cref` throughout.
- No undefined references, no duplicate labels, no citation warnings; `\PR` defined once; macros all
  present; the anonymous block behaves correctly. The `[capitalize]` option to `cleveref` is dead
  weight given the explicit lowercase `\crefname`s.
- The title's manual line breaks strand "of a" at the end of line 1; rebreak after "Freedom", and
  remove the breaks from any metadata copy of the title.
- Unused packages: `nicefrac`, `subcaption`, and `xcolor` once `\todo` is deleted.

### 5.6 Figures the paper should add

Ranked. All are feasible from committed result files unless noted.

1. **Move Figure 3 into §7.1.** Not new, but the highest-value change in this section.
2. **The $E_{\max}$ ceiling** (§1.5) — the estimate against true $r$ at two or more $E_{\max}$, with
   the $E_{\max}/2$ prediction drawn. This is the figure that would turn the paper's most-noticed
   weakness into a result. Needs the sweep of §1.5; partial data exists at $E_{\max} \in \{10,20\}$
   and $\{20,40\}$.
3. **The anisotropy experiment as a plot** — MG flat against $\rho$ while $\PR$ falls, one panel per
   $r$, from `e8_anisotropy/aniso_summary.csv` (which carries `MG_sd` and the closed-form
   prediction, so error bars and a reference curve are free). This is the separation the title turns
   on and it is currently invisible inside twelve rows of digits.
4. **Raw scalar-log traces, one per regime.** The paper is about reading time series and never shows
   one. Small multiples: a recurrent arm, a stochastic arm, a transient arm, and a real parameter-norm
   log from a grokking run, on a common axis, each labelled with its regime and its
   $\rho_{\mathrm{ident}}$. This single figure would make Table 1 and the whole of §6 concrete.
   Note: for the synthetic arms only windowed summaries are committed, so the raw series would need
   regenerating; the real logs are in `active_rank/results_fine/`. The accuracy, loss and
   parameter-norm traces for all four transformer runs already exist as plots in the previous draft
   (`icomp_article/images/{mod,s5}_wd{0,1}_{acc,loss,norm}.png`), so the real-log half of this
   figure is a restyling job rather than new work — and dropping every raw trace between drafts is
   arguably the one thing the rewrite lost.
5. **Sliding-window MG over training** for the transformer runs, from
   `e5_real_logs/real_logs_windows.csv` (281 windows already computed) — this is the missing
   experiment of §1.6, and it is a figure, not a table.
6. **Per-observer recovery with across-seed spread** — replaces or supports Table 5 and answers the
   practical question the paper raises and never resolves: which log should I keep? `observer_scores.csv`
   carries `seed_sd`.
7. **The $\tau$ sweep as a plot** (estimate against span in periods, log axes, one line per $r$),
   from `e6_tau/tau_sensitivity.csv`, which has 336 rows including seeds, so error bars are free.
8. **A schematic of the four quantities of §3.1** — one cartoon: an affine subspace, a torus inside
   it, an anisotropic cloud on the torus, and the map to outputs. No data needed, and §3.1 is the
   densest prose in the paper.
9. **The window-length sweep as a plot** (Appendix H), log-x, which makes the resolution-limit
   argument visual in a way fourteen numbers do not.

If only three can be added: 2, 3 and 4.

---

## 6. Structure, placement and the page budget

*Scope: what goes where. Owner: whoever does the restructure. Assumes the nine-page body limit is
the only hard constraint.*

### 6.1 The current distribution is indefensible

The body has **one figure in nine pages**. The evaluation section (§5) has **no float at all** —
its summary table is on page 14. The application section's table is on page 18 and both its figures
are on page 19. Meanwhile the body carries some of the densest prose I have read in a submission.
A reviewer's experience of this paper is nine pages of unbroken argument followed by a
well-illustrated appendix.

### 6.2 What should move into the body

- **Table 3** (the ladder summary) into §5. It is eight rows and it is cited four times from §§5–5.3.
- **Figure 3** (the rank collapse) into §7.1. It is named in the abstract and the conclusion.
- **Figure 4** (the diagnostics map) into §7.3 if anything is left over. It is the visual statement
  of the paper's methodology-first thesis: the instrument refusing two settings, without labels.
- **Table 12** should sit beside whichever of the two survives, not in a different appendix from the
  figure that plots it.

### 6.3 Where the space comes from

I estimate 1.2 to 1.4 pages are needed. In descending order of how painless the cut is:

1. **§2, related work.** Currently four dense paragraphs, of which "Dimension in optimisation" is
   the one that must stay at full strength — it is where the paper positions itself. The delay-
   reconstruction and spurious-dimension paragraphs are enumerative citation lists that can each
   lose half their length to a footnote or an appendix without losing a reference. Saving: ~0.4 page.
2. **De-duplicate the motif.** "The estimator returns a number whether or not an invariant set
   exists" appears in the abstract, twice in §1, in §2, at the top of §4 and in §6. Once, with
   force, in §1, and a pointer thereafter. The recovery headline ("under one component up to eight")
   likewise appears four times. Saving: ~0.2 page.
3. **§5.2–5.3**, restructured per §1.5 into the ceiling thread and the silence-control thread, with
   constructions and per-system numbers in the appendix. Saving: ~0.3 page.
4. **§6.4** can lose half its length: its role is confirmatory, one number pair in prose suffices and
   the rest belongs with the (new) anisotropy figure. Saving: ~0.2 page.
5. **§3.1** loses the forward-referenced numbers ($7.99 \to 6.43$ and the rank-10 Jacobian) which
   mean nothing to a reader on page 3. Saving: ~0.1 page.
6. **§7.3's** "two conclusions pointing opposite ways" paragraph re-derives §6.3 from scratch.
   Saving: ~0.1 page.

### 6.4 Section-level renaming and re-ordering

- §3.1's title names three quantities and defines four.
- §5.2 "Adding an optimiser, and then a network" says nothing about what was found. Under the split
  of §1.5 the two new headings write themselves: "The ceiling" and "The silence control".
- §8 "Limitations and conclusion" — the conventional order is the reverse, and the section
  currently opens with a summary and ends with limits, so the title is inverted relative to its own
  content.
- Appendix F, "Figures for the grokking application", is a dumping-ground title and it opens with a
  table.
- Appendix C, "Results by observer, and the alternatives", contains Table 3, which is neither.
- The protocol requirements are referenced by number a dozen times ("fails req. 4"). Name them —
  the silence control, the frozen configuration, the disjoint split, the matched bandwidth — and use
  the names. The reader will not otherwise remember which is which, and Table 3's `protocol` column
  becomes readable at a glance.

### 6.5 A proposed body layout

For reference, not prescription:

| § | content | floats |
| --- | --- | --- |
| 1 | motivation, the four quantities in one paragraph, contributions | — |
| 2 | related work, compressed | — |
| 3 | definitions, estimator, regimes, diagnostics | Table 1; schematic (fig. 5.6-8) if space |
| 4 | the protocol, requirements named | — |
| 5 | the ladder: the ceiling; the silence control; the image-data system | **Table 3** |
| 6 | conditions of validity | Figure 1; anisotropy figure |
| 7 | grokking: diagnostics refuse both settings; the direct collapse; what a log adds | **Figure 3**, **Figure 4** |
| 8 | practical guidance, limitations, conclusion | — |

---

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

### 7.2 The abstract

About 250 words in one block, and the third sentence is 47 words with two nested relative clauses
whose attachment is ambiguous. Target 160–180 words with this skeleton:

1. the quantity and why one would want it (two sentences);
2. the instrument and its hazard (one);
3. the validation, with **one** number (one);
4. the diagnostics (one);
5. grokking: the diagnostics refuse both settings, and the direct measurement shows the collapse —
   with **one** number (two).

Cut from the abstract: the weight-decay-zero norm behaviour (a fourth-order detail), "on withheld
ranks and seeds" (say it once in the body), and either the $0.87$ or the "median factor of six",
not both plus three more. Fix "at the onset of generalisation" per §4.3, and reconcile "random
subspace" with the construction actually used.

### 7.3 The systemic prose problems

These are patterns, not instances; the examples are illustrative and the fix should be applied
throughout.

**Periodic sentences that withhold the point until the end.** §3.1: *"So stated it exists only where
such a set does, which in two of the three regimes of §3.3 it does not, and that is why the
diagnostics of §3.4 come before any number."* → "It exists only where a recurrent set exists. In two
of the three regimes of §3.3, none does. That is why the diagnostics come before any number."

**Design and result compressed into one sentence.** The "Recovery" paragraph of §5.3 carries twelve
observers, five families, five estimator parameters, four selection observers, reserved seeds and
ranks, and the out-of-sample status of six of ten — before reaching a number. Split: one sentence of
design, one of result.

**Numbers in prose.** §5.3, §6.4 and §7.1 carry roughly ten inline numbers each. Rule: one anchor
number per claim in the prose, full precision in the table the sentence cites. Specific demotions:
the LB/MG bias pair, the $\rho$-sweep quadruples of §6.4, and the collapse ranges of §7.1
($2.9$–$15.6$, median $6.6$, $235$–$712$, $1.09$–$1.60$, "one to eight hundred") — the last of these
wants a per-run appendix table with columns for depth, bottom offset and re-expansion, which would
replace six inline ranges with one reference.

**Ambiguous antecedents.** §5.2: "the two series correlating above $0.98$" — which two? Appendix C:
"Both figures also use the delay lag that is best for that observer" — which figures?

**Headings that carry no finding.** "Adding an optimiser, and then a network"; "Figures for the
grokking application". Compare "Without weight decay the function settles and the norm does not",
which is exactly right and should be the model.

**Terminology drift.** Listed in 2.6. Add a one-line notation summary at the end of §3.1.

**Requirement list formatting.** Each of the six requirements is currently an aphorism with its
justification embedded in a subordinate clause — requirement 6 in particular ("or a construction in
which each added direction adds a higher frequency lets the roughness alone order the rank") takes
three readings. Use: **bold name** — one sentence of rule — one sentence of reason.

### 7.4 Line-level notes

- "on a scale fixed by no free parameter" → "with no tunable scale".
- "sklearn digits" → name it properly once (the 8×8 handwritten-digit set distributed with
  scikit-learn); a library name is not a dataset name in a caption.
- "uncalibrated" (contribution 2) is unexplained at first use.
- "two of them are results and not settings" appears verbatim in both §5.3 and Appendix E.
- §6.1: "the error rises tenfold" — from what to what?
- §6.3 says "about a third of a real four-component change" for the observer gain; the table gives
  29 % for the gain and 33 % for the state amplitude. The text is quoting the wrong row's number
  for the named factor.
- §5.1: "That is the clearest evidence available that the geometry does work smoothness cannot" —
  "does work smoothness cannot" needs a comma or a rewrite; it reads as a garden path.
- Table 3 mixes bare and signed correlations ($0.980$ against $+1.000$).
- §4's preamble ("Delay-embedding statistics return a finite number on data satisfying none of their
  hypotheses") is the fourth appearance of that sentence's content.
- The paper uses "the estimate", "the estimator" and "MG" for three different things in adjacent
  sentences in §6.4.

---

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

Twelve items remain, in the order I would do them. Items 2, 3, 4, 7 and 13 of the original list
were applied in installment 1 and are struck.

1. **§7.1's collapse factors do not reproduce and the controls fall further than reported.**
   Re-derive every number in that paragraph from a script that regenerates the committed file, and
   replace the magnitude framing with the shape framing §7.2 already uses. *(4.1)*
2. **Reconcile the remaining overclaims**: "on six systems", "orders it to twenty", "at the onset
   of generalisation", "a factor of four across seven runs", "$28.6$ to about $6$", "falls by more
   than half", "$r = 1, \dots, 8$". *(3.10, 3.12, 3.13, 1.9, 4.2, 4.5)*
3. **Figure 1(b) is computed on 54 numbers from one seed** and its caption says otherwise. *(5.1)*
4. **Reorder §7 so the instrument is present at the headline result**, or add a roadmap sentence.
   *(1.3)*
5. **Move Figure 3 and Table 3 into the body**, paid for from §2 and the repetitions — and pay
   back the page the first installment cost. *(6.2, 6.3)*
6. **Turn the saturation ceiling into a result** — $E_{\max}/2$ predicts every saturation point in
   the paper, and $E_{\max} = 40$ cannot recover twenty components by construction. *(1.5, 2.2)*
7. **Fix the estimand definition** — a finite orbit closure is a curve; define it at a resolution
   $\varepsilon$ and §6.4 becomes a confirmation rather than a caveat. *(2.1)*
8. **Run the sliding-window estimate on the grokking logs**, or withdraw §7.3's constructive claim.
   The 281 windows are already computed. *(1.6)*
9. **Confront whether the admissible regime is ever realised in undriven training.** *(1.10)*
10. **Define the twelve observers.** *(3.13)*
11. **Fix the paragraph labels and appendix cross-references** that print misleading numbers.
    *(5.5)*
12. **Regenerate Table 7's twenty-direction column** and print $\PR_{\mathrm{delay}}$ and the
    crossings in Table 12 instead of `n/a`. *(3.9, 3.7)*
