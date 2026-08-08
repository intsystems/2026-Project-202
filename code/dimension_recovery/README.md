# Can a 1-D log recover a known dynamical dimension? — 2026-08-07

Answer to the two questions in the brief, plus the errors found in the brief's own
experimental design and a candidate replacement for the criterion.

```
python exp1_recovery.py         # which scalar observers recover a known k     ~5 min
python exp2_transitions.py      # 1->2->3->4->3->2, and the detection lag      ~3 min
python exp3_censoring.py        # are the controls counterexamples? (REFUTED)  seconds
python exp4_criterion.py        # the weight-norm drawdown rule, and its margin ~1 min
python exp5_criterion_v2.py     # the drop criterion: defects, revision, calibration
python exp6_absolute.py         # does any metric recover k NUMERICALLY?       ~11 min
python exp7_broader.py          # all 25 runs in the repo, four tasks          ~8 min
python exp8_extended.py         # what the 120 000-step reruns overturn        ~4 min
python make_figures.py          # figures/fig1, figures/fig2
sh launch_extended.sh           # the 120 000-step reruns themselves, CPU, ~90 min
python extend_runs.py --plan    # the same idea as a GPU plan, 200 000 steps
```

Everything runs on the repo's own estimator code (`../grokking_analysis/edm`), so what
is validated is the code the report actually uses. Results and transcripts in `results/`,
raw training logs of the reruns in `results/extended/`.

Two documents. **[`report_0808.md`](report_0808.md) is the current one** — it supersedes
the answers below wherever they disagree, because `exp8_extended.py` refuted two of them.
This README is kept as written on 2026-08-07 apart from the marked corrections, so the
errors stay auditable.

---

## 0. The two answers, up front

**Question 1 — which 1-D observers recover the known dynamical dimension?**
Six of thirteen, under conditions that have to be stated with them. `norm_fro`,
`norm_fro_sq`, `trace`, `logdet`, `norm_l1` and one of four random projections track
k = 1..6 with Spearman +0.94 to +1.00, pass every one-dimensional control, and are exactly
invariant to amplitude. **The blanket claim "a 1-D log cannot recover the dimension" is
false, and the previous report should not be read as asserting it.** But the conditions
are severe: several hundred oscillations of the observable inside one estimator window,
signal-to-noise above about 10³, and a resolvable k that is bounded by how much of the
torus the window covers. A `weight_norm` window in this project contains *zero*
oscillations, so the training logs are not near the working regime — which is the
conditional statement the report should have made and did not.

**Question 2 — are the non-grokking runs real counterexamples?**
Partly, and the objection is correct for the run it matters most for.

| run | last-2k val acc | vs chance 1/113 | trend | verdict |
| --- | --- | --- | --- | --- |
| `lowdata15` | 0.00023 | **0.03×**, p ≈ 0 | ρ = −0.07 | ~~extended_non_grokking~~ → **CENSORED: generalises at step 110 940** (exp8) |
| `lowdata20` | 0.01379 | **1.56×**, p = 3.9e-55 | ρ = +0.29, peak at 96 % of the run | **CENSORED — confirmed: generalises at step 39 600** (exp8) |
| `wd0` | 0.03320 | 3.75× | val acc *exactly constant* for 19 000 steps | ~~frozen~~ → **unresolved**: at 120 000 steps two seeds reach 8–10× chance and are still rising (exp8) |

**Superseded 2026-08-08 by `exp8_extended.py`.** All three were censored, not one. Two of
them generalise at 120 000 steps and the third has not been resolved. Every false-alarm
count in this project is void, and the answer to question 2 as posed is: **none of the
non-grokking runs is a real counterexample.** See `report_0808.md` §1.5.

---

## 1. Errors in the proposed design, found by building it

The setup is sound in outline and one of its details is exactly right. Six things had to
be fixed before it could answer anything, and two of them silently produce a *positive*
result for the wrong reason — those are the dangerous ones.

**1.1. The bandwidth confound (dangerous — produces a false positive).** The natural way
to pick k incommensurate frequencies is f_j ∝ √p_j. Then adding an oscillator also adds a
*higher* frequency, so the observable's spectral centroid rises with k. Measured
**[computed]**:

```
                Spearman(MG, k)   Spearman(roughness, k)
widening band        +0.94              +1.00
matched band         +0.94              +0.26
```

Under the obvious construction the one-line ratio std(Δx)/std(x) — no embedding, no
neighbours, no likelihood — recovers the ordering of k *perfectly*. An experiment built
that way cannot distinguish geometry from smoothness and proves nothing. All k
frequencies must be held inside a fixed band. Note this is the same confound that
`../prediction_improved/report_0708_experiments.md` §5 identified in the real logs, where
Spearman(d̂, roughness) = +0.934; it would have been reproduced here and mistaken for
success.

**1.2. Signal-to-noise must be defined at the observable, not per coordinate (dangerous —
produces a false negative).** With noise added to each of D = 64 diagonal entries, a
single oscillator moves ‖W‖_F least, so **k = 1 has the worst SNR in the sweep and reads
*higher* than k = 2**. My first run produced exactly that inversion — 8.08 at k = 1 against
5.23 at k = 2 — which looks like the estimator failing and is an artefact of the design.
Standardising each observable and adding measurement noise at a stated fraction of its own
spread removes it. (Standardising is free: the W = 0 estimate is scale-invariant.)

**1.3. Rational independence is a live constraint, not a formality.** The orbit closure is
a k-torus of dimension k **iff** 1, f₁, …, f_k are rationally independent. Two ways to
break it by accident: an integer number of cycles per window makes the sampled orbit
exactly periodic, so the delay vectors contain exact repeats, the neighbour distances
collapse to the noise floor and the estimate becomes meaningless; and *equally spaced*
frequencies give the exact resonance f₀ − 2f₁ + f₂ = 0, which collapses the torus to a
subtorus while the schedule still says k. I hit the first of these while building this and
it produced a dimension change with the wrong sign. `systems.resonance_margin` reports the
distance from degeneracy with every system.

**1.4. There is a coverage ceiling, so k and the sampling rate must be swept together.**
The largest resolvable k depends on how much of the torus one window covers **[computed]**:

```
 cyc/win    k=1    k=2    k=3    k=4    k=5    k=6    rho
      10   1.35   1.71   2.14   2.07   2.08   2.07  +0.49
      30   0.61   2.39   2.57   2.53   2.58   2.50  +0.60
     100   1.50   3.58   3.89   3.42   3.76   3.73  +0.49
     300   1.22   2.41   3.47   4.19   4.79   4.64  +0.94
    1000   1.22   1.38   4.51   4.39   4.39   4.35  +0.49
```

Seeing a k-torus locally needs returns in all k directions, of order c^k points. At the
top of the range *low* k breaks instead: a single sinusoid seen over a thousand periods
has near-exact recurrences and MG collapses to 1.38. **There is no single setting correct
for every k, only a band** — so a sweep over k = 1..6 at one fixed sampling rate, as the
brief proposes, will always fail at one end or the other.

**1.5. "Same dimension at different speed" cannot hold at a fixed window, and should not
be asked to.** Speed *is* cycles-per-window, which §1.4 shows is the controlling variable.
Measured at k = 3, reference 3.468 **[computed]**: amplitude ×0.1 gives 3.465 — invariant to
three decimals, as a scale-free statistic must be — while amplitude ×10 gives 3.567, a
2.9 % shift, because at that amplitude the oscillation is comparable to b_i and the
quadratic term Σδ² is no longer negligible, so the observable stops being a near-linear
functional of the angles. Speed ×3 at a fixed window moves the estimate to 2.43 and speed
×⅓ to 1.73; rescaling the window to hold cycles-per-window fixed recovers most of it
(2.31, 2.01). The control has
to be stated as *invariance at matched coverage*, and the practical rule that follows is
that a window must be chosen in units of the system's own recurrence time, never in steps.

**1.6. A ramped transition has no ground truth inside the ramp.** An oscillator at 5 % of
its final amplitude is not a degree of freedom the data contains, whatever the schedule
says. Measured: at a ramp comparable to the segment length the k = 1 regime reads 2.11
instead of 1.22 and the whole trace flattens. Any detection lag quoted against a ramped
truth is measuring the ramp. Use abrupt switches, or report the ramp as the resolution
floor.

**1.7. What the brief gets right, and it is the crux.** Requiring b_i ≠ 0 is not cosmetic.
With b_i = 0 the squared norm is Σδ_i², invariant under δ → −δ and therefore non-injective
on the torus; with b_i ≠ 0 the linear term 2Σb_iδ_i dominates and the norm becomes a
generic linear functional of the k oscillators. That is precisely why `norm_fro` lands in
`validated_good` below.

**1.8. The limit that no fix removes.** This is not a model of training. There is no
optimiser, no feedback, no non-stationarity of the kind a transient has; the k oscillators
are independent and prescribed. It is the *easiest possible* case, which makes a negative
result here decisive and a positive result here weak. It establishes what a scalar
observer can do at its best; it does not establish that a training log does it.

---

## 2. Question 1 in detail

### 2.1 The working regime

Sweeping window ∈ {300, 1000, 3000} × cycles/window ∈ {1, 3, 10, 30, 100} × SNR ∈
{10², 10³, 10⁴, 10⁶}, and asking whether MG orders k = 1 < 2 < 3 **[computed]**: 47 of 60
settings order them, 33 of 60 also separate them by at least 0.5. Three distinct failures:

* **too few cycles** — the window sees an arc, every k returns the tangent constant. At
  1 cycle/window and SNR 10⁶ the three values are 1.35, 1.35, 1.38: ordered, and spread by
  0.03, which no finite sample can resolve. **This is the regime the training logs are in,
  and it is worse than it looks — a `weight_norm` window contains no cycles at all.**
* **too much noise** — at SNR 10² the ordering is destroyed at every window length.
* **too many cycles at low k** — near-exact recurrences, MG returns 0.30.

### 2.2 The observer taxonomy

At a deliberately favourable reference point (window 2000, 300 cycles/window, SNR 10⁶,
matched band, 3 seeds), MG against k **[computed]**:

```
observer                 k=1     k=2     k=3     k=4     k=5     k=6    rho   verdict
norm_fro                1.22    2.41    3.47    4.19    4.67    4.64  +0.94  validated_good
norm_fro_sq             1.22    2.41    3.47    4.19    4.67    4.64  +0.94  validated_good
trace                   1.22    2.28    3.42    4.19    4.52    4.53  +1.00  validated_good
logdet                  1.22    2.52    3.31    4.14    4.53    4.62  +1.00  validated_good
norm_l1                 1.22    2.28    3.42    4.19    4.52    4.53  +1.00  validated_good
proj_rand1              1.22    2.40    3.09    3.40    3.91    4.04  +1.00  validated_good
proj_rand0              1.22    3.51    3.22    4.22    4.05    4.02  +0.71  unknown
proj_rand2              1.22    2.20    3.18    3.35    3.93    4.27  +1.00  unknown  (seed sd 0.48)
proj_rand3              1.22    2.54    3.53    4.60    4.37    4.34  +0.77  unknown
coord_active0           1.22    1.22    1.22    1.22    1.22    1.22  -0.20  validated_lossy
proj_half_active        1.22    1.22    1.22    2.90    2.91    3.66  +0.77  validated_lossy
coord_inactive         13.17   13.18   13.20   13.20   13.20   13.20  +1.00  validated_bad
proj_inactive_only     12.02   12.36   12.21   12.26   11.76   11.97  -0.54  validated_bad
```

Every observer landed in the class predicted in `systems.EXPECTED` *before* the run. Two
rows deserve attention:

* `coord_active0` watches a single active coordinate, so its image is a circle and its
  true dimension is 1 for every k. It reads 1.22 for every k. **That is the estimator
  being right about a lossy observer, not the estimator failing** — the distinction the
  brief asks for, and the reason `validated_lossy` is a separate bucket from
  `validated_bad`.
* `proj_half_active` sees ⌊k/2⌋ angles, so its image dimension is 1, 1, 1, 2, 2, 3. It
  reads 1.22, 1.22, 1.22, 2.90, 2.91, 3.66 — it tracks its own image dimension exactly,
  and not k.

Three of four random projections came out `unknown`: they separate k = 1 from k ≥ 2
decisively but are non-monotone above that or vary too much across seeds. That is an
honest outcome at three seeds, not a failure.

### 2.3 The controls — the decisive table

All rows have four moving coordinates. Only the first has four degrees of freedom.

```
mode              true        LB        MG        PR     TwoNN roughness
quasiperiodic      4.0      5.74      4.19      2.63      4.85      0.90
sync               1.0      1.55      1.22      1.99      0.66      0.86
sync_phased        1.0      1.55      1.22      1.99      0.66      0.86
scale_periodic     1.0      1.55      1.22      1.99      0.66      0.86
scale_monotone     1.0      1.33      1.33      1.00   1556.43      0.00
noise_smooth       4.0      3.15      2.42      1.05      5.54      0.04
noise              inf     17.54     13.25     14.90     13.58      1.42
```

`sync_phased` is the control that matters: four coordinates visibly moving, four different
fixed phase offsets, one degree of freedom. **MG reads 1.22 against 4.19 for four genuine
oscillators, while roughness reads 0.86 against 0.90.** The geometry does work here that
smoothness cannot do — a 3.4× separation against a 5 % one. This is the strongest result in
the file and it is the one that licenses calling these observers validated at all.

Two failures worth recording: TwoNN returns 1556 on a monotone curve and is unusable
there; PR reads 1.99 for one-dimensional periodic motion, because a circle spans a
two-dimensional linear subspace and PR is linear.

### 2.4 Transitions 1 → 2 → 3 → 4 → 3 → 2

```
segment  true k    MG   sd over seeds  roughness
      0       1  1.22            0.00     0.8648
      1       2  2.27            0.25     0.9345
      2       3  3.46            0.21     0.8682
      3       4  4.20            0.16     0.9129
      4       3  3.46            0.21     0.8683
      5       2  2.27            0.25     0.9345
```

Monotone, reproducible across seeds, and **hysteresis is exactly zero** — k = 2 reads 2.27
whether it is reached from below or from above, k = 3 reads 3.46 either way. Roughness
spans 0.865 to 0.935 across all six regimes and carries no signal.

Detection lag, in units of the estimator window **[computed]**:

```
change    median lag   in windows
  1->2          1199          0.6
  2->3           599          0.3
  3->4           399          0.2
  4->3          1799          0.9
  3->2          1599          0.8
```

The asymmetry runs opposite to what I expected when writing the file, and it matters for
this project: **rises are detected in 0.2–0.6 of a window, falls in 0.8–0.9.** A new
oscillator widens the local neighbour structure as soon as it appears; a fall cannot
register until the old, higher-dimensional data has left the window. **A criterion built
on a dimension *drop* therefore pays the worse half of this asymmetry — about one full
window of lag — and the published 3000-step window already exceeds most plateaus in this
project.**

---

## 3. Question 2 in detail

### 3.1 What the existing logs already settle

Chance accuracy for modular addition mod 113 is 1/113 = 0.00885; the validation batch is
512, so the last 2000 steps pool 102 912 trials **[computed]**:

* **`lowdata15` is 0.03× chance** — 34× *below* guessing, p ≈ 0. A model that has learned
  an actively anti-correlated map on unseen pairs is not one step away from generalising.
  Its trailing mean never reached even 1× chance at any point in the run. Real
  counterexample.
* **`lowdata20` is 1.56× chance**, p = 3.9e-55, Spearman +0.295 over the last half of
  training, and its maximum validation accuracy occurs at step 19 230 — 96 % of the way
  through the budget. It has learned something about unseen pairs, and it is still
  improving when the log stops. **This is a censored observation and the brief is right
  about it.**
* **`wd0` is 3.75× chance but exactly constant** — not a single example changes for 19 000
  steps. Spearman is undefined for zero variance, and a naive "above chance and not
  falling" rule would have promoted it to a late-grokking candidate, which is backwards.
  Frozen, and with a mechanism behind it: with no weight decay the norm never enters the
  region the Omnigrok account requires.

Milestones make the gap concrete. The runs that generalised passed 1×, 2×, 4×, 10× and
50× chance in order. `lowdata20` reached 1× and stopped; `lowdata15` reached nothing;
`wd0` reached 2× at step 1080 and never moved again.

### 3.2 What follows for every false-alarm number in this project

Of three controls, one is censored. The consequences are specific:

* The MG-drop conjunct's "2 of 3 false alarms" at most settings is really **1 confirmed
  false alarm (`lowdata15`) and 1 unresolved**.
* The one cell reported as "0 of 3" is unchanged in substance but rests on two usable
  controls, not three.
* A Jeffreys interval on a false-alarm rate from 2 usable controls runs to 0.67.
  **Specificity in this project is currently unmeasured, not measured and poor.** That is
  a weaker and more accurate statement than the previous report made.

### 3.3 What would settle it

`extend_runs.py --plan` prints twelve commands: three configurations plus a positive
control, three seeds each, 200 000 steps — 10× the current budget and 16× the longest
observed gap. About 7 GPU-hours. Readings are pre-registered in the file so the result
cannot be reinterpreted afterwards. The positive control is not optional: if `mod_wd1`
fails to grok at 200 000 steps the harness is broken and no negative result means
anything.

---

## 4. A replacement criterion, and why it is only a hypothesis

If the MG estimate on `weight_norm` is largely a re-encoding of that series' shape, the
norm itself should do at least as well. It does. Rule, one line, causal, no validation
labels, no embedding:

> fire at the first step after `t_mem` at which `1 − ‖w‖ / peak(‖w‖) ≥ x`,
> where `peak` is the running maximum since `t_mem`.

Evaluated on 14 runs — 3 grokking seeds, the 8 fixed-configuration seeds in
`../edm_validation/results/conf/`, and the 3 controls **[computed]**:

```
    x   recall   FP (all 3)   FP (uncensored 2)   median lead   min lead
   5%    11/11          2/3                 1/2          1230        720
  10%    11/11          2/3                 1/2           740        340
  15%    11/11          1/3                 1/2           370        160
  18%    11/11          0/3                 0/2           260         30
  20%     9/11          0/3                 0/2           240         60
```

Against the dimension conjunct it would replace, whose best cell was 8/11 at 0/3 with six
free parameters: **11/11 at 0/3 with two.** And the two are not independent measurements —
within-run Spearman between the sliding MG estimate and the drawdown has median |ρ| = 0.80
over 14 runs, so the embedding machinery is largely re-encoding the drawdown.

**Three reasons this is a hypothesis and not a result**, and they should be stated
wherever the rule is:

1. **The margin is one percentage point.** The smallest drawdown any generalising run
   reaches before `t_gen` is 18.4 %; the largest any control ever reaches is 17.3 %. On
   11 runs against 3, that is a threshold sitting between two runs, not between two
   families — the same criticism this project levelled at the MG cell, and it applies here
   unchanged.
2. **The minimum lead is 30 steps.** A warning that can arrive 30 steps before the event
   is not a precursor in any useful sense, even though the median is 260.
3. **One of the three controls it is scored against is censored** (§3.1), so the 0/3 is
   really 0/2 with one unknown.

It also has a mechanism rather than being curve-fitted — on the Omnigrok account
generalisation follows the weight norm decaying out of the region favouring the memorising
solution — which is a reason to pre-register it, not a reason to believe it yet.

---

## 5. What this changed in the previous reports

All of the following have been **applied**, not merely proposed.

| document | change |
| --- | --- |
| `../prediction_improved/report_0708.md` §7 | Tier 3 recorded as largely done: this study is the positive control with a countable ground truth that E3.1/E3.2 asked for, so the strong claim is no longer blocked by its absence — only by coverage |
| `report_0708.md` §9.0 | correction added: `lowdata20` is censored, so "2 of 3 false alarms" is 1 confirmed and 1 unresolved, and specificity is unmeasured |
| `report_0708.md` §9.0 | the weight-norm drawdown named as the one statistic that does separate `grok` from `lowdata15`, with its 1.1-point margin and 30-step minimum lead attached |
| `report_0708.md` §11 | new permitted claims 13 and 14: the controlled-dimension recovery result, and `lowdata15` as a genuine counterexample |
| `report_0708.md` §12 | new prohibitions 14–16: no general "a 1-D log cannot recover dimension", no specificity or false-alarm number until the extension runs, no claim that the drawdown rule works |
| `report_0708.md` §14 | new item 0, blocking: `extend_runs.py`. Item 7 added: pre-register the drawdown rule |
| `report_0708_experiments.md` §0, §12 | specificity relabelled from "poor" to "unmeasured" |
| `report_0708_experiments.md` §1.3, §1.4 | the two corrections above added as corrections, ahead of the results they revise |
| `report_0708_experiments.md` §3 | the coverage precondition given its matched-bandwidth control, so the failure is specific to coverage rather than to smoothness |
| `report_0708_experiments.md` §9 | per-run table annotated with censoring status; false-alarm counts restated over 2 usable controls |
| `report_0708_experiments.md` §14 | reordered — `extend_runs.py` is item 1 and blocking |

One finding has **not** yet been placed in either report, because it belongs in the paper
rather than in an audit: the dimension-drop criterion pays the slow half of the detection
asymmetry (§2.4) — about one full window of lag on a fall, against 0.2–0.6 of a window on
a rise.

## 6. Order of work

1. **`extend_runs.py`** — 7 GPU-hours, and it is blocking. Until it runs, no specificity
   number in this project can be quoted, including the one in §4.
2. **Pre-register the drawdown rule at x = 18 %** on the extended runs and on any new
   configuration, before looking at the result. It is currently a one-percentage-point
   margin fitted on 14 runs.
3. **Put §1.1 and §2.3 in the paper.** The bandwidth confound plus the `sync_phased`
   control is the cleanest demonstration available that the geometry is measuring
   something smoothness is not — and it is also the cleanest demonstration of why the
   training logs are the wrong input for it.
4. **Do not extend the synthetic study.** §1.8 bounds what it can establish; the next
   real question is the ensemble design in `report_0708.md` §7 Tier 6.
