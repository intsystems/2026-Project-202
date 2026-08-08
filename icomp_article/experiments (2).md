# Experiments — 2026-08-07

Results of the two checks asked for after the audit: the section 7 hypotheses of
[`report_0708.md`](report_0708.md), and the composite criterion *MG drop AND V_t above a
floor*. Everything here is new measurement on data already in the repo plus synthetic
systems whose dimension is known by construction. Nothing needed a GPU.

```
python exp_dimension_of_what.py        # part I, about 2.5 minutes
python exp_composite_criterion.py      # part II, about 3 minutes
```

Transcripts and CSVs are written to `results/audit/`. Markers as before: **[computed]**
recomputed here, **[repo]** taken from a result file, **[source]** verified against the
primary paper, **[open]** not established.

**Revised 2026-08-07** after the controlled-dimension study in
[`../dimension_recovery/`](../dimension_recovery/README.md), which tested this document's
two central claims against systems whose dimension is known by construction and against
the question of whether the controls are genuine negatives. Two things changed and both
weaken what is claimed here: no general statement that a scalar log cannot recover a
dimension is permitted (§1.3), and the specificity of the drop conjunct is **unmeasured**
rather than measured and poor, because one of the three controls is right-censored
(§1.4). Sections 0, 3, 9, 12, 13 and 14 carry the consequences.

Four of this document's findings came out **against** the expectation held when the
relevant experiment was designed. They are reported first, as corrections, because they change what can be said.

---

## 0. Summary

| | question | answer |
| --- | --- | --- |
| **I** | Is the estimator sound? | **Yes.** It recovers a true 2 → 1 collapse to ratio 2.04–2.82 against a truth of 2.0, and the MG correction is unbiased to d = 2. The machinery is not the problem. |
| | Is it invariant under reparameterising the observable? | **Yes**, to 1.01× median, 1.11× worst — including warps that curve at window scale. The audit expected this to fail. It does not. |
| | Does it agree across *different* observables of one run? | **No.** 3.6× to 10.9× disagreement at the same instant, traces uncorrelated (ρ −0.44 to +0.48). |
| | So what is it a property of? | The **roughness of the individual series**: Spearman(d̂, std(Δx)/std(x)) = **+0.934** across all 18 run × observable cells. |
| | Does the claim survive the estimator's free parameters? | **No.** Over 72 settings, Cohen's d is positive in 35 and negative in 37. |
| | Can *any* scalar observer recover a known dimension? | **Yes** — six do, with Spearman +0.94 to +1.00, and they pass the one-dimensional controls that roughness cannot. But they need several hundred oscillations per window; `weight_norm` windows have none. §1.3. |
| **II** | Does a dimension drop exist inside the plateau? | **Yes at a 600-step window** — 41–57 % below the running peak in 8 of 11 generalising runs. **Not measurable at the published 3000-step window**, where 5 of 11 generalising runs afford no causal window at all and four more afford fewer than 20. |
| | Does the drop conjunct discriminate? | **Recall yes** (10/11 at window 600). **Specificity is unmeasured**: of three controls one (`lowdata15`) fires and is a real counterexample, one (`wd0`) is silent, and one (`lowdata20`) is a censored observation that may not be scored. §1.4. |
| | Does the V_t floor fix that? | **No.** Over 180 parameter combinations it removes a false alarm in **1** and costs a detection in **6**. It is a no-op where it matters. |
| | Is the lead time a prediction? | **No.** Correcting for survivorship, ρ(delay after t_mem, gap) falls from +0.88 to **+0.27**. |
| | Is there anything better? | The weight norm's own causal drawdown: 11/11 at 0/3 with two free parameters against the drop conjunct's 8/11 at 0/3 with six — but on a **1.1-point margin**, so a hypothesis to pre-register, not a result. §12. |

---

# Part I — what the dimension estimate is a property of

## 1. Corrections to the audit

**1.1. The reparameterisation test passes. The audit implied it would not.**
`report_0708.md` §6.4 lists change of observable as the theorem's own signature and treats
disagreement as evidence against the geometric reading. Half of that signature holds
cleanly. Feed the same series through six diffeomorphisms of its range — including two
whose curvature is tuned to the *within-window* range, so they cannot be absorbed into a
local affine map — and the estimate does not move **[computed]**:

```
spread of the same quantity across reparameterisations of the same series
  near-affine warps (x^2, x^3, log)   median 1.03x   worst 5.76x   2 of 12 series > 1.2x
  window-scale warps (wiggle L, L/3)  median 1.01x   worst 1.11x   0 of 12 series > 1.2x
```

Both outliers are in the near-affine half and both are `train_loss`: `wd0`, where the
series is numerical noise at the floor and the log warp changes 15.00 to 2.60, and
`grok_s2`, where the cube warp changes 6.02 to 10.50. Under the warps that actually test
the property, no series moves by more than 11 %. The estimate behaves as a diffeomorphism
invariant should. This is a real property of the statistic and the audit should credit it.

**1.2. The estimator resolves a known dimension change. The audit predicted it would
not.** §7 E0.1 states the prediction outright — "it will not resolve the change. If so,
that fixes an upper bound on every claim." Measured on a 2-torus whose second amplitude
is ramped to zero over a known interval, at the published window, stride, k and E
**[computed]**:

```
 cycles/win    W  d before  d after   ratio   true   verdict
        5.7    0      3.19     1.44    2.21    2.0   resolves it
        5.7   14      5.68     1.44    3.94    2.0   overshoots
       20.3    0      3.39     1.20    2.82    2.0   resolves it
       50.9    0      2.93     1.44    2.04    2.0   resolves it
```

The prediction was wrong. The estimator finds a genuine 2 → 1 collapse, at the settings
the paper uses, to within the calibration error of §2 below. **The upper bound the audit
hoped to establish does not exist**, and the case against the published analysis has to
be made on the precondition (§3) rather than on the estimator's resolving power.

*(An earlier version of this table reported the estimator moving in the wrong direction.
That was an artefact of my own synthetic signal: I chose an integer number of cycles per
window, which makes the sampled orbit exactly periodic, so the delay embedding contains
exact repeats and the neighbour distances collapse to the noise floor. With irrational
cycle counts the artefact disappears. The same bug affected the first draft of §3.)*

**1.3. No general claim that a 1-D log cannot recover a dimension is permitted, and an
early draft of this document came close to implying one.** On systems whose dimension is
known by construction — a diagonal matrix of which exactly k entries oscillate at
rationally independent frequencies — six scalar observers recover k = 1..6 with Spearman
+0.94 to +1.00: `norm_fro`, `norm_fro_sq`, `trace`, `logdet`, `norm_l1` and one of four
random projections **[computed, `../dimension_recovery/`]**. The controls pass too: four
coordinates driven from one phase with four different fixed offsets read 1.22 against 4.19
for four genuine oscillators, while the roughness ratio reads 0.86 against 0.90.

So the geometry does work that smoothness cannot do, and the correct claim is conditional
rather than general. The condition is the one §3 already identifies, now measured on the
other side of the boundary: recovery needs *several hundred* oscillations of the
observable inside one estimator window and a signal-to-noise ratio above about 10³. A
`weight_norm` window in this project contains none. **The training logs are outside the
working regime; scalar observers as such are not.**

**1.4. One of the three controls is a censored observation, not a negative.** Chance
accuracy for modular addition mod 113 is 1/113; pooled over the last 2000 steps
(102 912 validation trials) **[computed, `../dimension_recovery/exp3_censoring.py`]**:

```
run          last-2k val_acc   x chance          p   trend (last half)   verdict
lowdata15            0.00023      0.03x      ~0       rho -0.07          real counterexample
lowdata20            0.01379      1.56x    3.9e-55    rho +0.29          CENSORED
wd0                  0.03320      3.75x      ~0       exactly constant   frozen
```

`lowdata15` sits 34× *below* guessing and never reached even 1× chance at any point in the
run: a model that has learned an anti-correlated map on unseen pairs is not one step from
generalising. `wd0` does not change by a single validation example in 19 000 steps.
`lowdata20`, though, is above chance, still rising, and its maximum validation accuracy
falls at step 19 230 — 96 % of the way through the budget. Its event time is known only to
exceed 20 000, and it may not be scored as a negative until
`../dimension_recovery/extend_runs.py` has been run.

Every false-alarm count in Part II was computed treating all three as negatives. Where
this document says "2 of 3 false alarms" the accurate statement is **1 confirmed and 1
unresolved**.

## 2. The estimator's own calibration — E0.2 [computed]

Uniform i.i.d. samples from the unit d-cube: the estimator under its own assumptions.

```
 true d      n    LB k=5    MG k=5   MG k=10   MG k=20   MG(k=5)/true
      1    300      1.37      1.01      1.00      1.00         1.01x
      2    300      2.55      1.95      1.91      1.91         0.97x
      3    300      3.66      2.78      2.79      2.69         0.93x
      5    300      5.84      4.43      4.33      4.23         0.89x
      8    300      8.58      6.52      6.43      6.36         0.81x
```

Three things worth carrying into the paper:

1. **The MacKay–Ghahramani correction is exactly right.** At d = 1 raw LB at k = 5 returns
   1.33–1.38 and MG returns 1.00–1.01. The 1.333 factor is (k−1)/(k−2), as derived in
   `../grokking_analysis/README.md`, and it is now measured rather than only argued.
2. **An LB reading of 1.33 means d = 1** — a curve — and not "just above one dimension".
   Every W = 0 trace in the report that sits at 1.3–1.9 is reading a curve.
3. After the correction the estimator is unbiased to d = 2 and biased **down** from d = 3,
   by 7 % at d = 3 and 19 % at d = 8 at n = 300. Any high reading is a lower bound.

## 3. The precondition that actually fails — E0.1a [computed]

Same 300-sample window, same k = 5, max_E = 15. The only thing varied is how many
characteristic periods the window spans. Cell format is LB at W = 0 / LB at W = 14.

```
system     true d   LB=          0.3          1.1          2.3          5.7         11.3
circle        1.0  1.33  2.44 /18.02  1.47 /15.00  1.40 / 1.01  1.44 / 1.44  1.46 / 1.46
2-torus       2.0  2.67  1.55 /15.00  1.65 /15.00  1.90 /12.65  3.19 / 5.24  3.59 / 2.99
lorenz       2.06  2.75  1.39 /15.00  1.39 /15.00  1.72 /15.00  2.87 /16.19  3.01 / 3.24
```

Below roughly five cycles per window, a 2-torus and the Lorenz attractor and a circle all
return the same number, 1.4–1.9, and it is the tangent constant. Above it, all three
return their true dimension to within the §2 calibration.

**A 300-sample window of `weight_norm` contains no cycles at all** — the series is
monotone across the window. It is in the leftmost column by construction, and no choice
of k, E or W moves it out. This, not the estimator's resolving power, is the defect.

This boundary has since been mapped from the other side, on systems built so that the true
dimension is known and can be varied **[computed, `../dimension_recovery/exp1_recovery.py`]**.
Two results sharpen the paragraph above.

*The failure is coverage, not smoothness.* One might object that a dimension estimate
which improves with more oscillations per window is simply reading bandwidth. It is not,
and the control that shows it had to be built deliberately: if the k frequencies are
chosen the obvious way, so that adding an oscillator also adds a higher frequency, then
the roughness ratio std(Δx)/std(x) recovers the ordering of k with Spearman **+1.00** and
the experiment is worthless. Holding all k frequencies inside a fixed band drops roughness
to +0.26 while the MG estimate stays at +0.94. On the same footing, four coordinates
driven from a single phase with four different fixed offsets read MG 1.22 against 4.19 for
four independent oscillators, while roughness reads 0.86 against 0.90.

*The largest resolvable dimension is set by coverage, and there is no universally correct
setting.* At 10 cycles per window the estimate resolves k up to about 3 and saturates; at
300 it resolves up to 5; at 1000 it fails at low k instead, because a single sinusoid seen
over a thousand periods has near-exact recurrences. Seeing a k-torus locally needs returns
in all k directions, of order c^k points. So the honest form of the precondition is not
"more than five cycles" but *enough coverage for the dimension in question*, and the
training logs fail it by the widest possible margin — zero cycles, for any k.

The Theiler column carries a second finding that bears directly on the project's
history: at about one cycle per window or fewer, the exclusion returns 15.00–18.02 in
every one of the three systems — that is max_E. **Applying a Theiler window to a
sub-cycle window does not correct the estimate, it destroys it** — every genuine
neighbour is temporally adjacent, so what remains is the noise ball, whose dimension is
the embedding dimension. Both the plain and the Theiler readings of the original
dimension dynamics are therefore uninformative, but for opposite reasons: the first
returns the tangent constant, the second returns the embedding dimension, and neither
returns anything about the optimiser.

## 4. Different observables disagree — E1.1a [computed]

The half of the theorem's signature that fails. Three observables of one run, one
estimator, one window, one instant. Band = causal windows 1000–3000 steps after `t_mem`,
capped at `t_gen`.

```
window = 300 samples
  run          weight_norm   train_loss     val_loss  max/min  rho(wn,tl)  rho(wn,vl)
  grok                1.68         5.93         7.98      4.7x       -0.18       +0.01
  grok_s1              n/a          n/a          n/a       --        +0.06       -0.02
  grok_s2             1.65         8.02         3.03      4.9x       +0.48       +0.40
  lowdata15           1.86         7.38         2.71      4.0x       -0.42       -0.42
  lowdata20           1.68         6.30         3.17      3.7x       -0.18       -0.26
  wd0                 1.39        15.00         1.50     10.8x         --        +0.41

window = 100 samples: max/min 3.6x to 10.9x, same picture, and grok_s1 is measurable
```

Under Takens or either Stark theorem, two observables that both embed the same invariant
set give diffeomorphic images, hence bi-Lipschitz equivalent on a compact set, hence
equal box-counting, Hausdorff and correlation dimension. A 4.7× disagreement at one
instant is not a robustness problem; it is the failure of the property that would make
the number a property of the system.

Note the shape of the combined result from §1.1 and here. Invariance under
**reparameterising one observable** holds tightly; invariance across **genuinely
different observables** fails by a factor of five. Those two facts together are much more
informative than either alone, and they point at one explanation.

## 5. What the number actually tracks [computed]

Median LB estimate and median roughness ratio std(Δx)/std(x), 100-sample windows, all
six runs × three observables:

```
d_hat                                   roughness std(dx)/std(x)
           train_loss  val_loss  w_norm            train_loss  val_loss  w_norm
grok             6.22      4.41    1.84                 0.781     0.277   0.121
grok_s1         12.89     12.52    3.85                 1.476     1.579   0.223
grok_s2         12.05     13.00    6.52                 1.502     1.436   0.259
lowdata15        5.43      2.11    2.24                 0.719     0.172   0.168
lowdata20        4.15      2.55    2.61                 0.600     0.209   0.209
wd0             15.00      1.37    1.32                 0.800     0.010   0.001

pooled Spearman(d_hat, roughness) over all 18 cells: +0.934
```

This settles question A of the audit empirically and in the direction §6.2 predicted.
The estimate is a monotone function of the local noise-to-drift ratio of whichever
series is fed to it. It disagrees across observables **because their roughness differs**,
and the disagreement is entirely accounted for by that one number. `report.tex`'s hedge —
"smoothness is a strong correlate rather than a complete description" — was correctly
worded for the within-run comparison, where §6.2 of the audit measured 10–99 % explained.
Across observables the mapping is much tighter than within one.

## 6. Detrending, the missing null, and the parameter sweep

**E1.2, detrended [computed].** Replacing each window by its residual after a local
quadratic fit raises the level everywhere and improves the ordering — but the runs still
overlap, so this is a lead, not a result:

```
window = 100     raw band   detrended band
grok                 1.85             4.16
grok_s1              1.62             4.10
grok_s2              1.59             2.32     <- below lowdata20
lowdata15            1.82             2.73
lowdata20            1.79             3.22     <- above grok_s2
wd0                  1.37             1.69
                AUC 0.56          AUC 0.78
```

With three runs a side, AUC resolves only to 1/9, so 0.78 is "the ordering is not
inconsistent", not "it discriminates". Worth carrying into the ensemble design; not
worth a sentence in the paper.

**E1.5, the null the report is missing [computed].** Fit a quadratic, phase-randomise the
residual, recombine, 39 surrogates per window. Unlike IAAFT on a monotone series, this
null has power. The observed estimate sits **below** it in every run (median z −0.63 to
−17.93; 34–100 % of windows outside the band). So the series is not simply trend plus
spectrum-matched noise — there is phase structure. But the effect is small where it is
significant (`grok` 2.24 observed against 2.88 null) and the rejection rate does not
separate the families (`grok` 63 %, `lowdata20` 63 %, `lowdata15` 34 %). `wd0` rejects in
100 % of windows on a difference of 1.33 against 1.34 — significance with no effect,
which is worth stating explicitly because it is the clearest example in the project of a
p-value that means nothing.

**E1.3, sensitivity [computed].** For each of 72 settings (max_E ∈ {5,10,15,20},
k ∈ {5,10,20}, W ∈ {0, embedding}, window ∈ {100,200,300}), the band median per run, then
Cohen's d between the families:

> **72 settings. |d| ≥ 0.8 in 23 of them. The sign is positive in 35 and negative in 37.**

A claim whose direction is a coin flip across the estimator's free parameters is not a
claim about the data. This is the strongest single sentence available for the paper's
falsification section, and it needs no theory at all.

## 7. Verdict on question A

The audit's three levels stand, with one correction and one strengthening.

- **Level 1, guaranteed.** Unchanged from `report_0708.md` §6.1. Stark 2003 Theorem 2.4
  is the right citation for minibatch SGD; d ≥ 2m+1 with m ≈ 3D is unmet at E = 15;
  membership of the residual set is unverifiable in principle **[source]**.
- **Level 2, estimated.** Now measured rather than argued: the local noise-to-drift ratio
  of the specific series, Spearman +0.934 across observables. The estimator itself is
  sound and correctly calibrated; it is being applied to windows containing no returns,
  where §3 shows every system returns the same constant.
- **Level 3, desired.** Unreachable by this route, for the reason in `report_0708.md`
  §2.4 — under stochastic forcing the delay vectors at different steps do not lie on a
  common manifold — and now also for the empirical reason in §4: the number is not
  invariant across observables, which it would have to be.

---

# Part II — the composite criterion

*Sustained MG-dimension drop, AND V_t above a floor.* Evaluated strictly causally: windows
labelled by their right edge, required to lie wholly inside the plateau, running peak and
running median from history only, no run-global normalisation anywhere.

**Evaluation set.** The velocity conjunct needs the logit probe, which exists for 7 runs.
The dimension conjunct needs only a training log, so it runs on 14: the 3 grokking seeds,
the 8 fixed-configuration seeds in `../edm_validation/results/conf/` that vary only split
and init (gaps 1600–5700), and the 3 never-generalising controls. The conf set is the
only place in the repo where eleven real plateaus can be tested against one rule.

## 8. Is there a drop to detect? [computed]

Largest fall below the running peak available to *any* threshold rule, over causal windows
lying wholly inside `(t_mem, t_gen)`:

```
                  window 3000        window 600
run           gen  n win  max drop   n win  max drop
grok            Y     91       45%     115       53%
grok_s1         Y      0        --      12        6%
grok_s2         Y      0        --      12       34%
conf_s0_i0      Y      7        4%      31       48%
conf_s0_i2      Y     17       27%      41       49%
conf_s1_i1      Y      0        --      10       41%
conf_s1_i3      Y     15       17%      39       55%
conf_s2_i0      Y      0        --      18       43%
conf_s3_i1      Y      0        --      11       21%
conf_s42_i0     Y      5        2%      29       44%
conf_s4_i3      Y     27       30%      51       57%
lowdata15       N    164       33%     188       34%
lowdata20       N    162       36%     186       35%
wd0             N    164        4%     188        8%
```

Two facts, both new:

- **At the published 3000-step window, 5 of 11 generalising runs afford no causal
  in-plateau measurement at all**, and four more afford fewer than 20 windows — so 9 of
  11 are effectively unmeasurable. This is the `report_0708.md` §5.1 finding, now
  established on eleven runs instead of three.
- **At a 600-step window the drop is real and large** — 41–57 % in 8 of 11 generalising
  runs. It is also 34–35 % in `lowdata15` and `lowdata20`, which never generalise.

Caveat that must travel with every 600-step number: 60 samples embedded in R¹⁵ leaves
46 points, which §2 shows is below where the estimator is reliable for d ≥ 3.

## 9. The drop conjunct, swept [computed]

Recall over 11 generalising runs, false alarms over 3 controls judged on an equally long
slice after `t_mem`:

```
window 600      drop  sustain   recall    FP   median lead
                 10%        3    10/11   2/3          1410
                 20%        3     9/11   2/3          1510
                 30%        3     9/11   2/3          1510
                 30%        5     8/11   0/3          1585
                 50%        3     3/11   0/3          2210
```

One cell — drop 30 %, sustain 5 — reaches 8/11 with no false alarm. Run by run:

```
run           gen    gap    d30/s1    d30/s3    d30/s5    d40/s5   status (section 1.6)
grok            Y  12070     +9410     +9210     +5910     +5810
grok_s1         Y   1750         -         -         -         -
grok_s2         Y   1740      +650      +450         -         -
lowdata15       N     --       FIRE      FIRE         -         -   real counterexample
lowdata20       N     --       FIRE      FIRE         -         -   CENSORED, not scorable
wd0             N     --          -         -         -         -   frozen
conf_s0_i0      Y   3650     +1710     +1510     +1310      +910
conf_s0_i2      Y   4650     +2260     +2060     +1860     +1560
conf_s1_i1      Y   1600      +660      +460      +260         -
conf_s1_i3      Y   4500     +2850     +2650     +2450     +1950
conf_s2_i0      Y   2330     +1110      +910      +710         -
conf_s3_i1      Y   1730         -         -         -         -
conf_s42_i0     Y   3460      +960      +760      +560         -
conf_s4_i3      Y   5700     +3040     +2840     +2640     +2440
```

The 0/3 column is one step away from the 2/3 column. Both firing controls fire at
sustain 3 and neither fires at sustain 5, which places the threshold between two controls
rather than between two families. And the sweep has 4 drops × 3 sustains × 3 windows = 36
cells; if each control fired independently with probability one half, about 4 cells in 36
would show 0/3 by chance. One did.

The "FP" column above must be read with §1.4 in hand. Of the three controls, only
`lowdata15` and `wd0` are scorable; `lowdata20` is censored. So the honest reading of the
sweep is **1 confirmed false alarm out of 2 usable controls at drop 30 % / sustain 3, and
0 out of 2 at sustain 5** — an interval on the false-alarm rate that a Jeffreys prior puts
at [0, 0.67]. Nothing here measures specificity; it only establishes that a genuine
counterexample exists and that at least one setting is silent on it.

## 10. The velocity floor is a no-op [computed]

Fraction of post-memorisation steps at which V_t stays above the given fraction of its own
running median:

```
run           gen    floor 10%   floor 25%   floor 50%    V median   V last/med
grok            Y        100%         99%         84%    1.34e-01         0.65
grok_s1         Y         97%         97%         93%    8.34e-02         0.54
grok_s2         Y         97%         95%         73%    1.22e-01         0.36
lowdata15       N        100%        100%         99%    5.62e-02         0.69
lowdata20       N        100%        100%         99%    7.86e-02         0.93
wd0             N         98%         74%         18%    2.57e-04         0.42
```

The floor is satisfied 95–100 % of the time in every run except `wd0`. The two controls
that matter satisfy it **more** consistently than the runs it is meant to protect.

Swept exhaustively over window × drop × sustain × floor **[computed]**:

> **180 parameter combinations. The velocity floor removes a false alarm in 1 of them,
> and costs a detection in 6. The single combination where it helps requires a floor at
> 90 % of the running median.**

The reason is structural and was already visible in `report_0708.md` §8.1: the only
control the floor can reach is `wd0`, whose velocity falls ~500×, and `wd0` is also the
only control whose weight norm is an exact straight line — so the drop conjunct never
fires there either. **The two conjuncts fail on the same run.** Neither reaches
`lowdata15` or `lowdata20`.

## 11. The lead time does not survive survivorship [computed]

Capping the detector at `t_gen` — necessary so it cannot read past the event — also
discards every run that would have fired late. Correlating lead time over the survivors is
circular. Letting the detector run to the end of training instead, so all 11 generalising
runs contribute a firing step:

```
                            capped at t_gen      uncapped
rho(delay after t_mem, gap)        +0.88           +0.27
rho(lead before t_gen, gap)        +0.95           +0.92     (forced: lead = gap - delay)
```

The apparent relationship between firing time and plateau length is survivorship. Once it
is removed, the delay after memorisation carries almost no information about the gap, and
the "median lead of 1500 steps" in §9 is a property of which runs happened to have long
plateaus.

A baseline with no dimension estimate in it — *declare at `t_mem` + Δ* — reaches 11/11
detections at Δ = 1500 with median lead 1960. It cannot reject a control, and that is the
entire remaining case for the dimension statistic: not when it fires, but whether it
stays quiet. Section 9 shows it does not.

## 12. Verdict on the composite criterion

**It is not the minimal cheap fix.** Stated precisely:

1. **The second conjunct is redundant, not merely weak.** 1 helpful combination against 6
   harmful ones over 180. It is net-negative at every sensible floor.
2. **The failure is structural, not a threshold choice.** V_t was proposed to exclude
   dimension drops caused by stabilisation or norm pumping. The only run in the repo that
   does that is `wd0`, and `wd0` is already excluded by the first conjunct, because
   pure norm pumping makes the weight norm a straight line and pins LB at its tangent
   constant. Any statistic that only catches `wd0` adds nothing to a rule that already
   catches `wd0`.
3. **The first conjunct is better than the audit implied, and still not usable.** At a
   600-step window the drop is real, large and reproducible across 8 seeds of a fixed
   configuration — a genuinely positive result, and the first thing in this project that
   reproduces across the conf set. But `lowdata15` does the same thing, so recall is not
   the problem and never was.
4. **The published 3000-step window cannot be repaired by tuning.** It is longer than the
   plateau in 5 of 11 generalising runs.
5. **Specificity is unmeasured, not measured and poor** — see §1.4. `lowdata20` is a
   censored observation, so of the three controls only two may be scored against, and a
   Jeffreys interval on the false-alarm rate from two controls runs to 0.67. This
   is a weaker statement than the one this document made in its first version, and it is
   the correct one.

**What the missing conjunct has to do.** It must separate `grok` from `lowdata15` — runs
that differ in whether they generalise but agree in weight decay, in norm trajectory
direction, in dimension trace and, per `report_0708.md` §5.3 and §8.3, in velocity and in
straightness.

One statistic now does, and it is not a dimension. The weight norm's causal drawdown below
its own running post-memorisation peak reaches 18.4–36.8 % before `t_gen` in all 11
generalising runs, and never exceeds 17.3 % in any control **[computed,
`../dimension_recovery/exp4_criterion.py`]**. At a threshold of 18 % that is 11/11 recall
at 0/3 false alarms with two free parameters, against the drop conjunct's best cell of
8/11 at 0/3 with six. It is also largely the same measurement: within-run Spearman between
the sliding MG estimate and the drawdown has median |ρ| = 0.80 over 14 runs, so the
embedding machinery is re-encoding the norm rather than adding to it.

Three reasons that is a pre-registrable hypothesis and not a result, and they must travel
with it: the separating margin is **1.1 percentage points** (18.4 % against 17.3 %) on 11
runs against 3, which is the same criticism §9 makes of the MG cell; the minimum lead is
**30 steps**, so the warning can arrive too late to be a warning; and one of the three
controls it scores 0/3 against is censored.

---

## 13. What this changes in `report_0708.md`

| section | change |
| --- | --- |
| §1 | add 1.1 and 1.2 above: the reparameterisation test passes, and E0.1's prediction was wrong |
| §6.2 | strengthen — the roughness reading is now measured across observables at ρ = +0.934, not argued |
| §6.4 | split the invariance signature in two: reparameterisation holds, change of observable fails |
| §7 | mark E0.1, E0.2, E1.1, E1.2, E1.3, E1.5 as done; E0.3 done in part; E0.4 and Tiers 3, 5, 6 still open |
| §9 | record that the composite as specified is net-negative, with the 1-in-180 figure |
| §10 | keep the analysis, add that the floor is redundant against the drop conjunct specifically |
| §12 | add: "the estimator cannot resolve a dimension change" is **not** permitted — §1.2 above; and neither is "a 1-D log cannot recover a dimension" — §1.3 |

Added 2026-08-07, after the controlled-dimension study in `../dimension_recovery/`:

| section | change |
| --- | --- |
| §0, §12 | specificity is **unmeasured**, not measured and poor — one of three controls is censored (§1.4) |
| §1.3 | no general claim that scalar observers cannot recover dimension; six of them do |
| §3 | the precondition is coverage, and the matched-bandwidth control rules out the smoothness explanation for it |
| §9 | false-alarm counts relabelled: "2 of 3" is 1 confirmed and 1 unresolved |
| §12 | a candidate replacement exists — the weight norm's own drawdown — at 11/11 / 0/3 on a 1.1-point margin |

## 14. What is now worth doing, in order

1. **`../dimension_recovery/extend_runs.py`** — about 7 GPU-hours, and it is blocking.
   Twelve runs at 200 000 steps (10× the current budget, 16× the longest observed gap):
   three configurations at three seeds each, plus a positive control that must still grok
   or the sweep is void. Until it has run, **no specificity number anywhere in this
   project may be quoted**, including the 0/3 in §12. Readings are pre-registered in the
   script so the outcome cannot be reinterpreted afterwards.
2. **Use the 72-setting sign flip in the paper.** It needs no theory, no new runs, and it
   is the cleanest falsification available. One table.
3. **State the precondition result** (§3) beside it, now with its control: below the
   coverage threshold every system returns the tangent constant whatever its true
   dimension, `weight_norm` windows are at zero cycles, and the matched-bandwidth
   comparison shows this is coverage rather than smoothness. That is a positive
   contribution rather than a retraction.
4. **Pre-register the drawdown rule at 18 %** (§12) on the extended runs and on any new
   configuration, before looking at the outcome. Fitted on 14 runs with a 1.1-point
   margin, it is exactly the kind of rule this project has already been burned by.
5. **Stop proposing conjuncts computed from the same six logs.** §12 point 4 remains the
   binding constraint.
6. **Tier 6 ensemble sweep**, unchanged from `report_0708.md` §7. It is still the only
   route to a claim about dimension that does not depend on coverage, and §5 above
   sharpens why: on an ensemble at fixed t the sample is i.i.d. by construction, so the
   roughness confound that explains 87 % of the rank variance here does not exist.
7. **[open]** E0.4, the eigenvalue multiplicity check, is still unmeasured and still
   cheap relative to a training sweep.
