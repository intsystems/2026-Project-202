# Critical review — 2026-08-08

Answers to the six points raised, with the three claims kept apart throughout.

```
python exp5_criterion_v2.py     # the drop criterion: defects, revision, calibration
python exp6_absolute.py         # does any metric recover k numerically?
python exp7_broader.py          # all 25 runs in the repo, four tasks, five controls
python make_figures.py          # figures/fig1, figures/fig2
sh launch_extended.sh           # 120 000-step reruns of the controls (CPU, ~90 min)
```

> **The extension runs have landed, and they overturn the central label.**
> `launch_extended.sh` reran the controls at 120 000 steps. **`lowdata15` generalises**
> — at step 110 940, in one of three seeds, with a second seed at 0.94 validation
> accuracy and still climbing. **`lowdata20` generalises at 39 600.** The positive
> control groks at 5 110, so the harness is sound. Consequences are worked through in
> `exp8_extended.py` and folded into §0, §1.5, §6.2 and §7 below; two claims made before
> the runs landed are **retracted**, one of them the "rise, not fall" separation of §6.3.
> Sections 2 to 5 are unaffected — they are about the estimator and the criterion family,
> not about the labels.

---

## 0. The three claims, separated

| | claim | verdict | the number that decides it |
| --- | --- | --- | --- |
| **1** | the metric **responds** to a change in the number of active components | **supported** | Spearman(MG, k) = +0.92 pooled over 8 nuisance factors; and MG reads 1.22 for *k* coordinates driven from one phase against up to 4.65 for *k* independent ones, while the roughness null reads 0.86 vs 0.90 |
| **2** | the metric **estimates their absolute number** | **not supported** | mean bias is −0.02 but the bias *moves by 2.14 components* across the estimator's own free parameters; 1.34 of that comes from coverage alone, which is unknowable for a training log |
| **3** | the dimension drop is a **specific early predictor** of grokking | **not supported; specificity has never been measured, and the evidence against it was itself invalid** | the two runs the criterion was scored as false-alarming on, `lowdata15` and `lowdata20`, both generalise at a longer budget. They were not false alarms. Nor were they negatives: **not one of the five controls in this repository is established as a negative** (§6.2) |

**Retracted before publication.** An earlier draft of this document claimed a fourth
result — that what separates the families is not a fall but a *rise*, generalising runs
excursing to 3.2–18.0 against 2.37 for controls. It does not survive. The 12–18 values
came from a statistic that failed to cap at `t_gen` and so measured post-generalisation
windows, where the weight norm flattens and the estimate degenerates towards `max_E`.
Capped correctly and scored on the seven held-out extended runs, the grokking ones reach
p95 1.79–2.38 and the non-grokking ones 1.39–1.72, with `lowdata15_s0` (groks) at 1.79
against `lowdata15_s1` (does not) at 1.72 — the same configuration, 11 000 windows each.
The details are in `exp8_extended.py` §3 and the retraction is kept in §6.3.

---

## 1. `lowdata15`

### 1.1 The mean dimension does not fall — this is correct

Measured on the whole post-memorisation trace, 600-step window, MG, stride 10
**[computed]**:

```
median 1.54   mean 1.58   min 1.31   max 2.13   sd 0.13
Spearman(d, step) = -0.148        Theil-Sen slope = -0.003 per 1000 steps
```

Over 18 600 steps the level falls by about 0.06 in total. There is no trend to speak of.
Any description of this run as "the dimension drops" is a description of one excursion,
not of the run.

### 1.2 The trigger is not a single-sample outlier, but it is not a drop either

The peak that sets the running maximum is at step 3200, d = 2.13. It is not a spike: the
trace rises smoothly 1.62 → 2.13 over about 100 steps and comes back down over the next
100. The median of the twenty windows around it, excluding the peak itself, is 1.86.

So "caused by noise or an outlier" is not right as stated, and I should say so plainly.
What is right is the more damaging version: **the trace is a flat band with excursions,
and the rule fires on any excursion followed by a return.** In 18 600 steps this run
produces excursions to 2.13, 2.08, 2.01, 1.99, 1.94 and 1.85 (figure 1, right panel).
The rule needs only one.

### 1.3 The comparison that had been made was not a fair one

An earlier draft of this analysis reported a 14 000-step separation between the peak and
the trigger in `lowdata15`, against a few hundred steps in the positives. That was my
error: I scanned the control over its whole run and the positives only up to `t_gen`. At
a matched scan window the separation is 190 steps in `lowdata15` and 120–600 in the
positives — indistinguishable.

The same applies to the trace's variability. Drawn on its own axis, `lowdata15` looks far
noisier than a generalising run; at matched duration the standard deviations are 0.133
and 0.121. Figure 1 is drawn with matched axes for that reason.

### 1.4 The budget asymmetry is real and it is the mechanism

```
run              post-memorisation budget    delay to first trigger (running-max rule)
conf_s3_i1                    1 160                     370
grok                         12 070                   2 660
lowdata15                    19 330                   2 730
lowdata20                    19 160                   3 130
p_211_wd_0                  198 200                  28 100
```

A control is scanned for 19 330 steps and a positive for 1 160. Within its *first* 1 200
post-memorisation steps `lowdata15` does not fire at all — the false alarm arrives at
2 730. **A false-alarm count per run is therefore not a false-alarm rate.** The right
metric, which `method.md` already asked for and nothing in this project has reported, is
false alarms per 1 000 steps of control budget.

### 1.5 Is it a legitimate counterexample? No — it generalises

**Settled by measurement [computed, `exp8_extended.py`].** Three seeds of the `lowdata15`
configuration at 120 000 steps:

```
run             t_mem    t_gen      gap    x chance (last 10k)   max val   outcome
lowdata15_s0      620   110940   110320                 111.93    1.0000   GROKS
lowdata15_s1      630        -        -                  60.33    0.9414   still climbing
lowdata15_s2      660        -        -                   0.65    0.0156   no
lowdata20_s0      800    39600    38800                 112.97    1.0000   GROKS
grokpos_s0       1450     5110     3660                 113.00    1.0000   GROKS (control)
```

The configuration generalises. One seed reaches validation accuracy 1.0 after a plateau of
**110 320 steps** — nine times the longest gap previously seen in this project — and a
second is at 0.94 and rising when the budget ends. `lowdata20` generalises at 39 600. The
positive control behaves, so this is not a harness artefact.

**And the diagnostic I used to argue the opposite is refuted by direct counterexample.**
At step 20 000, `lowdata15_s0` — the seed that goes on to generalise — sits at 0.55× chance
with a *falling* trend (Spearman −0.417 over steps 10 000–20 000). That is precisely the
profile `exp3_censoring.py` labels `extended_non_grokking`, and precisely the argument I
made in the previous round: "34 times below guessing, never reached 1× chance, not a run
approaching a transition". **Being far below chance late in a truncated run carries no
information about whether it will generalise later.** `exp3_censoring.py` now carries that
refutation in its docstring; the only defensible label from a truncated run is *censored*.

So the answer to the question as posed: `lowdata15` is **not** a valid non-grokking
counterexample, and it never was. Scoring a criterion against it did not measure a false
alarm — it penalised the criterion for firing on a run that generalises.

---

## 2. The drop criterion

### 2.1 Why the running-maximum rule is not a drop detector

**Synthetic demonstration [computed].** Take a flat trace, 1.60 ± 0.05, 400 points, and
add one point:

```
spike +0.0    silent
spike +0.4    fires at index 303
spike +0.8    FIRES at index 51
spike +1.2    FIRES at index 51
```

A single sample 0.8 above the level is enough to arm a 30 % rule for the rest of the run.
Nothing about the trace changed.

**On the real traces this particular failure does not occur** — the largest and second
largest values are within 0.02 of each other in every run, so no single point is carrying
the trigger. That is a property of the weight norm being smooth, not of the rule being
safe. What the real traces do show is threshold fragility: replacing the running maximum
by the running 95th percentile — one order statistic more robust, same rule otherwise —
changes the verdict for 3 of 14 runs, silencing `lowdata15` and `s3_i1` and delaying
`s0_i2` by 510 steps.

The structural defect is the one in §1.4: the rule compares the present against the
best-ever, so its false-alarm probability grows with the length of the run.

### 2.2 The proposed criterion: four defects

$$R_t=1-\frac{\operatorname{median}\{d_s: s\in[t-H,t]\}}{\operatorname{median}\{d_s: s\in[t-2H,t-H)\}}\ \ge\ \delta,\qquad
\tilde b_t=\frac{H\,b_t}{d_{\mathrm{before}}(t)}\le-\beta_{\mathrm{rel}}$$

It is a large improvement: the reference is a median rather than an extremum, the two
intervals are adjacent so the fall is local by construction, and the hold condition
rejects a single dip. Four defects, in decreasing order of importance:

* **D1 — `Hold` reads the future.** The condition on $[t_0, t_0+Q]$ cannot be evaluated
  at $t_0$. The criterion *fires* at $t_0$ but is only *decidable* at $t_0+Q$, so any
  lead time must be measured from $t_0+Q$. At $Q=300$ this is the difference between a
  reported median lead of 1 315 steps and an honest 1 015.
* **D2 — the two intervals are not independent unless $H \ge W$.** Each $d_s$ is itself
  computed on a window of $W$ raw samples. With $H<W$ the "before" and "after" medians
  are built from overlapping weight-norm data, so $R_t$ is damped toward zero and the
  Theil–Sen fit sees autocorrelated points. **$H \ge W$ is required and is not stated.**
* **D3 — no stability requirement on the baseline**, although the brief's own list asks
  for one. $d_{\text{before}}$ can be the median of a trace that is itself moving, in
  which case $R_t$ measures the continuation of a trend rather than a drop.
* **D4 — $R_t$ and the slope test are largely redundant**, being computed on the same
  interval and measuring the same fall. The slope earns its place only as a
  *monotonicity* check, which is what Theil–Sen supplies; it is not independent evidence.

### 2.3 The revised criterion

$$
\text{fire at }t \iff
\underbrace{\frac{\operatorname{MAD}(d_{\text{before}})}{\operatorname{med}(d_{\text{before}})}\le\sigma_{\max}}_{\text{stable baseline}}
\ \wedge\
\underbrace{R_t\ge\delta}_{\text{real fall}}
\ \wedge\
\underbrace{\tilde b_t\le-\beta_{\text{rel}}}_{\text{directed}}
\ \wedge\
\underbrace{\operatorname{med}\{d_s\}_{[t,t+Q]}\le(1-\delta_{\text{hold}})\operatorname{med}(d_{\text{before}})}_{\text{held}}
$$

with $H\ge W$, and the decision time reported as $t+Q$.

**Data budget.** The criterion needs $2H+W$ steps of plateau before it can be evaluated
at all. At $H=W=600$ that is 1 800 steps, which excludes `grok_s1` (gap 1 750),
`grok_s2` (1 740), `s1_i1` (1 600), `s2_i0` (2 330 — marginal), `s3_i1` (1 730) and
`nogap`. **Six of the twenty-five runs in the repository cannot be evaluated by any
member of this family.**

### 2.4 Which conditions actually do anything

One factor at a time from $H=600,\ \delta=0.20,\ \beta=0.10,\ Q=300,\ \delta_{\text{hold}}=0.15,\ \sigma_{\max}=0.08$ **[computed]**:

```
parameter      value   recall  false alarms  median lead   fires on
H                300    10/11           1/3          960   lowdata20
H                600     8/11           0/3         1015   -
H                900     4/11           0/3          285   -
delta           0.10     9/11           2/3         1010   lowdata15, lowdata20
delta           0.15     9/11           1/3          960   lowdata15
delta           0.20     8/11           0/3         1015   -
delta           0.25     7/11           0/3          380   -
beta_rel        0.00     9/11           0/3          510   -
beta_rel        0.20     8/11           0/3         1015   -
Q                  0     8/11           0/3         1315   -
Q                300     8/11           0/3         1015   -
Q                900     4/11           0/3          415   -
delta_hold      0.05     8/11           0/3         1015   -
delta_hold      0.25     8/11           0/3         1015   -
disp_max        0.06     7/11           0/3          510   -
disp_max        1.00     9/11           0/3          890   -
```

**Only $\delta$ and $H$ affect the false-alarm column.** At the operating point,
$\beta_{\text{rel}}$, $\delta_{\text{hold}}$ and $\sigma_{\max}$ change nothing, and $Q$
costs 300 steps of lead for nothing. $\delta_{\text{hold}}$ is inert for a structural
reason worth stating: while $\delta_{\text{hold}}\le\delta$ the hold-depth test is weaker
than the fall test it follows, so only the *persistence* part of `Hold` can ever bind.

This does not mean the conditions are worthless — they are insurance against failure
modes this dataset does not contain (§2.1 shows one). It does mean that on the available
evidence the five-condition criterion is a two-condition criterion, and that the extra
three should be justified by the failure mode they guard against rather than by
performance.

### 2.5 Calibration on positives only, then one look at the controls

With three controls, one of them censored, any threshold tuned to silence them is tuned
on two data points. So: **choose parameters using generalising runs only** — among the
540 grid settings reaching a target recall on the 8 `conf` runs, take the strictest —
then evaluate once on the held-out headline runs and the controls.

```
calibration target   chosen setting                                 test result
8/8   delta 0.15, beta 0.05, Q 300, hold 0.15, disp 0.08   recall 2/3, FALSE ALARM on lowdata15
7/8   delta 0.15, beta 0.20, Q 300, hold 0.20, disp 0.08   recall 2/3, 0 false alarms
6/8   delta 0.15, beta 0.20, Q 600, hold 0.20, disp 0.08   recall 1/3, 0 false alarms
```

**This is the central result of §2.** Under an honest protocol the criterion produces a
false alarm on `lowdata15`. The setting that avoids it differs only in
$\beta_{\text{rel}}$ and $\delta_{\text{hold}}$ — two parameters that §2.4 shows have *no
effect on the positives*. So the calibration data cannot distinguish the safe setting
from the unsafe one; which of the two you end up with is decided by an arbitrary
tie-break. **The specificity of this family is not determined by the data available.**

Test recall is 2/3 in every row because `grok_s1` fails the $2H+W$ budget test — not
because the criterion missed it.

---

## 3. Is it legitimate to call MG a *dimension*?

### 3.1 What the diagonal experiment does establish

$W(t)=\operatorname{diag}(b_i+\delta_i(t))$ with $b_i\ne0$ and exactly $k$ coordinates
oscillating at rationally independent frequencies. The orbit closure is a $k$-torus, so
the ground truth is exactly $k$. At the most favourable settings found **[computed]**:

```
metric          k=1     k=2     k=3     k=4     k=5     k=6    bias    MAE    MRE     sd    rho
MG             1.22    2.40    3.47    4.13    4.65    4.65   -0.08   0.49    15%   0.24  +0.94
LB             1.55    3.31    4.48    5.71    6.01    6.07   +1.02   1.02    39%   0.36  +1.00
TwoNN          0.66    2.42    3.22    5.01    5.04    4.74   +0.01   0.55    18%   0.25  +0.83
PR             1.99    2.55    2.75    2.63    2.55    2.70   -0.97   1.48    46%   0.13  +0.60
CorrDim        1.06    2.55    2.74    2.42    2.45    2.57   -1.20   1.40    32%   0.15  +0.37
roughness      0.86    0.92    0.87    0.90    0.88    0.89   -2.61   2.61    64%   0.04  +0.26
```

Three things are genuinely established, and the project should say so:

1. **MG responds to $k$**, with mean absolute error 0.49 components at $k\le5$ — much
   better than the earlier reports implied.
2. **MG separates $k$ independent oscillators from $k$ synchronised ones.** Driving $k$
   coordinates from a single phase with distinct fixed offsets gives 1.22 for every $k$
   (truth: 1), against 4.13 and 4.65 for four and six independent ones — a gap of 2.9 and
   3.4. The roughness null gives 0.86 against 0.90, a gap of 0.04. **The geometry is
   doing work that smoothness cannot do**, and this is the single strongest piece of
   evidence in favour of the method anywhere in the project.
3. **The LB → MG correction is exactly the $(k-1)/(k-2)$ factor**: LB's bias is +1.02 and
   MG's is −0.08 on the same data.

### 3.2 What it does not establish

* **It is not a model of training.** No optimiser, no feedback, stationary, and the $k$
  components are independent by construction. It is the easiest possible case, which
  makes a negative result decisive and a positive result weak.
* **It does not license reading an absolute number off a training log**, for the reason
  in §4: the bias is not stable.
* **It does not transfer the sync control to the real setting.** What is shown is that
  MG can tell $k$ independent *sinusoids* from one; whether it can tell $k$ independent
  optimiser directions from one is not tested by this.
* **It says nothing about coverage**, which the training logs fail outright: a
  `weight_norm` window contains zero oscillations, and §4 shows coverage is the largest
  single source of bias.

### 3.3 Why absolute calibration requires MG ≈ k *stably*

If $\hat d = g(k)$ for some fixed unknown monotone $g$, then $\hat d$ ranks configurations
and nothing more: any statement of the form "there are about four active directions"
requires $g \approx \operatorname{id}$, and a *stable* $g$, because the calibration must be
transferable from the system where it was measured to the system where it is used. If $g$
depends on the window, on $\tau$, on $E$, or on the sampling rate, then the number read
off a log is a number about the analysis, not about the system. §4 measures exactly this
dependence.

### 3.4 Alternative explanations that survive

1. **A bandwidth or roughness proxy.** Ruled out here, and it took work: with the obvious
   frequency construction (each new oscillator at a higher frequency) the roughness ratio
   recovers the ordering of $k$ with Spearman **+1.00**. Only when all $k$ frequencies are
   held inside a fixed band does roughness fall to +0.26 while MG stays at +0.94.
   Any future version of this experiment that omits the matched band proves nothing.
2. **A noise-to-drift ratio.** Not ruled out in general. It is ruled out as the *whole*
   story here, because SNR was varied over three decades with a bias range of 0.05.
3. **An amplitude-distribution effect.** Partly open: amplitude ×10 moves the bias by
   0.40, which is consistent with the quadratic term $\sum\delta_i^2$ becoming
   non-negligible relative to $2\sum b_i\delta_i$.
4. **Local curvature of the delay curve.** Not separated from dimension by any measurement
   here, and it is the explanation §6.2 of `report_0708.md` argues for on the real logs.

---

## 4. The metric pool: does anything recover $k$ numerically?

One factor at a time around the base configuration; systematic offset
$\operatorname{mean}(\hat d - k)$ per level, and its range **[computed]**:

```
factor      levels                        MG bias range   min rho over levels
cycles      30, 100, 300, 1000                    1.34            0.66
window      1000, 2000, 4000                      0.96            0.89
tau         1, 2, 3                               0.80            0.94
max_E       10, 15, 20                            0.60            0.94
amp         0.02, 0.1, 0.5                        0.40            0.94
k_neighbors 5, 10, 20                             0.29            0.94
SNR         1e3, 1e4, 1e6                         0.05            0.94
n           10k, 20k, 40k                         0.02            0.94
```

Pooled over every configuration tested:

```
metric       mean bias  mean MAE  mean MRE  worst MAE  bias range  mean rho
MG               -0.02      0.63       20%       1.43        2.14     +0.92
TwoNN            +0.10      0.72       25%       1.88        1.69     +0.74
LB               +0.99      1.14       40%       2.02        2.72     +0.92
PR               -0.94      1.53       47%       2.26        2.20     +0.25
roughness        -2.59      2.61       65%       3.41        1.88     +0.23
```

**Answer: no metric recovers $k$ stably.** MG comes closest and its *average* bias is
−0.02, which looks like calibration; but the bias travels over a range of 2.14 components
as the nuisance parameters move, and 1.34 of that comes from coverage alone — the one
parameter that cannot be set for a training log because it depends on the system's own
recurrence time. Ordering survives everything ($\rho\ge0.89$ except under coverage), which
is claim 1; absolute recovery does not, which is claim 2.

Across observers of the same system at matched settings the bias is consistent (−0.08 to
−0.50, MAE 0.49–0.85 for `norm_fro`, `trace`, `logdet` and two random projections). This
locates the 3.6×–10.9× disagreement seen on the *training* logs: it is not an intrinsic
property of the observable but the consequence of those observables sitting at very
different SNR and coverage.

Figure `figures/fig2_bias_vs_parameters.png` plots all of this; the grey band is
±0.5 components.

---

## 5. What would settle claim 2

Ordered by cost. Each has a stated outcome that would refute the claim.

* **E1 — a calibration surface, not a calibration point.** Measure $\hat d(k;\ c,W,\tau,E)$
  on the torus family over a full factorial in coverage $c$, window $W$, $\tau$ and $E$;
  fit $\hat d = g(k;\theta)$ and report the residual. *Refuted if* no
  $\theta$-parameterisation with fewer than four free parameters brings the residual below
  0.5 components. Hours, no GPU.
* **E2 — inversion.** Invert the surface: given $(\hat d, c, W, \tau, E)$, predict $k$ on
  systems held out from the fit. *Refuted if* the held-out MAE exceeds 1 component. This is
  the operational form of the claim and it is currently untested.
* **E3 — coverage is measurable, or it is not.** The blocking obstacle is that $c$ is
  unknown for a training log. Test whether $c$ can be estimated from the series itself
  (autocorrelation time, first minimum of delayed mutual information, dominant spectral
  peak) well enough to enter the inversion of E2. *Refuted if* the recovered $c$ does not
  reduce the E2 error. Without this, E1 and E2 are unusable on real data.
* **E4 — a non-torus ground truth.** Coupled oscillators with a known number of active
  modes, a deep linear network with a countable number of learned modes (Saxe et al.), a
  scheduled-rank parameterisation. *Refuted if* the calibration from the torus family does
  not transfer. This is the test of whether the calibration is about dimension or about
  tori.
* **E5 — the ensemble route, which sidesteps all of the above.** 200 seeds, intrinsic
  dimension of the 200-point cloud at fixed $t$. i.i.d. by construction, no delay
  embedding, no coverage requirement. ~11 GPU-hours. This remains the only design in this
  project in which the word "dimension" needs no qualification.

For claim 3 the blocking experiment is different and cheaper: **the extension runs**, plus
a false-alarm rate expressed per 1 000 steps of control budget rather than per run.

---

## 6. Wider evidence: 25 runs, four tasks, five controls

### 6.1 The evaluation set was much smaller than it needed to be

The repository contains 25 usable logs — modular addition mod 113 and mod 211, $S_5$,
$S_6$, full-batch and minibatch, three weight decays — of which 20 generalise and 5 never
do. Every criterion in this project has been scored against three controls from a single
task. The three unused controls are `ma_S_5_without_weight_decay` (11 250 steps of
post-memorisation budget), `p_211_wd_0` (**198 200 steps**) and the low-data pair.

### 6.2 Item 6: at WD = 0 the estimate has *not* fallen — it never rose

**Read with §1.5 in hand: the labels in the left column are no longer reliable.** Two of
these four "controls" generalise at a longer budget. What the table still establishes is
the shape of the estimate around memorisation, which does not depend on the label.

The question was whether the dimension might still be high at memorisation and fall later,
which would make WD = 0 a non-trivial control. It does not **[computed]**:

```
run                             t_mem   d before   d at t_mem   d after   max after
- wd0                             640       1.33         1.34      1.31        1.41
- ma_S_5_without_weight_decay    3745       1.34         1.35      1.35        1.39
- lowdata15                       660       1.33         1.32      1.54        2.13
- lowdata20                       830       1.33         1.34      1.75        2.37
```

At WD = 0 the estimate sits at 1.31–1.35 before memorisation and stays there: it is
already at the floor, so it cannot fall later. (`p_211_wd_0` cannot be measured before
`t_mem` at all — its 60-row window is 2 940 steps and `t_mem` is 1 800.)

### 6.3 RETRACTED: "the real signal is a rise, not a fall"

This section previously reported that generalising runs excurse to 3.2–18.0 after
memorisation while controls stay at or below 2.37, and called it the strongest result of
the round. It is withdrawn. Two independent defects, either of which is fatal:

**The statistic was contaminated.** `exp7_broader.py` took the maximum over everything
after `t_mem` *without capping at `t_gen`*. For a generalising run that includes
post-generalisation windows, where the weight norm flattens and the estimate degenerates
towards `max_E = 15`. The 12.8–18.0 readings were that artefact. They measured what
happens after the event, not before it.

**Capped correctly, it does not separate held-out runs** **[computed,
`exp8_extended.py` §3]**:

```
                       groks   windows   median    p95    max
previously used
  s1_i3                    Y       431     1.59   2.78   3.57
  grok                     Y      1188     1.86   2.60   3.09
  s3_i1                    Y       154     1.53   2.38   2.42
  lowdata20                N      1897     1.75   2.12   2.37
  lowdata15                N      1914     1.54   1.83   2.13
  wd0                      N      1916     1.31   1.36   1.41
held out (the extended runs)
  grokpos_s0               Y       347     1.61   2.38   2.86
  lowdata20_s0             Y      3861     1.58   1.97   2.56
  lowdata15_s0             Y     11013     1.51   1.79   2.30
  lowdata15_s1             N     11917     1.48   1.72   2.41
  lowdata15_s2             N     11914     1.49   1.69   2.12
  wd0_s1                   N     11917     1.30   1.39   1.59
```

`lowdata15_s0` generalises and reaches p95 1.79; `lowdata15_s1` does not and reaches 1.72.
Same configuration, different seed, 11 000 windows each. The gap that looked complete on
the runs the statistic was built from is 0.07 on runs it was not.

What survives is only the observation in the first column: every run in the repository
sits at the estimator's floor, 1.30–1.38, before memorisation. That is a statement about
the floor, not about grokking.

### 6.4 The rules on the full set, and how a false-alarm rate must be reported

At the §2.3 defaults, over the 16 runs long enough to evaluate **[computed]**:

```
rule                  hits            false alarms          fires on
revised (§2.3)        9 / 11          1 / 5 controls        p_211_wd_0, at 49 250 steps
running maximum      10 / 11          3 / 5 controls        lowdata15, lowdata20, p_211_wd_0
```

The single control that defeats the revised rule is the one with ten times the budget of
any other: `p_211_wd_0` is watched for 198 200 steps. That is §1.4 made concrete, and it
is why the count per run is the wrong statistic. Normalising by the 267 290 steps of
control budget actually observed:

```
rule                per 1000 steps    expected false alarms in a plateau of
                                      1 500     3 000     5 000    12 000 steps
revised                    0.0037     0.006     0.011     0.019     0.045
running maximum            0.0112     0.017     0.034     0.056     0.135
```

**These rates are void, and §1.5 is why.** Two of the five runs in the denominator
(`lowdata15`, `lowdata20`) generalise at a longer budget, so their firings were not false
alarms; the other three have never been run long enough to be called negatives. What the
arithmetic still establishes is the *form* the quantity must take — events per unit of
control budget, not per run — and the ratio between the two rules, which does not depend
on the labels: per unit time the running-maximum rule fires three times as often as the
revised one. Read against the observed positive plateaus (3 460 – 110 320 steps), even
that ratio is of limited use, because a 110 000-step plateau makes "fires before `t_gen`"
nearly free.

---

## 7. What I got wrong in this round

Ordered by consequence.

* **I argued that `lowdata15` was a genuine counterexample, and it is not.** The argument
  was that its validation accuracy sat 34× *below* chance and never reached 1× chance in
  20 000 steps, so it could not be a run approaching a transition. The seed that
  generalises at step 109 860 has exactly that profile at step 20 000 (§1.5). The
  reasoning was wrong, not just the conclusion: a truncated run below chance is still a
  censored observation, and I built a classification rule on the opposite assumption and
  put it in `exp3_censoring.py`. That rule now carries its own refutation.
* **The "rise, not fall" result of §6.3 was contaminated and is retracted.** The statistic
  read past `t_gen` into the post-generalisation regime, where the estimate degenerates
  towards `max_E`. On held-out runs the separation is 0.07 and inverted in sign for two
  seeds of one configuration.
* I first reported a 14 000-step separation between the running-maximum reference and the
  trigger in `lowdata15`, and read it as non-locality. It was an artefact of scanning the
  control over its whole run and the positives only to `t_gen`. At matched scan windows
  the separation is 190 steps against 120–600 (§1.3).
* I first drew figure 1 with unmatched time axes, which made the control look far noisier
  than a generalising run. At matched duration the trace standard deviations are 0.133 and
  0.121.
* My first `exp7` normalisation of the Theil–Sen slope dropped a factor of the logging
  stride, so the revised rule fired nowhere. Fixed; the numbers in §6.4 are post-fix.

The common thread in the first two is that both were measured on the runs that motivated
them and neither had a held-out set. The extension runs cost seven CPU-hours and refuted
both.

## 8. What to do next, in order

1. **Extend the remaining three controls** — `wd0` beyond 120 000 steps, and
   `ma_S_5_without_weight_decay` and `p_211_wd_0`, which have never been rerun. Until a
   control is shown not to generalise at a budget comparable to 110 320 steps, it is a
   censored observation and cannot appear in a specificity figure. **Blocking for
   claim 3, and now known to be blocking rather than suspected.**
2. **Report false alarms per 1 000 steps of control budget**, never per run (§1.4), and
   lead time as a fraction of the gap, never in steps (§6.4).
3. **Re-score every earlier result** whose denominator contained `lowdata15` or
   `lowdata20`: `report_0708.md` §9.0 and §11, `report_0708_experiments.md` §1.4, §9 and
   §12, and §2.5 of this document.
4. **E3 then E2** from §5 — whether coverage can be recovered from the series decides
   whether claim 2 is testable on real data at all.
5. **Do not add conditions to the drop criterion** without naming the failure mode each
   one guards against; §2.4 shows three of the five currently do nothing.
6. **Hold out runs before measuring anything on them.** Both retractions in §7 would have
   been caught by splitting the runs before the statistic was chosen, which costs nothing.
