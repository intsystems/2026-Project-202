# Audit — 2026-08-07

What we are measuring, what the embedding theorems do and do not license, and what
can honestly be claimed. Written against `../../icomp_v2/report.tex` (the current
report), `../grokking_analysis/README.md` (the estimator audit), `method.md` (the
criterion), and the phase 1–15 results in `../edm_validation/`.

## 0. Status markers

Every factual claim below carries its provenance. Nothing is asserted from memory.

| marker | meaning |
| --- | --- |
| **[computed]** | recomputed from repo data during this audit; script and numbers given |
| **[repo]** | taken from a repo result file, checked for internal consistency |
| **[source]** | verified against the primary paper, quoted |
| **[open]** | not verified; stated as a question, not a fact |

Primary sources consulted in full: Stark, *Delay Embeddings for Forced Systems. I.
Deterministic Forcing*, J. Nonlinear Sci. **9** (1999) 255–332; Stark, Broomhead,
Davies, Huke, *Delay Embeddings for Forced Systems. II. Stochastic Forcing*,
J. Nonlinear Sci. **13** (2003) 519–577. Levina–Bickel (2004) and the
MacKay–Ghahramani note were checked at the level of their stated assumptions; the
MG algebra is re-derived below and does not depend on the source. Sugihara et al.
(2012) supplementary material was **not** obtainable — see §9.3.

---

## 1. Corrections to my own earlier statements

Listed first because they change the conclusions.

**1.1. The sample-budget criticism was wrong.** I claimed the report's "ten
independent observations in the transition" is a one-draw number and that the
canonical run, whose gap is 7× longer, would give ~70. That assumed the correlation
time is constant across runs. It is not. Measured (1/e crossing of the train_loss
ACF inside `(t_mem, t_gen)`, on the 10-step-resolution logs) **[computed]**:

```
run        gap      tau (steps)   independent
grok      12070        710           17.0
grok_s1    1750        160           10.9
grok_s2    1740        250            7.0
```

τ scales with the gap, so the budget does **not** improve with a longer plateau. The
report's figure of 10 (from `phase14_budget.txt`, τ=163 on a 1615-step gap, measured
on dense logs with a different τ definition) sits inside the measured 7–17 range. The
report's conclusion is correct and my criticism of it was not. The only wording change
worth making is to quote a range rather than a single value.

**1.2. My genericity argument about ‖w‖₂ was overstated, and is now superseded by
something stronger.** I argued that ‖w‖₂ is non-generic because it is constant on
O(D)-orbits. That does not follow: the delay map is built from ‖w‖ evaluated along the
orbit, and the optimiser dynamics does not descend to the quotient by ‖·‖, so no
factoring argument applies. The correct statement is stronger and is the authors' own
**[source]** (Stark 2003, §2.3):

> "In the present context, unfortunately, it does not seem possible to characterize
> the set of f and ϕ satisfying the theorem; one can be certain only that this set is
> residual. **This makes it impossible to verify the relevance of the theorem to any
> specific system or class of systems.**"

So for a stochastically forced system the question "is ‖w‖₂ generic here?" is not
merely unanswered — the theorem's authors state it cannot be answered for any specific
system. Invoking Stark to license a named observable on a named training run is
therefore not a claim that is unproven; it is a claim the cited theorem cannot support
in principle.

**1.3. The "straightness" statistic I proposed is not robust.** I reported that the
net-to-gross displacement ratio of the normalised probe logits separates grokking from
never-generalising runs where V_t fails. Under horizon sensitivity it does not
**[computed]** — see §8.3. The ordering inverts at a 4000-step horizon. It survives as
a hypothesis, not as a finding.

**1.4. The dimension "turnover" I read off the phase-aligned trace is inside the
noise.** With 10th–90th percentiles attached, `grok`'s late decline overlaps its own
band **[computed]** — see §5.2.

---

## 2. What the embedding theorems actually say

This section replaces the report's §1 and §3 framing entirely. It is the most
consequential part of the audit.

### 2.1. Takens

**[source]** Stark 1999, Theorem 2.1, quoting Takens 1980:

> "Let M be a compact m dimensional manifold. Then if d ≥ 2m + 1, the set of (f, ϕ)
> for which the map Φ_{f,ϕ} is an embedding is open and dense in
> D^r(M) × C^r(M, ℝ) for r ≥ 1."

Genericity is defined **[source]** (Stark 1999, §2.2) as holding on a *residual*
subset — one containing a countable intersection of open dense sets.

Hypotheses, complete: **M compact**; **f a C^r diffeomorphism of M**; **ϕ ∈ C^r(M,ℝ)**;
**d ≥ 2m+1**; **(f,ϕ) in a residual set**.

There is no recurrence hypothesis. No ergodicity hypothesis. No assumption that the
orbit returns near past states. I have now read both Stark papers in full, including
the statement and proof of the standard theorem (Stark 1999, §4), and no such
condition appears anywhere.

The sharpened form **[source]** (Stark 1999, Theorem 2.2, after Huke 1993) makes the
conditions on f explicit: *f has only a finite number of periodic orbits of period
less than d, and the eigenvalues of each such orbit are distinct*. Only the conditions
on **f** can be made explicit; those on ϕ cannot, even in the unforced case.

And one hypothesis is not merely technical **[source]** (Stark 1999, Lemma 2.3):

> "Suppose that f ∈ D^r(M) has a fixed point x ∈ M such that T_x f has two linearly
> independent eigenvectors with the same eigenvalue. Then Φ_{f,ϕ} fails to be an
> immersion at x **for all** ϕ ∈ C^r(M, ℝ)."

This is a statement that at such a point *no observable works*. It is directly
relevant to over-parameterised networks: near a minimum, the linearised optimiser map
has large eigenvalue multiplicity from flat directions and from permutation symmetry.
Whether the multiplicity is exact is **[open]** and measurable — see §7, E0.4.

### 2.2. Stark I — deterministic forcing

The premise of the whole paper is a warning against exactly the move the article makes.
**[source]** (Stark 1999, §3.2), on enlarging the state to (x,y) and applying Takens:

> "since C^r(M, ℝ) is not generic in C^r(M × 𝕋¹, ℝ), and D^r(M × N, M) × D^r(N) is not
> generic in D^r(M × N), we cannot conclude that typical skew products and typical
> functions on M lead to an embedding. […] **We thus conclude that the existing
> versions of the Takens Theorem are not relevant to forced systems.**"

The article's Problem Statement defines the extended state s_t = (w_t, h_t) with h_t
the optimiser's internal variables and ξ_t the minibatch forcing, and then invokes
Takens on the enlarged system. Stark's opening argument is that this specific step is
invalid, and the two papers exist to repair it.

The repairs **[source]**:

- **Theorem 3.1 (Forced Takens).** M, N compact of dimension m, n; periodic orbits of
  period < 2d of g ∈ D^r(N) isolated with distinct eigenvalues; **d ≥ 2(m+n)+1**. Then
  an open dense set of (f,ϕ) gives an embedding of **M × N** — the product, so the
  required delay dimension includes the forcing dimension.
- **Theorem 3.2 (Bundle/fibre).** **d ≥ 2m+1** suffices — but only to embed each fibre
  M × {y}, and this presupposes the forcing state y is independently known. Condition
  on g: closure of the set of periodic orbits of period ≤ d has zero Lebesgue measure.
- **Periodic forcing is the excluded case** (Stark 1999, §3.4): "when τ/T is a rational
  with denominator less than d, every point in 𝕋¹ will be periodic (for g) with period
  less than d, and hence the hypotheses of Theorems 3.1 and 3.2 will unfortunately not
  be satisfied." Theorem 3.3 then only embeds the finite set of sampled phases
  M × {θ₀,…,θ_{q−1}}, not M × 𝕋¹.

### 2.3. Stark II — stochastic forcing, which is our case

Setting: x_{i+1} = f(x_i, ω_i), M a smooth compact manifold, N compact,
Σ = N^ℤ, skew product T(x,ω) = (f(x,ω₀), σ(ω)). Standing assumptions **[source]**:
M compact ("the case of M noncompact can be treated … but will not be considered here
further"), N a compact manifold.

- **Theorem 2.3 (Stochastic Takens).** µ_Σ an **invariant** measure on Σ with µ_{d−1}
  absolutely continuous w.r.t. Lebesgue on N^{d−1}; d ≥ 2m+1; then a residual set of
  (f,ϕ) gives an embedding for µ_Σ-almost every ω. Invariance of µ_Σ is the
  stationarity requirement, stated as such **[source]** (§2.1): "The restriction to
  σ-invariant measures corresponds to a **stationarity condition** on the corresponding
  random process."
- **Theorem 2.4 (Iterated Function Systems).** dim N = 0, i.e. N a *finite* set of
  points; d ≥ 2m+1; then an **open dense** set of (f,ϕ) gives an embedding for **every**
  ω ∈ Σ.

**Theorem 2.4 is the correct citation for minibatch SGD.** Selecting a minibatch from a
finite training set is exactly an IFS: a finite family of maps, one applied per step.
This is a point *in favour* of the theory applying in form — and the report should say
so rather than arguing that the theorems do not reach training at all.

What Theorem 2.4 gives, and what it costs:

1. **The reconstructed object is M, per realisation.** The forcing is not reconstructed
   **[source]** (§2.2): "φ is a function of M only and hence attempting to reconstruct Σ
   using φ seems foolhardy."
2. **The image moves with the input history** **[source]** (§2.2): "Note that in general
   the image Φ_ω(M) will be different for each ω (but all these images will be
   diffeomorphic to M)." And §2.4: "different F_ω have different domains (each a subset
   of ℝ^d diffeomorphic to M). There is no reason in general why these should all be
   disjoint."
3. **The reconstruction is a NARMA** **[source]** (§2.4, eq. 2.4):
   φ_{i+d} = G(φ_i,…,φ_{i+d−1}, ω_i,…,ω_{i+d−1}). The delay vector does **not** determine
   the next value; the unobserved inputs enter. "It is clear that estimating both G and
   ω in such a model is a major challenge."
4. **d ≥ 2m+1 with m = dim M**, the extended optimiser state.
5. **Membership of the residual set is unverifiable** (quoted in §1.2).

### 2.4. The consequence, which is the report's real argument

Point (2) alone disposes of intrinsic-dimension estimation on delay vectors of a
stochastically forced system, and it has nothing to do with recurrence:

> The delay vectors z_i = Φ_{f,ϕ,σ^i(ω)}(x_i) lie in **different** embedded copies of M,
> one per shift of the input sequence. A sliding window of delay vectors is therefore
> not a sample from one manifold. Levina–Bickel assumes an i.i.d. sample from a density
> on a *single* d-manifold. The hypothesis fails structurally, not statistically.

Point (4) is the decisive quantitative statement. For AdamW the natural extended state
is (w, m, v) ∈ ℝ^{3D}, so m = 3D and the theorem asks for d ≥ 6D+1. The article uses
E = 15. E = 15 is licensed by Stark only if one already knows the dynamics is confined
to an invariant submanifold of dimension ≤ 7 — **which is the article's conclusion**.
The theorem is being used to license a measurement whose validity presupposes the
answer. That is a clean, citable circularity, and it replaces the recurrence argument
entirely.

Point (3) reframes the simplex/determinism results. For a stochastically forced system,
one-step prediction from delay coordinates alone is bounded by the entropy of the
unobserved inputs. Low simplex skill on `train_loss` is what the theory *predicts*; it
is not evidence that the log lacks deterministic structure.

---

## 3. What is strictly correct in the report

1. **Proposition 1 is arithmetically correct.** Verified independently: k=5, W=0 gives
   4/[2 ln 3 + 2 ln(3/2)] = 1.3297; k=5, W=14 gives 4/[2 ln(17/15) + 2 ln(17/16)] =
   10.765. `phase8_line_constants.csv` measures 1.3293 and 10.702 on a synthetic line
   (rel. err 3.1e-4 and 5.8e-3) **[repo]**.
2. **The MacKay–Ghahramani correction is stated correctly** in
   `../grokking_analysis/README.md`. Re-derived: if d·S_i ~ Γ(k−1,1) then
   E[1/S_i] = d/(k−2), so the local estimate (k−1)/S_i is inflated by (k−1)/(k−2), and
   pooling gives the unbiased d̂ = (n(k−1)−1)/ΣS_i. The AM–HM argument that MG ≤ LB is
   correct.
3. **The phase-10 retraction is exemplary.** The identifiability diagnostic is strongly
   length-dependent (Lorenz 8.86 at n=2000 against white noise 4.39), the original
   comparison was length-mismatched, and it was withdrawn and replaced by a weaker true
   statement about the data budget **[repo]**.
4. **The recurrence-radius bug** (recomputing the radius after exclusion, which forces
   RR to equal the quantile for any input) was a real error, found and fixed.
5. **The 1/√L normalisation of the stationarity index** is the right correction; without
   it the criterion silently loosens with window length.
6. **The sample-budget argument** — that independent samples = window / correlation time
   and neither term depends on the logging rate — is correct, and §1.1 confirms it more
   strongly than the report claims.
7. **The CCM section.** Ghost controls are the right control. 199 surrogates instead of
   39 (whose p-floor of 0.025 *is* the two-test Bonferroni threshold) is a necessary
   correction. Calibrating the direction convention on coupled logistic maps
   (0.331→0.993 true vs 0.585→0.569 false) before reading direction off project data is
   exactly right. Declining to claim unidirectionality because of contemporaneous
   coupling is correct.
8. **`verify_numbers.py` passes 25/25** against the result files **[computed]**.
9. **The delay-variability result** (30 cells, 1290–5600, sd/mean > 1/3) is solid and by
   itself invalidates any precursor validated on a single run.

---

## 4. What is overstated

**4.1. Recurrence and stationarity are presented as hypotheses of the theorems.**
Abstract: "The embedding theorems of Takens and Stark reconstruct a compact invariant
set that the trajectory revisits." §3: "The guarantee is asymptotic and geometric: the
orbit must return near its past states." Neither is in either theorem (§2.1–2.3). The
guarantee is a genericity statement about a map on a compact manifold; it is not
asymptotic and it says nothing about returns. Recurrence is a requirement of the
*estimator*, not of the theorem, and — per §2.4 — it is not even the binding one.

**4.2. The abstract claims the dimension estimate "reproduces closed-form constants of
a straight line."** Only `wd0` does. Phase-aligned, causally labelled **[computed]**:

```
run          t_mem   t_gen     +1-2k             +2-3k             +3-5k             +5-8k            +8-12k
grok          1630   13700   1.58 [1.53,1.61]  2.07 [1.67,2.37]  3.03 [2.63,3.12]  2.73 [2.66,2.93]  2.56 [1.58,2.93]
grok_s1       1240    2990       n=0               n=0               n=0               n=0               n=0
grok_s2       1400    3140   1.65 [1.65,1.65]      n=0               n=0               n=0               n=0
lowdata15      660    None       n=0           1.86 [1.70,2.05]  2.27 [2.09,2.68]  3.20 [2.89,3.49]  3.34 [2.94,3.60]
lowdata20      830    None       n=0           1.68 [1.63,1.89]  2.26 [1.97,2.75]  3.50 [3.25,3.86]  3.63 [3.29,4.05]
wd0            640    None       n=0           1.39 [1.38,1.39]  1.36 [1.35,1.37]  1.34 [1.33,1.34]  1.33 [1.33,1.33]
```

(median [10th, 90th percentile]; bands measured from `t_mem`; windows labelled by right
edge and dropped once they would contain data past `t_gen`; line constant 1.330.)

`grok` runs 1.53–3.19 and `lowdata` 1.62–4.55 — 15 % to 240 % above the constant. The
sentence "it pins the estimate to the tangent constant, which is why such curves look
stable" is false for every WD=1 run. What is true, and is the stronger claim, is in the
table itself: **at matched phase the estimate is indistinguishable between runs that
generalise and runs that never do.**

**4.3. "The longest window indistinguishable from stationary is 25 steps."** The report
defines index ≈ 1 as indistinguishable. At L=25 the index is 2.83–3.21 **[repo]**. The
threshold of 3.0 is a choice made in `phase14_local_stationarity.py:151`; it does not
mean "indistinguishable". Correct statement: *no window at any length in any run is
indistinguishable from stationary; the minimum index, ≈3, occurs at L=25.*

**4.4. The determinism test on `weight_norm` is degenerate, not negative.**
`phase15_local_determinism.csv`: skill −0.588 against a surrogate mean of 0.937
**[repo]**. Negative out-of-sample skill on a monotone series is the signature of
extrapolation beyond the library, not of absent determinism. The `train_loss` row
(0.739 vs 0.722, p=0.52) is interpretable; the `weight_norm` row is not and should not
be cited as evidence.

**4.5. The CCM section's "satisfies that hypothesis by construction" is too strong for
two of the three drivers.** Stark 1999 Theorem 3.1 requires g ∈ D^r(N), a
*diffeomorphism*. The logistic map is not invertible, so Stark I does not apply to it;
the correct citation is Stark 2003 (arbitrary/stochastic forcing), under which the
driver's own dynamics is not reconstructed at all. See §9.2 for the sinusoid, where the
mismatch is more interesting.

---

## 5. What is wrong

### 5.1. The seed comparison does not test the article's claim

The report's internal check — "across three seeds the estimate falls in one run
(−0.54) and rises in the other two (+0.94, +0.90)" — is computed on whole-run trends.
The article's claim is about `(t_mem, t_gen)`. At the published window (300 samples =
3000 steps), count of stride-10 window positions whose full span lies inside the
plateau **[computed]**:

```
run        t_mem  t_gen   gap    W=3000  W=1500  W=600
grok        1630  13700  12070     908    1058   1148
grok_s1     1240   2990   1750       0      26    116
grok_s2     1400   3140   1740       0      25    115
```

`grok_s1` and `grok_s2` afford **no** measurement inside the interval the claim is
about. Their "+0.94 / +0.90" is a post-generalisation trend. Likewise the "spread of
7.7 within one configuration" compares `grok` just after `t_gen` with `grok_s1/s2` ten
thousand steps after `t_gen`.

This must be rewritten. The honest version is a statement about the data budget, and it
is a good one: *at the published window length, two of three seeds of the headline
configuration have no causal pre-transition window at all, because the estimator's
window is longer than their entire plateau.* That is more damaging to the original
analysis than the seed argument, and it is true.

Event detection is robust: t_mem/t_gen shift by ≤ 70 steps across smoothing windows
w ∈ {1,5,11,21}, and at w=1 reproduces the values documented in `method.md` exactly
(grok 1700/13700, grok_s1 1310/2990, grok_s2 1460/3140, nogap 770/780) **[computed]**.

### 5.2. The shape difference is inside the noise

From the table in §4.2: `grok` goes 3.03 [2.63,3.12] → 2.73 [2.66,2.93] → 2.56
[1.58,2.93]. The bands overlap; the decline is not established. `lowdata15`'s rise from
2.27 [2.09,2.68] to 3.20 [2.89,3.49] is outside its bands and *is* established. So the
only defensible statement is: **all three WD=1 runs rise from ~1.6–1.9 to ~2.6–3.6 at
matched phase; the never-generalising runs continue to rise while the grokking run goes
flat, by an amount comparable to the within-band spread.**

### 5.3. The velocity comparison is phase-confounded

The report's velocity result uses "the second half of training". For `grok`
(t_gen = 13700 of 20000 steps) that spans the transition and everything after. For
`lowdata` (never generalises) it is late memorisation. This is the same defect the
report correctly identifies in §3.3 — "the apparent step in an aggregate over runs
arises from averaging runs whose transitions occur at different steps."

My reproduction of the report's own numbers is exact **[computed]**, so the arithmetic
is not in question:

```
run          V(band +1-2k after t_mem)   V(late half)    NOTES.md quotes
grok                    0.1691             0.01806          1.84e-2
grok_s1                 0.07109            0.01398          1.38e-2
grok_s2                 0.05944            0.01916          1.94e-2
nogap                     —                0.02371          2.34e-2
lowdata15               0.07077            0.05291          5.30e-2
lowdata20               0.1107             0.0722           7.26e-2
wd0                     0.01714            0.0001557        1.53e-4
```

At matched phase the ordering is **mixed**: `grok` is the highest of all runs, while
`grok_s1` and `grok_s2` sit at or below `lowdata15`. The conclusion "V_t is not a
generalisation precursor" survives — it does not separate. The specific claim that
"the runs that never generalise sustain a *higher* velocity … the two families do not
overlap … separated by a factor of 2.73" is a **late-training** statement and must be
labelled as such, not presented as a property of the pre-transition window.

### 5.4. Editorial

`report.tex:278` promises "Four observations"; five paragraphs follow, and both
`report.tex:323` and `report.tex:365` begin "The third observation".

### 5.5. Two things the report should be claiming and is not

Both are supplied by §2 and are stronger than what is there now:

- **The delay-dimension requirement.** Stark needs d ≥ 2m+1 with m the state dimension;
  for AdamW that is ~6D+1. The published analysis uses E = 15. This is the cleanest
  possible statement of the gap and it needs no measurement.
- **The moving fibre.** Under stochastic forcing the images Φ_ω(M) differ with the
  input history, so delay vectors from different steps are not a sample from one
  manifold — which is precisely what Levina–Bickel assumes.

---

## 6. What MG/LB actually measures — answer to question A

### 6.1. Level 1 — mathematically guaranteed

Under Stark 2003 Theorem 2.4 (the IFS case, which is the right one for minibatch SGD):
for an open dense set of (f,ϕ), with M a compact manifold of dimension m and
d ≥ 2m+1, the delay map Φ_{f,ϕ,ω}: M → ℝ^d is an embedding for every input sequence ω.

That is a statement about **separating states**: two distinct optimiser states have
distinct delay vectors. It is not a statement about the geometry of any finite sampled
orbit. Four of its hypotheses are unmet or unverifiable here:

| hypothesis | status |
| --- | --- |
| M compact | weight space is ℝ^D; explicitly out of scope in Stark 2003 **[source]** |
| f_ω a diffeomorphism of M for each ω | plausible for the AdamW map on the extended state; **[open]**, untested |
| d ≥ 2m+1 | m = 3D for AdamW; the analysis uses d = 15 |
| (f,ϕ) in the residual set | **not verifiable in principle** **[source]**, §1.2 |

What the theorem *does* license, and what should be quoted in the paper: the invariants
preserved by the reconstruction **[source]** (Stark 1999, §2.3) — "the correlation
dimension and Liapunov exponents of **corresponding invariant measures**". Note the
qualifier: it is the dimension of an *invariant measure*, which presupposes one exists.

### 6.2. Level 2 — what is practically estimated

Levina–Bickel is a kNN estimator on a point cloud, assuming an i.i.d. sample from a
smooth density on a d-manifold, with the local neighbour count Poisson-approximated.
Feed it delay vectors from a finite orbit segment. Three facts:

1. **The sample is a curve.** {x_t} for a smooth trajectory is the image of an interval
   under a smooth map. Its intrinsic dimension as a set is 1. Everything above 1 comes
   from noise thickening, curvature at neighbourhood scale, and non-uniform sampling.
2. **The samples are not i.i.d.** Consecutive delay vectors are deterministically
   related; the Poisson model is wrong at exactly the scales used.
3. **Under stochastic forcing they are not even on one manifold** (§2.4).

Expand locally, x(t) = x₀ + v·δ + ½a·δ² + ε with noise scale σ. The ordered neighbour
distances are

    r_j ≈ |v|·s_j · sqrt( 1 + κ²s_j² + (σ/(|v|s_j))²·χ²_E ),   s ∈ {W+1, W+1, W+2, …}

| regime | LB returns | information content |
| --- | --- | --- |
| σ ≪ \|v\|W, curvature negligible | the line constant (1.330 at k=5, W=0) | none |
| σ ~ \|v\|s_j | rises toward E | local noise-to-drift ratio |
| curvature significant at scale W | intermediate | curvature × scale / speed |

Proposition 1 in the report is the exact statement of the first regime. It is correct
and it is a special case, not a description of the estimate on real data.

**Answer: the fifth option in the question — the local geometry of a finite set of
delay vectors — and, because that set is a sampled curve, specifically the local
noise-to-drift ratio and curvature of one scalar series at the scale set by (k, W, τ).**

But — and this is the user's point 3, which is correct — that is a *real measurement of
a real thing*. It is a reproducible property of ‖w‖₂. The illegitimate step is calling
it the dimension of the hidden system, not the measurement itself.

**How much information is in it beyond the one-line roughness ratio?** Rank-residual
after regressing rank(d̂) on rank(std(Δx)/std(x)) **[computed]**:

```
run         spearman(d,rough)   sd(rank residual)   sd(ranks)   variance explained
grok             +0.875              23.90            49.36           77 %
grok_s1          +0.884              23.04            49.36           78 %
grok_s2          +0.955              14.63            49.36           91 %
lowdata15        +0.684              36.02            49.36           47 %
lowdata20        +0.321              46.74            49.36           10 %
wd0              +0.996               4.64            49.36           99 %
```

The mapping is not run-invariant: 10 % explained in `lowdata20`, 99 % in `wd0`. The
report's hedge — "smoothness is a strong correlate rather than a complete description"
— is exactly right and should not be strengthened.

### 6.3. Level 3 — the desired interpretation

"Number of active degrees of freedom of the hidden optimisation system." The distance
from level 2 is qualitative, not quantitative: it is the distance between *the
dimension of a cloud that is a curve* and *the dimension of a set that curve would have
to cover*.

### 6.4. When must different good observables agree?

If Φ_y and Φ_z are both embeddings of the same compact invariant set A, then
Φ_z ∘ (Φ_y|_A)^{−1} is a diffeomorphism onto its image; on a compact set a C¹
diffeomorphism with C¹ inverse is bi-Lipschitz; and box-counting, Hausdorff and
correlation dimensions are bi-Lipschitz invariants. Hence
**dim Φ_y(A) = dim Φ_z(A)**. Stark states the same conclusion directly for correlation
dimension and Lyapunov exponents of the corresponding invariant measures **[source]**
(1999, §2.3).

They can differ for seven reasons, which must be kept separate:

1. **Non-generic observable** — Φ not injective; the image can have strictly smaller
   dimension. Legitimate, not an artefact. Unverifiable here (§1.2).
2. **Insufficient E** — at E ≤ 2d the map may fail injectivity; projection lowers the
   estimate.
3. **Finite sample** — LB is strongly biased down for large d and unusable below
   n ≈ 6000 at these settings **[repo]**.
4. **Noise** — additive noise of scale σ thickens the set to dimension E below scale σ.
   Different observables have different σ; estimates diverge.
5. **Non-stationarity** — no single A exists; the estimate mixes over a moving object.
6. **Estimator target** — LB averages local estimates (dominated by points with small
   S_i); MG pools the likelihood, which *postulates a single d for the window* and
   yields a harmonic-type mean dominated by low-dimensional points. On a set of varying
   local dimension these answer different questions. Documented MG/LB ratios in the repo
   run 0.73–1.00 **[repo]**.
7. **Scale** — k, τ, E, W each set a physical scale. Comparison across observables is
   meaningful only at matched physical scales, which nobody does.

Reason (1) versus (2)–(7) is exactly what experiment E1.1 below is designed to separate.

---

## 7. Validation programme — answer to question B

Ordered so each tier can fail cheaply and stop the next.

### Tier 0 — calibration (hours, no training)

- **E0.1 — synthetic systems with known, *changing* dimension.** A torus whose second
  frequency amplitude is ramped to zero (true d: 2 → 1 at a known step); Lorenz → limit
  cycle through a known bifurcation. Run at the *same* window, stride, k, τ as the logs.
  Prediction from Proposition 1 and the phase-10 table: it will not resolve the change.
  If so, that fixes an upper bound on every claim and belongs in the paper.
- **E0.2 — calibration curves.** d̂ vs true d ∈ {1..8} at n ∈ {100,300,1000,3000},
  k ∈ {5,10,20}, with and without noise. Converts any number from a log into an interval.
- **E0.3 — residual over roughness.** Regress d̂ on std(Δx)/std(x); test whether the
  residual discriminates. Existing data; §6.2 gives the starting point.
- **E0.4 — eigenvalue multiplicity of the optimiser map.** Estimate the spectrum of the
  linearised AdamW step near convergence (Lanczos on the Jacobian-vector product). If
  eigenvalues are repeated, Stark 1999 Lemma 2.3 says no observable gives an immersion —
  a theorem-level statement about our own system. Cheap and currently unmeasured.

### Tier 1 — what is being measured (the decisive tier)

- **E1.1 — multiple observers.** 8–12 scalar series per run: `train_loss`, `val_loss`,
  ‖w‖₂, ‖g‖₂, grad cosine, median ‖z_i‖, softmax entropy, and **16 fixed random linear
  functionals of w** and **16 of Δw**. The random projections are the point: they are
  drawn from the class the genericity statement is about, whereas ‖w‖₂ is a single
  structured function. Under §6.4 the estimate must agree across them, at matched scale,
  if it is a property of the system. If the drop appears only in ‖w‖₂ it is a property
  of ‖w‖₂. One training run per configuration.
- **E1.2 — raw vs detrended.** Same estimator on the series and on the residual after a
  local linear/quadratic fit. The most direct test of "is the drop a trend artefact",
  and it is currently absent.
- **E1.3 — sensitivity grid.** E ∈ {5,10,15,20}, τ ∈ {1, τ_DMI}, k ∈ {5,10,20},
  window ∈ {100,200,300,600}, W ∈ {0, (E−1)τ, τ_acf}. Require the *claim* to be
  invariant, not the number.
- **E1.4 — block bootstrap and n_eff.** Block length ≥ correlation time; point bootstrap
  understates the interval by an order of magnitude here. Always print
  n_eff = L / τ_corr beside the estimate.
- **E1.5 — the right surrogate.** IAAFT on a monotone series nearly reproduces it and has
  no power. The null that matters is *smooth trend + spectrum-matched noise*: fit the
  trend, phase-randomise the residual, recombine. That separates "geometry beyond the
  trend" from "geometry of the trend".

### Tier 2 — negative controls

Pure rescaling w → c(t)w (must return the line constant); frozen model plus noise;
cross-entropy saturation; memorisation without generalisation (`lowdata`, exists);
WD = 0 (exists).

### Tier 3 — positive controls with known active degrees of freedom

- **E3.1 — deep linear network on a whitened task** (Saxe et al.): modes are learned
  sequentially and the number learned by step t is countable exactly.
- **E3.2 — scheduled-rank parameterisation:** rank r(t) reduced on a known schedule; ask
  whether MG/LB tracks r(t).

Without Tier 3 the strong claim is impossible in principle — there is nothing to
compare against.

### Tier 4 — comparison estimators

PCA participation ratio on the same delay matrix (exact reference PR = 1 on a straight
line, better than LB's 1.33); local PCA; correlation dimension with a Theiler window
and the Takens–Theiler estimator; **TwoNN** (Facco et al.), far less sensitive to k.
Agreement across four estimators with different failure modes is worth more than any
p-value.

### Tier 5 — reproducibility of the moment

≥10 seeds × ≥3 configurations. Is t_drop reproducible *relative to* t_mem and t_gen?
The sweep already shows sd/mean > 1/3 for the gap, so any lead-time claim needs n ≳ 20
before it has power.

### Tier 6 — the ensemble redesign (recommended centrepiece)

**E6.1.** Train 200 seeds of one configuration. At each step t, estimate the intrinsic
dimension of the **200-point cloud** — of weight projections, or of normalised probe
logits — across seeds.

Why this changes the problem:

- the points are i.i.d. **by construction**, so Levina–Bickel's hypothesis holds literally;
- no delay embedding, so Takens and Stark are not invoked and none of §2 applies;
- no recurrence requirement, so the report's central objection is irrelevant;
- "the dimension of the set of functions the optimiser is exploring at step t" is a
  well-posed object with a well-posed estimator;
- the controls (`lowdata`, `wd0`) come free.

Cost: 200 × 3.4 min on a T4 ≈ 11 GPU-hours per configuration. This converts an
unsupportable claim into a supportable one.

### Logging changes required (cheap, blocking)

`probe.py:155` normalises the logits **before** snapshotting, so scale is destroyed and
the scale/direction decomposition of §8 cannot be done on existing data. Probe labels
are not stored, so margins cannot be computed offline either. Add:

- ‖z_i(t)‖ per example (or median plus quantiles);
- probe labels, enabling margins on train and on a fixed held-out probe;
- cosine between consecutive weight-space updates;
- ‖w‖, ‖Δw‖ per layer;
- **a probe run at `log_every=1`** — the one unclosed experiment of the function-space
  route (see §9.1).

### What licenses which claim

| claim | requirements |
| --- | --- |
| "MG records a stable local simplification of the observed dynamics" | E0.1, E0.3, Tier 2, E1.2, E1.4, E1.5, Tier 4 — with the observable **named**, causal labels, and the drop reproducible at matched phase across seeds. Achievable. |
| "MG estimates the local dimension of the hidden optimisation system" | Additionally: E1.1 (same drop in generic projections); invariance of the estimate under change of observable (§6.4 — the theorem's own signature); Tier 3 (known ground truth); and a sample that covers a *set*. The last is unreachable within one run — §2.4 shows the delay vectors are not even on a common manifold — so only Tier 6 can deliver it. |

---

## 8. Dimension drop without generalisation — answer to question C

The hypothesis under test: after memorisation the model moves in a simpler regime
(growing margins/norms, few directions), so the observed dynamics genuinely becomes
low-dimensional without generalisation.

### 8.1. For WD=0 the answer is more specific than that

`wd0`'s function-space speed collapses by a factor ≈ 500 (3.083 → 0.006 between the
+0–1k and +8–12k bands after t_mem) **[computed]**, and its ‖w‖ series becomes an exact
straight line, so LB is pinned at 1.33–1.39 (§4.2 table). **The WD=0 "drop" is the
estimator's line constant, not a simpler regime.** The user's mechanism is not what is
happening there.

### 8.2. For the dangerous control there is no drop to explain

`lowdata15/20` keep speed high (1.6–3.1) and their MG estimate **rises** monotonically,
exactly as `grok`'s does (§4.2). So the case that actually threatens the criterion is
not "dimension falls without generalisation" — it is "dimension does the same thing in
both".

### 8.3. Shape of the motion: the pair is confounded in opposite directions

Computed from the existing snapshots (normalised logits, every 200 steps, 256 probes ×
114 logits, 100 snapshots per run). Straightness D = ‖Σ ΔZ‖ / Σ‖ΔZ‖ over a window,
phase-aligned to the band t_mem+1000 … t_mem+2000, capped at t_gen **[computed]**:

```
                  H=600    H=1000   H=2000   H=4000
grok              0.481    0.390    0.316    0.249
grok_s1           0.628    0.610    0.438    0.349
grok_s2           0.668    0.580    0.476    0.353
lowdata15         0.474    0.334    0.313    0.294
lowdata20         0.416    0.319    0.253    0.251
wd0               0.677    0.664    0.649    0.456
```

(train and val probes agree to ≤ 0.01 throughout.)

Two readings, and only one of them survives.

**Survives, at every horizon:** V_t and D are confounded in *opposite* directions.
`wd0` has the highest D and by far the lowest speed; `lowdata` has high speed and the
lowest D. So neither statistic alone can work, and the failure modes are complementary.
This is a real structural argument for a composite criterion, and it is the strongest
support the composite idea has.

**Does not survive:** the claim that the pair separates grok from lowdata. At H=1000 it
does (0.390 vs 0.334, a 17 % margin on five runs). At H=2000 it ties (0.316 vs 0.313).
At H=4000 it **inverts** (0.249 vs 0.294). One free parameter, five runs, one horizon
out of four working — that is consistent with chance. Reported as a pre-registrable
hypothesis, not a result.

Caveat that must accompany any use of D: path length is measured at 200-step
granularity, so D is an upper bound on true straightness and its value depends on the
snapshot cadence. Recomputing at `log_every=1` could change the ordering.

### 8.4. Tests to separate the cases

1. **Scale/direction decomposition** — requires ‖z_i‖, currently discarded (§7).
   Criterion must fire only when *direction* moves.
2. **Net-to-gross displacement D** — computed above; needs horizon sensitivity reported,
   never a single horizon.
3. **Effective rank of the update window** — computed; PR 2.2–3.4; does not separate.
4. **Margins on train and on a fixed held-out probe** — not computable offline (no
   labels stored). Cheap fix.
5. **Cosine between consecutive weight-space updates** — not logged. Cheap fix.
6. **Dimension of detrended vs raw series** — the most direct test of "is this the
   scale trend", and absent from the report (E1.2).

**Conclusion for C.** The only reliable way to exclude norm pumping is to make the
observable scale-invariant *and* require net displacement rather than path length. The
existing probe does the first; V_t does the opposite of the second.

---

## 9. The simplified criterion — answer to question D

Taken as the last simplified version: `method.md` §3 (observable + statistic + single
causal rise rule), with the corrections in §7 — **not** the superseded MG-dimension
draft with its ~22-parameter table, baseline state machine, consensus fraction and
two-tier WARNING/CONFIRMED, all of which §9 of that document already discards.

| question | answer |
| --- | --- |
| What should it predict | As written, departure from local linearity of a function-space series. That is a **structural-reorganisation detector**, not a generalisation predictor. No version tested is an early warning. |
| Necessary conditions | (a) memorisation confirmed, defining the window; (b) a scale-invariant function-space observable; (c) a movement floor; (d) causal labelling. |
| Superfluous conditions | The MG dimension of ‖w‖₂ — §4.2 shows it does not separate `grok` from `lowdata` in level or direction. The consensus fraction over 16 projections — the projections are correlated, so the fraction is uninterpretable, as `method.md` §9 already concedes. |
| Norm pumping | Handled correctly and verifiably: class-centring plus L2 normalisation removes the uniform rescaling mode, and `wd0` drops to 1.53e-4. This is the one component that works as designed. |
| Remaining false positives | `lowdata` (WD>0, memorises forever) is excluded by nothing currently proposed. That is the whole problem. |
| Usable before val accuracy rises | In principle yes — the train probe never touches validation data. In practice **no** at the published window: 300 samples = 3000 steps exceeds the typical gap (1290–5600). Any usable criterion needs a window ≪ min gap, which forces the estimator away from kNN dimension toward SVD/PR/roughness, which need tens of samples. |
| Avoiding future data | Right-edge labels (done). Add: forbid normalisation by any run-global quantity — "25 % below the running peak" is admissible, "below the global peak" is not. |
| Threshold calibration | Split by whole runs; fit only on a designated calibration set of *configurations*; freeze; report separately on held-out seeds and held-out configurations, because the sweep shows seed variance is large enough that the two generalisations are not the same claim. |
| Metrics | Precision, recall, the **distribution** of lead time (not its mean), false alarms per 1000 steps on never-generalising runs. At n = 7 none of these is estimable. A ±0.2 CI on precision needs ≈ 40 runs (20/20) across ≥ 3 configurations. |

### 9.1. The unclosed experiment

`method.md` §7 records that roughness saturates at its ceiling (0.97 of 1) because at
10-step logging the per-row projection noise (~4.4e-3) is comparable to the range inside
a short window. That is a **sampling** failure, not a statistical one. Dense logging
arrived in phase 13 and has never been applied to the probe. The report and NOTES treat
the function-space route as closed; it is not.

### 9.2. A theorem-level point the report is missing

The report explains the sinusoid miss statistically: spectrum-preserving surrogates are
deterministic functions of the same phase and therefore have no power. That is correct.
There is also a hypothesis-level statement **[source]** (Stark 1999, §3.4): for periodic
forcing with sampling interval rationally related to the period with denominator < d,
*every* point of the forcing circle is periodic with period < d, and the hypotheses of
both Theorem 3.1 and Theorem 3.2 fail. Theorem 3.3 then embeds only the finite set of
sampled phases.

So the periodic case is excluded by the theorem, not merely underpowered by the null.
Whether our injected sinusoid's period is rationally related to the logging interval is
**[open]** and takes five minutes to check in `inject.py`. If it is, the report gains a
much stronger statement than the one it has.

### 9.3. Differencing

Here I largely disagree with the objection. Requirement 3 of the report already says
"Justify preprocessing **per series**", which is the correct framing. What is missing is
the positive half:

Differencing is a linear filter with transfer |H(ω)| = 2|sin(ω/2)| — it suppresses low
frequencies as ω and amplifies high ones.

- **Required** when a shared trend is a nuisance that would otherwise manufacture
  cross-map skill through common monotonicity. First differencing of trending series is
  standard EDM practice.
- **Destructive** when the hypothesis concerns a slow or periodic driver whose power is
  at low frequency. Our case: 0.81 → 0.09.
- **The choice follows the hypothesis**, not a general rule.
- **The arbiter is the control, not the argument.** The ghost runs share the trend, the
  loss scale and the logging cadence with the coupled runs. If the ghost stays silent
  under the chosen preprocessing, the preprocessing did not manufacture the signal. This
  should be stated explicitly: *the ghost control makes the preprocessing choice
  empirically testable rather than a matter of taste.*

One citation caution: I could not retrieve the Sugihara et al. (2012) supplementary
material (the PDF does not extract), so I cannot confirm from the primary source what
preprocessing that paper applied to the sardine/anchovy/SST series. Do not cite it for
this point until someone checks the printed SI. **[open]**

---

## 10. The correct role of V_t

`method.md` §6 explicitly designates V_t as **N0 — the null model** the geometry layer
must beat. The report §3.2 presents it as a "candidate signal" and its failure as a
result. That should be corrected: the report refutes V_t in a role it was never
assigned.

The composite the user proposes,

    sustained MG-dimension drop  ∧  substantial change in normalised logits,

is the right *shape*. Two corrections to the second conjunct:

**V_t is admissible only as a one-sided floor, never as a monotone score.** Measured:
`lowdata20` 7.26e-2, `lowdata15` 5.30e-2, `grok` 1.84e-2 **[repo, reproduced]**. A guard
of the form "more movement is better evidence" fires *harder* on the negatives — worse
than no guard at all. A floor at, say, 10 % of the run's own post-memorisation median
excludes `wd0` (which falls ~500×) and touches nothing else.

**V_t is a speed, not a displacement.** A model jittering in place under weight-decay
pressure has large V_t and no net movement. The discriminating quantity is the ratio of
the two, D of §8.3 — for which V_t is the *denominator*. That, and not "still moving",
is the role in which V_t earns its place. It is also why the pair is structurally
sensible even though the pair's empirical separation does not currently hold up.

---

## 11. Claims already permitted

1. CCM recovers an externally injected driver from a 1-D loss log: 7/8 coupled runs,
   0/5 ghosts, two architectures, two nulls, 199 surrogates.
2. The dominant direction is recovered in 7/8. Unidirectionality is **not** established;
   the reason is structural (contemporaneous coupling / generalised synchrony).
3. Spectrum-preserving nulls have no power against a strictly periodic driver — and,
   pending §9.2, the periodic case may also violate Stark's hypotheses outright.
4. Proposition 1 as an analytic fact: for a locally straight, uniformly sampled
   trajectory the LB estimate is a function of (k, W) alone; verified to ≤ 1.1 %.
5. Intrinsic-dimension estimation is strongly length-dependent; below n ≈ 6000 the
   identifiability diagnostic has no power (Lorenz at n=2000 scores worse than white
   noise).
6. No window in any run is locally stationary: the minimum index is ≈ 3 at L = 25,
   shorter than one correlation time.
7. The transition affords 7–17 independent observations, and this does not improve with
   a longer plateau because τ scales with the gap **[computed]**.
8. The memorisation→generalisation delay is broadly distributed: 30 runs of one
   configuration span 1290–5600, sd/mean > 1/3, against 11 790 for the canonical run.
9. CCM between a run's own metrics fires in 11/16 directed pairs and carries no
   information about the transition.
10. Normalised probe-logit velocity separates runs by weight decay, not by
    generalisation — stated as "does not separate", with the late-training caveat of §5.3.
11. Differencing is a high-pass filter that removes slow drivers: 0.81/0.80 raw against
    0.09/0.12 differenced.
12. **New, and stronger than anything currently in the report:** Stark's theorems require
    d ≥ 2m+1 with m the dimension of the state manifold; for AdamW on D parameters that is
    ~6D+1 against the E = 15 used. And under stochastic forcing the reconstructed images
    Φ_ω(M) differ with the input history, so delay vectors from different steps are not a
    sample from one manifold — which is exactly Levina–Bickel's hypothesis.

## 12. Claims not yet permitted

1. "Takens/Stark require recurrence or stationarity." They do not (§2).
2. "The dimension estimate reproduces the constants of a straight line." Only on WD=0.
3. "The estimate contains no information about the data." True only inside Proposition
   1's idealisation.
4. "The signal fails to reproduce across seeds." Two of three seeds have no measurement
   in the relevant interval (§5.1).
5. "The transition affords ten independent observations." Quote 7–17.
6. "None of the five training-side observables discriminates." Two of five were not
   tested in the correct configuration.
7. "Differencing is inadmissible." See §9.3.
8. "V_t refutes the composite criterion." It refutes V_t as a standalone predictor.
9. "d̂ is a dimension." In no version.
10. Any statement about lead time at the current n.
11. That the pair (V_t, D) separates the families — §8.3.

---

## 13. Proposed reformulation of the central hypothesis

**Old (unsupportable).** *Grokking is a topological collapse of the optimiser's
attractor, detectable as a fall in the intrinsic dimension of a delay reconstruction of
a scalar log.* It fails at four independent points: the required delay dimension is
~6D+1 and not 15; membership of the genericity set is unverifiable in principle; delay
vectors under stochastic forcing are not a sample from one manifold; and the estimation
window exceeds the transition.

**H1 — recommended, primary.**

> Across an ensemble of independently seeded runs of a fixed configuration, the set of
> functions realised at step t contracts onto a low-dimensional subset **before**
> validation accuracy rises. The contraction is measured by a kNN intrinsic-dimension
> estimator applied to the ensemble at fixed t — a setting in which the i.i.d.
> hypothesis holds by construction — and is absent in matched configurations that
> memorise forever.

Neither Takens nor Stark is invoked, so none of §2 applies; Levina–Bickel is used as
defined; the controls already exist; it is falsifiable by one ~11 GPU-hour sweep; and
the word "dimension" is used honestly.

**H2 — secondary, within a single run, currently unsupported.**

> In (t_mem, t_gen) the model's motion in function space becomes more *coherent* — the
> net-to-gross displacement ratio D_t rises — while speed V_t is held above a floor; and
> this does not occur in matched never-generalising runs. V_t enters only as a one-sided
> floor and as the denominator of D_t, never as a score.

Status: the horizon sensitivity of §8.3 does not support it. Pre-registrable, not a
finding.

**H3 — retained, already established.** The driven-regime CCM result (§11.1–11.3).

**H4 — negative, established, but reworded.**

> The Levina–Bickel/MacKay–Ghahramani estimate on delay vectors of a scalar training log
> measures the local noise-to-drift ratio and curvature of that series, not the dimension
> of a hidden system. On a transient the object sampled is a curve, not the set the curve
> would have to cover; and under stochastic forcing the delay vectors at different steps
> do not lie on a common manifold. The quantity is real and reproducible; only its
> interpretation is wrong.

---

## 14. What to do first

1. **E0.3 + E1.2** — residual over roughness, and raw vs detrended. Existing data, about
   an hour. Answers whether the embedding machinery buys anything at all.
2. **Rewrite §1 and §3 of the report** around §2 of this document: drop the recurrence
   framing; state the d ≥ 2m+1 requirement, the moving-fibre obstruction, the
   unverifiability of the genericity set, and the NARMA bound on forecast skill. Every
   clause is citable to Stark 1999/2003. This makes the report's argument stronger, not
   weaker.
3. **Fix the specific errors** of §4 and §5: the abstract's line-constant claim; the
   "indistinguishable from stationary" wording; the seed comparison; the phase
   confound in the velocity comparison; the `weight_norm` determinism row; the duplicated
   "third observation"; the "by construction" claim for the logistic driver.
4. **Check the sinusoid's period against the logging interval** (§9.2). Five minutes,
   potentially upgrades a statistical limitation to a theorem-level one.
5. **E1.1** — 16 generic random projections of w, one run per configuration. The direct
   test of the theorem's own signature and of "the dimension of what".
6. **Dense-logged probe run** plus ‖z_i‖, probe labels and update cosines (§7). Closes
   the one open experiment of the function-space route.
7. **Tier 6** — the ensemble sweep. This is what turns the dimension chapter from a
   retraction into a result.

## 15. Sources

- J. Stark, *Delay Embeddings for Forced Systems. I. Deterministic Forcing*,
  J. Nonlinear Sci. **9** (1999) 255–332. Theorems 2.1, 2.2, Lemma 2.3, §3.2, Theorems
  3.1–3.4, §3.4, §2.3.
- J. Stark, D. S. Broomhead, M. E. Davies, J. Huke, *Delay Embeddings for Forced
  Systems. II. Stochastic Forcing*, J. Nonlinear Sci. **13** (2003) 519–577. §2.1–2.4,
  Theorems 2.1–2.6.
- E. Levina, P. Bickel, *Maximum Likelihood Estimation of Intrinsic Dimension*, NIPS 2004
  — i.i.d. sampling from a density on a d-manifold, Poisson approximation to neighbour
  counts.
- D. MacKay, Z. Ghahramani, *Comments on 'Maximum Likelihood Estimation of Intrinsic
  Dimension'* — pooled-likelihood correction; algebra re-derived in §3.2 above and not
  dependent on the source.
- J. Theiler, *Spurious dimension from correlation algorithms applied to limited
  time-series data*, Phys. Rev. A **34** (1986) 2427 — temporal-neighbour bias, the
  exclusion window.
- G. Sugihara et al., *Detecting Causality in Complex Ecosystems*, Science **338** (2012)
  496–500 — CCM. Supplementary material **not verified**; see §9.3.

Every **[computed]** number in this document is reproduced by
[`verify_report_0708.py`](verify_report_0708.py), which reads only files already in the
repo and prints the tables of §1.1, §4.2, §5.1, §5.3, §6.2 and §8.3 in that order:

```
python verify_report_0708.py
```
