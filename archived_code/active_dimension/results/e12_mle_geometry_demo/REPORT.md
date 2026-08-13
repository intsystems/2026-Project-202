# MLE versus roughness: two visual controls

## Question

Does the delay-embedding MLE measure geometry that the one-line roughness statistic cannot see?

## Experiment A: four clocks versus one clock with four hands

Both scalar logs are sums of four sinusoids and are matched to the same roughness. In one case the
four phases are independent (dimension 4); in the other every frequency is a harmonic of one
master phase (dimension 1). The scheduled trace is 4D -> 1D -> 4D.

| arm | truth | MG | roughness | PRdelay | specPR |
| --- | --- | --- | --- | --- | --- |
| four independent clocks | 4.000 | 3.932 | 0.090 | 2.213 | 4.777 |
| one clock, four hands | 1.000 | 1.116 | 0.090 | 4.090 | 6.160 |

Result: MLE follows the number of independent phases while roughness is constant. The linear
delay PR and spectral PR do not give the correct count in both arms. This is the simplest answer
to "why MLE rather than roughness?": roughness measures how rapidly the scalar moves, not how many
independent clocks generate it.

## Experiment B: same four clocks, different instrument scales

The same 4D scalar log is passed through invertible monotone maps. The state dynamics and their
dimension are unchanged, but the shape and roughness of the recorded signal change.

| observer | MG | roughness | PRdelay |
| --- | --- | --- | --- |
| identity | 3.932 | 0.090 | 2.213 |
| x + 0.1 x^7 | 4.032 | 0.165 | 4.261 |
| x + 0.25 x^3 | 4.011 | 0.097 | 2.332 |
| x + x^3 | 4.037 | 0.109 | 2.589 |
| x + x^5 | 3.983 | 0.145 | 3.621 |

Result: roughness spans 84%; median MLE spans only
0.104 components and remains near four.

## Pre-specified verdict

- Geometry switch: **PASS**. Required MLE 4 -> 1 -> 4
  and less than 5% roughness change.
- Observer-scale control: **PASS**. Required at least 30%
  roughness span, every median MLE within 0.5 of four, and less than 0.5 MLE span.

## What this establishes—and what it does not

It establishes that MLE can carry phase-geometric information absent from roughness and can be
stable when roughness changes for purely observational reasons. It does not establish that MLE is
the best detector of grokking: on the current parameter-norm logs roughness detects the transition
more strongly. The clean claim is complementary: roughness detects loss of innovation; MLE can,
in an admissible recurrent setting, count independent degrees of freedom.
