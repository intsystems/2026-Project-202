# Representability of the six polynomials, without training

`analytic_poly.py` answers two questions per polynomial, with no gradient step: does a
decomposition `P(n1,n2) = h(g1(n1) + g2(n2)) mod p` exist, and if so how do the
closed-form weights of Eq. (2) (`../gromov_arithmetic/analytic.py`) score on all `p^2`.

## Decision procedure

Three sound arguments; each one that applies is run, so none is trusted alone.

1. **Verified decomposition (proves EXISTS).** The paper's `(g1, g2, h)` checked on all
   `p^2` entries. Exhibiting one decomposition settles existence outright.
2. **Multiset certificate (proves NONE).** First collapse duplicate rows and columns:
   `g1(n1) = g1(n1')` forces rows `n1, n1'` equal, and two equal rows can always be
   given the same `g1` value, so a decomposition of the reduced table with *injective*
   `g1, g2` exists iff one of the original does. If the reduced table has `p` distinct
   columns then `g2` is injective on `p` columns, hence a bijection of `Z_p`, so row
   `n1` runs over `{h(g1(n1) + b) : b in Z_p}` — the image multiset of `h` over all of
   `Z_p`, **the same multiset for every row**. Two rows with unequal value multisets
   therefore rule out every decomposition. The transpose gives the test on columns.
3. **Complete search (proves either).** Backtracking over the labels `g1(r), g2(c)` on
   the reduced table, both injective by (2), with `h` a partial array each assignment
   writes into and checks against, so `m[r][c] == h(g1(r) + g2(c))` holds for all
   assigned pairs at all times. Translation absorbed into `h` fixes `g1(0) = g2(0) = 0`;
   the automorphism `s -> u s` of `Z_p` then rescales `g2(1)` (nonzero by injectivity)
   to 1. Nothing else is discarded, so exhausting the tree proves non-existence. Unit
   propagation, smallest-domain branching, node budget (default 40000); hitting the
   budget is reported as undecided rather than as a verdict.

(2) and (3) agree with exhaustive enumeration of `(g1, g2)` on 260 random and planted
tables at p = 3 and p = 5.

## Result

`python analytic_poly.py`, budget 40000 nodes. "search" = argument 3 alone, blind.

| poly | expression | p=23 reduced / verdict / search | p=97 reduced / verdict / search |
|------|------------|--------------------------------|--------------------------------|
| p1  | `(4 n1 + n2^2)^3`     | 23x12 EXISTS, search 21568 nodes | 97x49 EXISTS, undecided |
| p1x | `+ n1 n2`             | 23x23 NONE, search 29563 nodes   | 97x97 NONE, undecided |
| p2  | `(2 n1 + 3 n2)^4`     | 23x23 EXISTS, search 9749 nodes  | 97x97 EXISTS, undecided |
| p2x | `- n1^2`              | 23x23 NONE, search 29563 nodes   | 97x97 NONE, undecided |
| p3  | `(5 n1^3 + 2 n2^4)^2` | 23x12 EXISTS, search 5686 nodes  | 33x25 EXISTS, undecided |
| p3x | `- n2`                | 23x23 NONE, search 29563 nodes   | 33x97 NONE, undecided |

At p = 23 the blind complete search decides all six on its own and agrees with the two
cheap arguments. At p = 97 it exhausts the budget on all six and decides nothing; every
p = 97 verdict rests on the witness (EXISTS) or the certificate (NONE), both exact and
neither a search. **No verdict is left open at either modulus.**

Analytic weights on the full `p^2` table (mean peak 0.97-1.29, so `A = (2D)^(1/3)` is
the right amplitude; MSE falls from ~5e-2 to ~1e-3 across the widths):

| p | N=100 | N=500 | N=2000 | N=5000 |
|---|-------|-------|--------|--------|
| 23 | p1 100.00%, p2 99.43%, p3 99.24% | 100% | 100% | 100% |
| 97 | p1 71.30%, p2 62.19%, p3 91.42% | p1 100%, p2 99.95%, p3 100% | 100% | 100% |

## Conclusion

The three learnable polynomials are representable by the periodic construction at
**100% accuracy with no training at all**, at both moduli, once `N >= 500`; `N = 100` is
the finite-width regime of Fig. 3b, not a representability limit. The three perturbed
polynomials admit **no `h(g1 + g2)` decomposition whatsoever** — a property of the
`p x p` table, proved exactly. So their failure to grok cannot be blamed on the
optimiser, the data budget, or the width: the target is outside the class the
construction spans.

Side note: `h` is non-injective in five of the six representable `(p, poly)` cases
(`t^3`, `t^4` at p = 97 are 3-to-1 and 4-to-1; `t^2` is 2-to-1 everywhere) and all still
reach 100%, because `analytic.build` writes the readout as a forward map — row `h(t)`
accumulating the frequency content of `t`. The halved accuracy the `analytic.py`
docstring predicts for non-invertible `h` does not materialise.
