"""Can the network *represent* each polynomial at all?  Two questions, no training.

Hypothesis 5.1 of arXiv:2406.03495 says the two-layer quadratic MLP generalises on
``P(n1, n2)`` exactly when ``P = h(g1(n1) + g2(n2)) mod p``.  Training runs conflate
that claim with everything else the optimiser does, so this file separates the two
halves and settles both without a gradient step:

1.  For the three learnable polynomials the decomposition is known, so the closed-form
    weights of Eq. (2) can be written down directly (``analytic.build``) and scored on
    the full ``p^2`` table.  If they hit 100%, representability is not the obstacle for
    the perturbed twins either -- the difference has to be the target itself.

2.  For the three perturbed polynomials the claim is that *no* decomposition exists.
    That is a statement about the ``p x p`` table alone, and it is decidable.
    ``_multiset_certificate`` (cheap, one-sided) and ``_search`` (complete) both decide
    it; their docstrings give the soundness argument, and both were checked against
    exhaustive enumeration of ``(g1, g2)`` on random tables at p = 3 and p = 5.

The decomposition for the learnable three, all arithmetic mod p:

    (4 n1 + n2^2)^3       g1 = 4 n1,      g2 = n2^2,      h = t^3
    (2 n1 + 3 n2)^4       g1 = 2 n1,      g2 = 3 n2,      h = t^4
    (5 n1^3 + 2 n2^4)^2   g1 = 5 n1^3,    g2 = 2 n2^4,    h = t^2

    python analytic_poly.py
    python analytic_poly.py --p 23 --nodes 100000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import polynomials as P
from _core import Config           # also puts ../gromov_arithmetic on sys.path
from analytic import build, evaluate

DECOMPOSITIONS = {
    # Degrees stay under 5 and p under 100, so int64 cannot overflow here and the
    # object-dtype dance of polynomials.py is unnecessary.
    "p1": (lambda n: 4 * n, lambda n: n ** 2, lambda t: t ** 3),
    "p2": (lambda n: 2 * n, lambda n: 3 * n, lambda t: t ** 4),
    "p3": (lambda n: 5 * n ** 3, lambda n: 2 * n ** 4, lambda t: t ** 2),
}


def table(name, p):
    """The full ``p x p`` answer table."""
    n1, n2 = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return P.evaluator(name, p)(n1.reshape(-1), n2.reshape(-1)).reshape(p, p)


def verify(name, p):
    """Check the claimed ``(g1, g2, h)`` on every one of the ``p^2`` entries.

    Exhibiting one decomposition settles existence, so a pass here is a complete proof
    that the polynomial has the Hypothesis 5.1 form -- nothing else is needed for YES.
    """
    g1, g2, h = DECOMPOSITIONS[name]
    n1, n2 = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    s = (np.asarray(g1(n1)) + np.asarray(g2(n2))) % p
    return bool((np.asarray(h(s)) % p == table(name, p)).all())


# ---------------------------------------------------------------------------
# deciding whether a decomposition exists
# ---------------------------------------------------------------------------

def _reduce(t):
    """Collapse duplicate rows and duplicate columns.

    ``g1(n1) = g1(n1')`` forces rows ``n1`` and ``n1'`` to be equal, and conversely two
    equal rows can always be given the same ``g1`` value without changing any table
    entry.  So a decomposition of the deduplicated table with *injective* ``g1`` and
    ``g2`` exists iff one of the original table does, and injectivity is what gives the
    search below anything to prune with.
    """
    return np.unique(np.unique(t, axis=0).T, axis=0).T


def _multiset_certificate(m, p):
    """A cheap one-sided proof of non-existence; returns the reason, or None.

    If the table has ``p`` distinct columns then ``g2`` is injective on them and hence a
    bijection of ``Z_p``, so row ``n1`` runs over ``{h(g1(n1) + b) : b in Z_p}`` -- the
    image of ``h`` over all of ``Z_p``, the *same* multiset of values for every row.
    Two rows with different value multisets therefore admit no decomposition at all.
    The transposed statement tests columns.  Nothing is concluded when the test passes.
    """
    def uniform(a):
        return len({tuple(np.bincount(r, minlength=p)) for r in a}) == 1

    rows, cols = m.shape
    if cols == p and not uniform(m):
        return "p distinct columns, rows have unequal value multisets"
    if rows == p and not uniform(m.T):
        return "p distinct rows, columns have unequal value multisets"
    return None


def _search(m, p, budget):
    """Complete backtracking search for ``(g1, g2, h)`` on the reduced table.

    Variables are the labels ``a[r] = g1(r)`` and ``b[c] = g2(c)``, both injective by
    ``_reduce``; ``h`` is a partial array that every assignment writes into and checks
    against, so the invariant ``m[r][c] == h(a[r] + b[c])`` holds for all assigned pairs
    at all times.  Two normalisations cost nothing: translating ``g1``, ``g2`` by
    constants absorbed into ``h`` fixes ``a[0] = b[0] = 0``, and the automorphism
    ``s -> u s`` of ``Z_p`` then rescales ``b[1]`` (nonzero by injectivity) to 1.  What
    remains is exhaustive, so returning False is a proof of non-existence.

    Returns ``(verdict, nodes)`` with verdict True / False / None, the last meaning the
    node budget ran out and nothing was decided.
    """
    n_rows, n_cols = m.shape
    # Most-distinct-values first: those rows and columns constrain h the hardest.
    r_ord = np.argsort([-np.unique(m[i]).size for i in range(n_rows)], kind="stable")
    c_ord = np.argsort([-np.unique(m[:, j]).size for j in range(n_cols)], kind="stable")
    m = m[np.ix_(r_ord, c_ord)]

    h = np.full(p, -1, dtype=np.int64)
    alpha = np.full(n_rows, -1, dtype=np.int64)
    beta = np.full(n_cols, -1, dtype=np.int64)
    used_a = np.zeros(p, dtype=bool)
    used_b = np.zeros(p, dtype=bool)
    grid = np.arange(p)
    nodes = 0

    def assign(is_row, i, v, trail):
        if is_row:
            alpha[i], used_a[v] = v, True
            pairs = [((v + beta[c]) % p, m[i, c]) for c in np.flatnonzero(beta >= 0)]
        else:
            beta[i], used_b[v] = v, True
            pairs = [((alpha[r] + v) % p, m[r, i]) for r in np.flatnonzero(alpha >= 0)]
        for s, want in pairs:
            if h[s] < 0:
                h[s] = want
                trail.append(s)
            elif h[s] != want:
                return False
        return True

    def retract(is_row, i, v, trail):
        for s in trail:
            h[s] = -1
        if is_row:
            alpha[i], used_a[v] = -1, False
        else:
            beta[i], used_b[v] = -1, False

    def domains():
        """Smallest live domain as ``(size, is_row, index, values)``; None if any is empty."""
        best = None
        done = True
        for is_row in (True, False):
            todo = np.flatnonzero((alpha if is_row else beta) < 0)
            if not todo.size:
                continue
            done = False
            fixed = np.flatnonzero((beta if is_row else alpha) >= 0)
            ok = np.repeat((~used_a if is_row else ~used_b)[None, :], todo.size, axis=0)
            if fixed.size:
                other = (beta if is_row else alpha)[fixed]
                seen = h[(grid[:, None] + other[None, :]) % p]
                want = m[np.ix_(todo, fixed)] if is_row else m[np.ix_(fixed, todo)].T
                ok &= ((seen < 0)[None] | (seen[None] == want[:, None, :])).all(2)
            sizes = ok.sum(1)
            j = int(sizes.argmin())
            if sizes[j] == 0:
                return None, False
            if best is None or sizes[j] < best[0]:
                best = (int(sizes[j]), is_row, int(todo[j]), np.flatnonzero(ok[j]))
        return best, done

    def descend():
        """One node: propagate every forced label, then branch on the smallest domain."""
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise TimeoutError
        forced = []
        try:
            while True:
                best, done = domains()
                if done:
                    return True
                if best is None:
                    return False
                if best[0] > 1:
                    break
                _, is_row, i, vals = best
                trail = []
                forced.append((is_row, i, int(vals[0]), trail))
                if not assign(is_row, i, int(vals[0]), trail):
                    return False
            _, is_row, i, vals = best
            for v in vals:
                trail = []
                if assign(is_row, i, int(v), trail) and descend():
                    return True
                retract(is_row, i, int(v), trail)
            return False
        finally:
            # The verdict is the only output, so the state is always unwound.
            for is_row, i, v, trail in reversed(forced):
                retract(is_row, i, v, trail)

    root = []
    assign(True, 0, 0, root)
    assign(False, 0, 0, root)
    if n_cols > 1 and not assign(False, 1, 1, root):
        return False, nodes
    try:
        return descend(), nodes
    except TimeoutError:
        return None, nodes


def decide(name, p, budget):
    """Verdict on ``P = h(g1 + g2)``, plus every argument that bears on it.

    Three sound arguments, cheapest first: a verified decomposition proves EXISTS, the
    multiset certificate proves NONE, and the complete search proves either.  All three
    are run when they apply, so the cheap ones are cross-checked rather than trusted;
    ``budget = 0`` drops the search.
    """
    m = _reduce(table(name, p))
    verdict, notes = None, []

    if name in DECOMPOSITIONS and verify(name, p):
        verdict = True
        notes.append("(g1,g2,h) verified on all p^2 entries")
    why = _multiset_certificate(m, p)
    if why is not None:
        verdict = False
        notes.append(f"multiset certificate: {why}")

    if budget:
        t0 = time.time()
        found, nodes = _search(m, p, budget)
        label = {True: "search EXISTS", False: "search NONE",
                 None: f"search undecided at {budget} nodes"}[found]
        notes.append(f"{label} ({nodes} nodes, {time.time() - t0:.0f}s)")
        if verdict is None:
            verdict = found
    return verdict, m.shape, "; ".join(notes)


# ---------------------------------------------------------------------------
# the analytic weights
# ---------------------------------------------------------------------------

def construct(name, p, width):
    """Score the Eq. (2) weights for ``name`` at width ``width`` on the full table."""
    g1, g2, h = DECOMPOSITIONS[name]
    cfg = Config(p=p, width=width, task=name)
    w1, w2 = build(cfg, [g1, g2], h)
    return evaluate(cfg, P.evaluator(name, p), w1, w2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, nargs="+", default=[23, 97])
    ap.add_argument("--widths", type=int, nargs="+", default=[100, 500, 2000, 5000])
    ap.add_argument("--nodes", type=int, default=40_000,
                    help="node budget for the complete search; 0 skips it")
    args = ap.parse_args()

    for p in args.p:
        print(f"\n=== p = {p} ===\n")
        head = f"{'poly':<5}{'expression':<28}{'rows':>5}{'cols':>5}  {'h(g1+g2)':<10}how"
        print(head)
        print("-" * (len(head) + 34))
        exists = []
        for name in ("p1", "p1x", "p2", "p2x", "p3", "p3x"):
            ok, (rows, cols), how = decide(name, p, args.nodes)
            verdict = {True: "EXISTS", False: "NONE", None: "?"}[ok]
            print(f"{name:<5}{P.EXPRESSIONS[name]:<28}{rows:>5}{cols:>5}  "
                  f"{verdict:<10}{how}", flush=True)
            if ok:
                exists.append(name)

        print(f"\nanalytic weights, A = (2D)^(1/3) = {(4 * p) ** (1 / 3):.3f}\n")
        head = f"{'poly':<5}{'N':>6}{'acc':>9}{'MSE':>12}{'peak':>9}{'|W|':>8}"
        print(head)
        print("-" * len(head))
        for name in exists:
            for width in args.widths:
                r = construct(name, p, width)
                print(f"{name:<5}{width:>6}{r['acc']:>8.2%}{r['mse']:>12.3e}"
                      f"{r['mean_peak']:>9.3f}{r['weight_norm']:>8.3f}", flush=True)
            print()


if __name__ == "__main__":
    main()
