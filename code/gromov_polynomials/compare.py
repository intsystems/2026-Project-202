"""Put the finished runs next to Table 2 of arXiv:2406.03495 and report the gap.

Reading a reproduction by eye invites grading on a curve, so the comparison is made
mechanically here.  Two judgements are made per run:

*generalised*  final validation accuracy >= 95%.
*chance*       final validation accuracy <= the majority-class share of the label
               distribution, times 1.5.

The second is why this file exists rather than a printf in ``run_poly.py``.  Chance is
not ``1/p`` for these targets: ``(2 n1 + 3 n2)^4 mod 97`` only lands on fourth powers,
of which there are 25, so a constant predictor scores about 4% and the paper's 3.93%
for the perturbed version of that polynomial is *at* chance rather than slightly above
it.  Comparing against ``1/p`` would make three of the six failures look like partial
successes.

    python compare.py --results ./results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polynomials as P


def load(results: Path):
    path = results / "summary.json"
    if not path.exists():
        raise SystemExit(f"no summary.json under {results} -- fetch the run outputs first")
    return {r["key"]: r for r in json.loads(path.read_text())}


def verdict(rec):
    chance = rec["majority_share"]
    if rec["final_val_acc"] >= 0.95:
        return "generalised"
    if rec["final_val_acc"] <= 1.5 * chance:
        return "chance"
    return "partial"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="./results")
    args = ap.parse_args()
    runs = load(Path(args.results))

    for arm, label in (("f", "paper-faithful: Adam lr 5e-3, wd 5.0, N 5000, alpha 0.5"),
                       ("g", "Gromov no-wd: full-batch GD, wd 0, N 500")):
        present = [k for k in runs if k.startswith(f"{arm}_")]
        if not present:
            continue
        print(f"\n=== {label} ===")
        head = (f"{'polynomial':<30}{'p':>4}{'form':>11}{'paper':>8}{'ours':>8}"
                f"{'chance':>8}{'grok@':>8}  verdict")
        print(head)
        print("-" * len(head))
        for p in (97, 23):
            for name in P.POLYNOMIALS:
                rec = runs.get(f"{arm}_{name}_p{p}")
                if rec is None:
                    continue
                form = "h(g1+g2)" if rec["learnable"] else "perturbed"
                grok = "never" if rec["t_grok"] is None else str(rec["t_grok"])
                print(f"{P.EXPRESSIONS[name]:<30}{p:>4}{form:>11}"
                      f"{rec['paper_test_acc']:>7.1%}{rec['final_val_acc']:>8.1%}"
                      f"{rec['majority_share']:>8.1%}{grok:>8}  {verdict(rec)}")
            print()

    # Tallied per arm, not over the pooled set: the two arms use different optimisers,
    # and the `f_*` arm is invalid in this parametrisation (report.md Result 4), so a
    # pooled score would blame Hypothesis 5.1 for a configuration error. "partial" is
    # counted separately rather than as agreement -- g_p3x_p97 reaches 73% test accuracy
    # on a provably non-decomposable target, and calling that "agrees" would hide the
    # one entry in the table that the hypothesis does not cleanly account for.
    for arm, label in (("g", "Gromov no-wd"), ("f", "paper-faithful")):
        agree = disagree = partial = 0
        for key, rec in sorted(runs.items()):
            if not key.startswith(f"{arm}_"):
                continue
            got, expect_gen = verdict(rec), rec["learnable"]
            if got == "partial" and not expect_gen:
                partial += 1
                print(f"  PARTIAL  {key}: perturbed target at {rec['final_val_acc']:.2%} "
                      f"(chance {rec['majority_share']:.2%}) -- neither generalised nor chance")
            elif (got == "generalised") == expect_gen:
                agree += 1
            else:
                disagree += 1
                print(f"  MISMATCH {key}: expected "
                      f"{'generalised' if expect_gen else 'chance'}, got {got} "
                      f"({rec['final_val_acc']:.2%})")
        total = agree + disagree + partial
        if total:
            print(f"{label}: {agree}/{total} runs agree with Hypothesis 5.1 "
                  f"(learnable <=> generalises)"
                  + (f", {partial} partial" if partial else "") + ".")


if __name__ == "__main__":
    main()
