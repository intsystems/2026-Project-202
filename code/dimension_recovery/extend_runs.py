"""Settle the censoring question: run the controls far past the current budget.

``exp3_censoring.py`` can show that ``lowdata20`` is still above chance and still rising
at step 20000, and that ``lowdata15`` is below chance and falling. It cannot show what
either does at step 200000, and no analysis of a 20000-step log ever will. This is the
experiment that decides it.

Two modes, because this machine has no GPU:

    python extend_runs.py --plan                 # print the commands, no torch needed
    python extend_runs.py --analyse DIR          # read the resulting CSVs, no torch needed
    python extend_runs.py --run --outdir DIR     # actually train (needs torch + a GPU)

Design notes that matter for the answer being usable:

* **Budget.** 200000 steps is 10x the current one and 16x the longest observed gap
  (12070). Power et al. (2022) report modular-arithmetic grokking times growing sharply
  as the training fraction falls, with generalisation at small fractions arriving after
  1e5 steps or not at all, so 2e4 is not a budget from which "never" can be concluded.
* **Seeds.** Three per configuration. One run cannot separate "this configuration does
  not grok" from "this seed did not grok", and the sweep in this repo already shows the
  gap varying by a factor of 4.3 across seeds of one configuration.
* **No protocol change.** Same optimiser, schedule, batch size, logging cadence. Only
  ``max_steps`` and ``seed`` move. A longer run with a different learning-rate schedule
  answers a different question.
* **Pre-registered readings**, so the result cannot be reinterpreted afterwards:
    - validation accuracy reaches 0.95  -> ``late_grokking``. The run was censored, it
      must be removed from the negative set, and every false-alarm count computed
      against it in this project is void.
    - validation accuracy stays at or below chance for the whole 200000 steps ->
      ``extended_non_grokking``. It is a real counterexample and may be used as one.
    - anything between -> ``insufficient_budget`` again, and the honest report is that
      specificity remains unmeasured.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P_MOD, VAL_BATCH = 113, 512
CHANCE = 1.0 / P_MOD

# name -> (base run key, overrides that define it, why it is being extended)
#
# These reproduce ../prediction_improved/controls.py exactly; they are spelled out here
# rather than imported because controls.py hardcodes --tag to the control's name, so all
# three seeds of one control would overwrite each other's log. Going through
# grokking_train/train.py instead also drops the logit probe, which is pure overhead at
# this length: the question is when validation accuracy moves, nothing more.
TARGETS = {
    "lowdata15": ("mod_wd1", {"fraction": 0.15},
                  "below chance and falling at 20000: expected to stay a real negative"),
    "lowdata20": ("mod_wd1", {"fraction": 0.20},
                  "above chance and still rising at 20000: censored, must be resolved"),
    "wd0": ("mod_wd0", {},
            "frozen at 20000: expected to stay a real negative, and cheap to confirm"),
    # positive control: a configuration known to grok at ~13700. If this one fails to
    # grok at 200000 the harness is broken and no negative result below means anything.
    "grok_positive": ("mod_wd1", {},
                      "positive control -- must still grok, or the sweep is void"),
}
SEEDS = (0, 1, 2)
MAX_STEPS = 200_000
TRAIN = "../grokking_train/train.py"


def commands(outdir):
    for name, (base, over, why) in TARGETS.items():
        for seed in SEEDS:
            tag = f"{name}_s{seed}"
            sets = " ".join(f"--set {k}={v}" for k, v in over.items())
            yield name, why, (
                f"python {TRAIN} {base} --outdir {outdir} {sets} "
                f"--set max_steps={MAX_STEPS} --set seed={seed} --set init_seed={seed} "
                f"--set csv={tag}_train.csv --force")


def plan(outdir):
    print(__doc__.split("Design notes")[0].strip())
    print("\nCommands (each is one configuration at one seed):\n")
    last = None
    for name, why, cmd in commands(outdir):
        if name != last:
            print(f"  # {name}: {why}")
            last = name
        print(f"  {cmd}")
    print(f"\n  # then, back on any machine:")
    print(f"  python extend_runs.py --analyse {outdir}")
    print(f"\nCost: {len(TARGETS) * len(SEEDS)} runs x {MAX_STEPS} steps. The canonical")
    print("20000-step run takes about 3.4 minutes on a T4, so this is roughly 7 GPU-hours.")


def analyse(outdir):
    outdir = Path(outdir)
    files = sorted(outdir.glob("*_train.csv"))
    if not files:
        print(f"no *_train.csv in {outdir}")
        return 1
    print(f"  {'file':<28}{'steps':>9}{'final val':>11}{'x chance':>10}{'max val':>9}"
          f"{'t_gen':>9}   reading")
    rows = []
    for f in files:
        d = pd.read_csv(f)
        s, v = d["step"].to_numpy(), d["val_acc"].to_numpy()
        tail = v[s >= s.max() - 5000]
        acc = float(tail.mean())
        hit = np.flatnonzero(pd.Series(v).rolling(5, min_periods=1, center=True)
                             .mean().to_numpy() >= 0.95)
        t_gen = int(s[hit[0]]) if len(hit) else None
        if t_gen is not None:
            reading = "late_grokking -- REMOVE FROM THE NEGATIVE SET"
        elif acc <= CHANCE:
            reading = "extended_non_grokking -- usable as a counterexample"
        else:
            reading = "insufficient_budget -- still unresolved"
        print(f"  {f.name:<28}{s.max():>9}{acc:>11.5f}{acc / CHANCE:>10.2f}"
              f"{v.max():>9.3f}{str(t_gen):>9}   {reading}")
        rows.append({"file": f.name, "steps": int(s.max()), "final_val": acc,
                     "x_chance": acc / CHANCE, "max_val": float(v.max()),
                     "t_gen": t_gen, "reading": reading})
    frame = pd.DataFrame(rows)
    out = HERE / "results" / "extend_summary.csv"
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\n  written to {out}")
    late = int(frame.reading.str.startswith("late").sum())
    if late:
        print(f"\n  {late} run(s) generalised after the old budget. Every false-alarm")
        print("  count in report_0708_experiments.md section 9 and exp4_criterion.py")
        print("  that used those runs as negatives has to be recomputed.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the commands and exit")
    ap.add_argument("--analyse", metavar="DIR", help="summarise logs already produced")
    ap.add_argument("--run", action="store_true", help="train now (needs torch + GPU)")
    ap.add_argument("--outdir", default="extended_logs")
    args = ap.parse_args(argv)

    if args.analyse:
        return analyse(args.analyse)
    if not args.run:
        plan(args.outdir)
        return 0

    sys.path.insert(0, str(HERE.parent / "grokking_train"))
    import train                                             # needs torch
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for name, (base, over, why) in TARGETS.items():
        for seed in SEEDS:
            tag = f"{name}_s{seed}"
            argv_run = [base, "--outdir", args.outdir, "--force"]
            for k, v in over.items():
                argv_run += ["--set", f"{k}={v}"]
            argv_run += ["--set", f"max_steps={MAX_STEPS}", "--set", f"seed={seed}",
                         "--set", f"init_seed={seed}", "--set", f"csv={tag}_train.csv"]
            print(f"\n=== {tag}: {why} ===", flush=True)
            try:
                train.main(argv_run)
            except Exception as exc:                          # noqa: BLE001 - keep going
                print(f"[{tag}] FAILED: {exc}", flush=True)
    return analyse(args.outdir)


if __name__ == "__main__":
    raise SystemExit(main())
