"""The control suite of ``method.md`` §6, as a runnable registry.

The first results (§7) showed logit velocity separating a grokking run from a WD=0 run
by ~360x. That alone cannot distinguish the two live hypotheses, because the runs differ
*only* in weight decay:

    H1  the signal tracks impending generalization
    H2  the signal tracks weight decay reorganizing the function

``lowdata*`` is the experiment that separates them: weight decay stays at 1.0 and only
the training fraction drops, so the model memorizes but never generalizes. If its
velocity looks like ``grok``, H2 wins and the honest result is a negative one. If it
looks like ``wd0``, H1 survives its first real test.

``nogap`` guards the other end -- with plenty of data, generalization is immediate, and
a detector must not report a "lead" where there is no gap to lead. The ``seed*`` runs ask
whether any of this is a property of the configuration or of one lucky run.

    python controls.py --outdir /content/out               # the whole suite
    python controls.py lowdata15 nogap --outdir /content/out
    python controls.py --list
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONTROLS = {
    # --- the phenomenon, and the control we already have -------------------
    "grok": dict(
        run="mod_wd1", overrides={},
        expect="grokks ~13700",
        role="treatment: delayed generalization"),
    "wd0": dict(
        run="mod_wd0", overrides={},
        expect="never generalizes",
        role="control: no weight decay, memorizes forever"),

    # --- the confound-breaker: weight decay ON, generalization never -------
    "lowdata15": dict(
        run="mod_wd1", overrides={"fraction": "0.15"},
        expect="memorizes, never generalizes",
        role="DECISIVE: WD=1.0 held fixed, only the data is cut"),
    "lowdata20": dict(
        run="mod_wd1", overrides={"fraction": "0.20"},
        expect="memorizes, may or may not generalize",
        role="DECISIVE: second point on the data axis"),

    # --- the other end: no gap to lead -------------------------------------
    "nogap": dict(
        run="mod_wd1", overrides={"fraction": "0.8"},
        expect="generalizes almost immediately",
        role="control: no plateau, so no lead time to claim"),

    # --- is it a property of the configuration or of one run? --------------
    "grok_seed1": dict(
        run="mod_wd1", overrides={"seed": "1"},
        expect="grokks", role="seed replicate"),
    "grok_seed2": dict(
        run="mod_wd1", overrides={"seed": "2"},
        expect="grokks", role="seed replicate"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="controls to run (default: all)")
    parser.add_argument("--outdir", default="probe_logs")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="extra override applied to every run")
    args = parser.parse_args(argv)

    if args.list:
        width = max(len(k) for k in CONTROLS)
        for key, spec in CONTROLS.items():
            print(f"  {key:<{width}}  {spec['role']}")
            print(f"  {'':<{width}}  base={spec['run']} {spec['overrides']} "
                  f"-> expect: {spec['expect']}")
        return 0

    names = args.names or list(CONTROLS)
    unknown = [n for n in names if n not in CONTROLS]
    if unknown:
        parser.error(f"unknown control(s): {', '.join(unknown)}. Try --list.")

    import run_probe                            # deferred so --list needs no torch

    failures = []
    for i, name in enumerate(names, 1):
        spec = CONTROLS[name]
        print(f"\n{'=' * 70}\n[{i}/{len(names)}] {name} -- {spec['role']}\n"
              f"expect: {spec['expect']}\n{'=' * 70}", flush=True)

        argv_run = [spec["run"], "--outdir", args.outdir, "--tag", name,
                    "--force", "--progress-every", "100"]
        for key, value in spec["overrides"].items():
            argv_run += ["--set", f"{key}={value}"]
        for extra in args.overrides:
            argv_run += ["--set", extra]

        try:
            run_probe.main(argv_run)
        except Exception as exc:                            # noqa: BLE001 - keep going
            print(f"[{name}] FAILED: {exc}", flush=True)
            failures.append(name)

    print(f"\n{'=' * 70}")
    print(f"done: {len(names) - len(failures)}/{len(names)} succeeded"
          + (f", failed: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
