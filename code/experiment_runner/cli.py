"""Run the collection and analysis workflows used by the article experiments."""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import List, Optional, Sequence

from experiment_runner import device as device_mod
from experiment_runner import registry as reg
from experiment_runner import store as store_mod
from experiment_runner.context import build as build_context
from experiment_runner.parallel import default_jobs


def _fmt_minutes(minutes: float) -> str:
    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    if minutes < 90:
        return f"{minutes:.0f}m"
    return f"{minutes / 60:.1f}h"


def _provenance(exp: reg.Experiment) -> Optional[dict]:
    path = store_mod.runs_root() / exp.id / "provenance.json"
    if not path.exists():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _has_run(exp: reg.Experiment) -> bool:
    """Has this experiment produced a real result?

    A ``--fast`` run does not count. It writes a provenance record like any other, and
    treating it as done would let a smoke test at teatime quietly drop an experiment from
    the overnight campaign.
    """
    record = _provenance(exp)
    return bool(record) and record.get("status") == "ok" and not record.get("fast")


def _measured_minutes(exp: reg.Experiment) -> Optional[float]:
    """What the experiment actually took last time, if it has run.

    The declared cost is an estimate written by hand and is routinely wrong by a factor
    of two or three; a measured time is not. Both are shown, because a plan made before
    anything has run has only the estimate to go on.
    """
    record = _provenance(exp)
    if not record or record.get("fast"):
        return None
    seconds = record.get("wall_seconds")
    return float(seconds) / 60.0 if seconds else None


def _preflight(experiments: List[reg.Experiment], device: str) -> None:
    """Say what an unattended run is about to need, before it needs it.

    A campaign that fills the disk at three in the morning has wasted the night, and the
    trajectory sketches are large: a single sketched transformer run is 60 to 95 MB and a
    full regeneration writes upwards of a gigabyte.
    """
    free = device_mod.free_disk_gb(str(store_mod.repo_root()))
    gpu_steps = [e for e in experiments if e.device == reg.GPU]
    if gpu_steps and device_mod.resolve(device) == "cpu":
        print(f"note: {len(gpu_steps)} of {len(experiments)} steps want a GPU and none was "
              f"found; they will be skipped unless --allow-cpu is given")
    if free < 2.0:
        print(f"warning: {free:.1f} GB free. A full regeneration writes about 1.2 GB into "
              f"runs/, and a run that fills the disk loses the campaign, not just the step.")
    totals = reg.cost(experiments)
    print(f"{len(experiments)} step(s), about {_fmt_minutes(totals['cpu_minutes'])} of CPU "
          f"work and {_fmt_minutes(totals['gpu_minutes'])} of GPU work, {free:.1f} GB free\n")


def cmd_list(args: argparse.Namespace) -> int:
    experiments = reg.select(args.targets) if args.targets else list(reg.load().values())
    experiments.sort(key=lambda e: (e.tier, e.id))
    width = max((len(e.id) for e in experiments), default=10)
    print(f"{'id':<{width}}  dev  {'cost':>6}  ran  paper")
    print("-" * (width + 34))
    for exp in experiments:
        print(f"{exp.id:<{width}}  {exp.device:<3}  {_fmt_minutes(exp.minutes):>6}  "
              f"{'yes' if _has_run(exp) else ' - ':<3}  {', '.join(exp.paper)}")
    totals = reg.cost(experiments)
    print(f"\n{len(experiments)} experiments   "
          f"cpu {_fmt_minutes(totals['cpu_minutes'])}   "
          f"gpu {_fmt_minutes(totals['gpu_minutes'])}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    targets = reg.select(args.targets) if args.targets else list(reg.load().values())
    ordered = reg.order(targets)
    for index, exp in enumerate(ordered, 1):
        mark = "ok " if _has_run(exp) else "   "
        measured = _measured_minutes(exp)
        cost = (f"{_fmt_minutes(measured):>6} measured" if measured is not None
                else f"{_fmt_minutes(exp.minutes):>6} est.    ")
        print(f"{index:3}. {mark} {exp.id:<32} {exp.device:<3} {cost}  {exp.title}")
    totals = reg.cost(ordered)
    print(f"\n{len(ordered)} steps   cpu {_fmt_minutes(totals['cpu_minutes'])}   "
          f"gpu {_fmt_minutes(totals['gpu_minutes'])}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.targets and not args.all:
        raise SystemExit("name an experiment, a prefix, or pass --all")
    targets = reg.select(args.targets) if args.targets else list(reg.load().values())
    ordered = reg.order(targets, include_needs=not args.no_deps)
    # An experiment named on the command line always runs. A prerequisite pulled in
    # behind one runs only if it has not already, unless --force.
    if not args.force:
        ordered = [e for e in ordered if e in targets or not _has_run(e)]

    jobs = default_jobs(args.jobs)
    _preflight(ordered, args.device)
    failures: List[str] = []
    for index, exp in enumerate(ordered, 1):
        if exp.device == reg.GPU and device_mod.resolve(args.device) == "cpu" and not args.allow_cpu:
            print(f"[{index}/{len(ordered)}] {exp.id}: needs a GPU, skipping "
                  f"(pass --allow-cpu to run it anyway)", flush=True)
            continue

        print(f"[{index}/{len(ordered)}] {exp.id}: {exp.title}", flush=True)
        started = time.time()
        ctx = build_context(exp.id, device=args.device, jobs=jobs, seed=args.seed,
                            fast=args.fast, options=_parse_options(args.set))
        try:
            exp.fn(ctx)
        except Exception:
            ctx.store.close("failed")
            traceback.print_exc()
            failures.append(exp.id)
            if not args.keep_going:
                print(f"\nstopped at {exp.id}. Re-run it alone once fixed:\n"
                      f"  python -m experiment_runner.cli run {exp.id}")
                return 1
            continue
        ctx.store.close("ok")
        print(f"     done in {_fmt_minutes((time.time() - started) / 60)} "
              f"-> runs/{exp.id}/", flush=True)

        # An experiment declares the outputs the article reads. If it did not write one,
        # say so now rather than at promotion time: without --promote the mismatch would
        # surface only after the campaign, and with it the run would already have taken
        # its hours before failing on a filename.
        undeclared = [name for name in exp.promotes
                      if not (ctx.store.dir / name).exists()]
        if undeclared:
            print(f"     WARNING: declared but not written: {', '.join(undeclared)}",
                  flush=True)

    if failures:
        print(f"\n{len(failures)} of {len(ordered)} failed: {', '.join(failures)}")
        print("Everything else finished and is skipped on the next run. To retry only "
              "these:\n  python -m experiment_runner.cli run " + " ".join(failures))
        return 1
    return 0


def _parse_options(pairs: Optional[Sequence[str]]) -> dict:
    options = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        options[key.strip()] = _coerce(value.strip())
    return options


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("none", "null"):
        return None
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment_runner.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_targets(sub, help_text):
        sub.add_argument("targets", nargs="*", metavar="ID", help=help_text)
        sub.add_argument("--all", action="store_true", help="every registered experiment")

    listing = subparsers.add_parser("list", help="what exists and what it costs")
    add_targets(listing, "experiment id or prefix")
    listing.set_defaults(func=cmd_list)

    plan = subparsers.add_parser("plan", help="the order a regeneration would take")
    add_targets(plan, "experiment id or prefix")
    plan.set_defaults(func=cmd_plan)

    run = subparsers.add_parser("run", help="run experiments")
    add_targets(run, "experiment id or prefix")
    run.add_argument("--device", default="auto", help="auto, cpu, cuda:0")
    run.add_argument("--jobs", type=int, default=0, help="worker processes (0 = cores - 1)")
    run.add_argument("--seed", type=int, default=0, help="base seed for every stream")
    run.add_argument("--fast", action="store_true",
                     help="smoke test: tiny grids, for checking the plumbing")
    run.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    run.add_argument("--no-deps", action="store_true", help="do not run prerequisites")
    run.add_argument("--keep-going", action="store_true", help="continue past a failure")
    run.add_argument("--allow-cpu", action="store_true", help="run GPU experiments on the CPU")
    run.add_argument("--set", action="append", metavar="KEY=VALUE",
                     help="override an experiment option")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    if getattr(args, "all", False):
        args.targets = []
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
