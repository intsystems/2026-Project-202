"""The command line.

    python -m actdim list                 what exists, what it costs, what has run
    python -m actdim plan --all           the order a full regeneration would take
    python -m actdim run sys.matrix       run one experiment, or a prefix, or --all
    python -m actdim promote sys.matrix   copy its article-facing outputs into data/
    python -m actdim verify               check data/ against the manifest
    python -m actdim diff sys.matrix      what moved against the archived results
    python -m actdim bootstrap            seed data/ from ../archived_code, marked as such
    python -m actdim doctor               environment, device and disk

Every command works from any directory: paths resolve against the package, not the shell.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

from . import device as device_mod
from . import registry as reg
from . import store as store_mod
from .context import build as build_context
from .parallel import default_jobs


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
    record = _provenance(exp)
    return bool(record) and record.get("status") == "ok"


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
                  f"(pass --allow-cpu to run it anyway)")
            continue

        print(f"[{index}/{len(ordered)}] {exp.id}: {exp.title}")
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
                      f"  python -m actdim run {exp.id}")
                return 1
            continue
        ctx.store.close("ok")
        print(f"     done in {_fmt_minutes((time.time() - started) / 60)} "
              f"-> runs/{exp.id}/")
        if args.promote and exp.promotes:
            if args.fast:
                # A --fast run computes the smallest grid that exercises every branch.
                # Its outputs have the right columns and the wrong numbers, and data/ is
                # what the article reads, so the two must never meet.
                print("     not promoted: --fast output is a plumbing check, not a result")
            else:
                store_mod.promote(exp.id, exp.promotes)
                print(f"     promoted {len(exp.promotes)} file(s) to data/{exp.id}/")

    if failures:
        print(f"\n{len(failures)} of {len(ordered)} failed: {', '.join(failures)}")
        print("Everything else finished and is skipped on the next run. To retry only "
              "these:\n  python -m actdim run " + " ".join(failures))
        return 1
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    for exp in reg.select(args.targets):
        if not exp.promotes:
            continue
        promoted = store_mod.promote(exp.id, exp.promotes)
        for rel in promoted:
            print(f"{exp.id} -> data/{rel}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = store_mod.verify()
    for rel in result["changed"]:
        print(f"CHANGED  {rel}")
    for rel in result["missing"]:
        print(f"MISSING  {rel}")
    print(f"\n{len(result['ok'])} unchanged, {len(result['changed'])} changed, "
          f"{len(result['missing'])} missing")
    return 1 if result["changed"] or result["missing"] else 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from . import archive

    targets = ([e.id for e in reg.select(args.targets)] if args.targets
               else sorted(archive.BASELINE))
    seeded = archive.bootstrap(targets)
    for rel in sorted(seeded):
        print(f"archived -> data/{rel}")
    absent = archive.missing()
    print(f"\n{len(seeded)} file(s) seeded from ../archived_code, marked "
          f"'source: archived' in the manifest.")
    if absent:
        print(f"{len(absent)} archived file(s) named by the mapping are not on disk:")
        for experiment, name in absent:
            print(f"  {experiment}/{name}")
    print("Re-running an experiment and promoting it clears the mark.")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from .compare import compare_experiment, format_report

    moved = 0
    for exp in reg.select(args.targets):
        try:
            reports = compare_experiment(exp.id, tolerance=args.tolerance)
        except FileNotFoundError as error:
            print(f"{exp.id}: {error}")
            continue
        if not reports:
            continue
        print(f"\n=== {exp.id} ===")
        for report in reports:
            print(format_report(report))
            moved += len(report.get("changed", []))
    print(f"\n{moved} column(s) differ from the archived results.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .provenance import git_state, library_versions

    print("device")
    for key, value in device_mod.describe(args.device).items():
        print(f"  {key:<16} {value}")
    print(f"  {'cores':<16} {default_jobs()} usable")
    print(f"  {'free disk':<16} {device_mod.free_disk_gb(str(store_mod.repo_root())):.1f} GB")
    print("\nlibraries")
    for key, value in library_versions().items():
        print(f"  {key:<16} {value}")
    print("\ngit")
    for key, value in git_state().items():
        print(f"  {key:<16} {value}")
    experiments = reg.load()
    ran = sum(1 for e in experiments.values() if _has_run(e))
    print(f"\nexperiments        {ran}/{len(experiments)} have run")
    manifest = store_mod.load_manifest()
    print(f"tracked data       {len(manifest.get('files', {}))} files in data/manifest.json")
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
    parser = argparse.ArgumentParser(prog="actdim", description=__doc__,
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
    run.add_argument("--promote", action="store_true", help="copy outputs to data/ when done")
    run.add_argument("--set", action="append", metavar="KEY=VALUE",
                     help="override an experiment option")
    run.set_defaults(func=cmd_run)

    promote = subparsers.add_parser("promote", help="copy article-facing outputs to data/")
    add_targets(promote, "experiment id or prefix")
    promote.set_defaults(func=cmd_promote)

    verify = subparsers.add_parser("verify", help="check data/ against the manifest")
    verify.set_defaults(func=cmd_verify)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="seed data/ from ../archived_code so figures work before a re-run")
    add_targets(bootstrap, "experiment id or prefix (default: everything mapped)")
    bootstrap.set_defaults(func=cmd_bootstrap)

    diff = subparsers.add_parser(
        "diff", help="compare a regenerated run against the archived results")
    add_targets(diff, "experiment id or prefix")
    diff.add_argument("--tolerance", type=float, default=1e-9,
                      help="absolute difference below which a cell counts as unchanged")
    diff.set_defaults(func=cmd_diff)

    doctor = subparsers.add_parser("doctor", help="environment, device and disk")
    doctor.add_argument("--device", default="auto")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    if getattr(args, "all", False):
        args.targets = []
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
