"""The two checks that stand behind the trajectory sketch.

Both are measurements the article quotes and neither produces a result of its own. They are
here rather than beside the training campaigns because what they measure is the instrument,
not the runs.

``check.sketch.noninvasive``  appendix I's third check: the log is bit-identical with the
                             observer attached and without it. The task module seeds one
                             global torch stream that the train/validation split, the
                             initial weights and the mini-batch order all continue, so a
                             single stray draw inside the observer changes the initial
                             weights, and the initial weights decide whether these runs
                             generalise at all. ``tests/test_sketch.py`` proves the property
                             at two hundred steps and a probe of sixteen; this runs a
                             registry configuration at the production sketch width, which
                             is the width the published sketches were taken at and the only
                             width whose cost and probe size are the ones in use.

``check.sketch.accuracy``    appendix I's first check: the compression is scored against
                             the same statistic computed without it, on trajectories whose
                             rank is known. The shortfall against the nominal rank at five
                             and ten directions is present in the uncompressed column too,
                             so the table separates the price of the sketch from the price
                             of a sixty-sample participation ratio.

``check.sketch.cost``        appendix S's number: what the observer costs in time and in
                             storage. Three things about it are stated in the record rather
                             than left to the reader. The device is the resolved one, since
                             the archived measurement wrote ``"auto"`` and so cannot say
                             whether it ran on a GPU (errata item 30). The overhead is
                             reported as a bound where it comes out negative, which it does
                             whenever the sketch costs less than the run-to-run variation of
                             the machine. And the architecture is recorded, because the
                             measurement trains the perceptron while the appendix describes
                             the number at the transformer's logging stride, and the two
                             have different parameter counts.

Both want a GPU. Both will run on a CPU under ``--allow-cpu``, which is how their plumbing
is checked on a machine without one; the timing measurement is then a measurement of that
CPU and must not be quoted for the article.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from ..runtime import CPU, GPU, Context, experiment

#: The width the published sketches were taken at: 1024 coordinates under two independent
#: hash families, on a probe of 256 training examples. Appendix I quotes the disagreement
#: between the two families as the error of the compression, so a check at one family would
#: not exercise what the article relies on.
SKETCH_DIM = 1024
SKETCH_FAMILIES = 2
SKETCH_PROBE = 256

#: The configuration appendix S measures on: 1500 full-batch steps of the perceptron at
#: p = 97, logged every ten, which is the 151 logged steps and 145,500 parameters the
#: appendix quotes.
COST_STEPS = 1_500
COST_LOG_EVERY = 10

#: Repeats of the alternating bare/probed pair. Even, so that each arm runs first exactly
#: half the time and neither can be favoured by a warm cache or a thermal ramp. Four rather
#: than the archived two, because the archived measurement came out negative -- the probed
#: arm faster than the bare one -- and a bound that tight needs more than two samples of the
#: machine's own variation before it is quoted.
COST_REPEATS = 4


def _equal(left: Any, right: Any) -> bool:
    """Bit-identical, counting two NaNs in the same place as identical.

    A skipped validation pass writes NaN, and ``NaN != NaN`` would report a difference where
    both arms did the same thing. Every other difference, down to the last bit, is a
    difference.
    """
    import numpy as np

    if left.shape != right.shape:
        return False
    if left.dtype.kind not in "fc" or right.dtype.kind not in "fc":
        return bool(np.array_equal(left, right))
    both_nan = np.isnan(left) & np.isnan(right)
    return bool(np.all((left == right) | both_nan))


def _largest_difference(left: Any, right: Any) -> float:
    import numpy as np

    if left.dtype.kind not in "fc" or right.dtype.kind not in "fc":
        return float(np.sum(left != right))
    difference = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    finite = difference[np.isfinite(difference)]
    return float(finite.max()) if finite.size else 0.0


def _first_difference(left: Any, right: Any) -> int:
    """The first row on which the two arms parted, or -1 if they never did.

    Which row it is says what went wrong. Row zero means the observer moved the
    initialisation, which is the failure this check exists for. A row deep into the run
    means the two arms started together and drifted, which is a different diagnosis.
    """
    import numpy as np

    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        same = (left == right) | (np.isnan(left) & np.isnan(right))
    else:
        same = left == right
    moved = np.flatnonzero(~same)
    return int(moved[0]) if moved.size else -1


def _compare_logs(left: Any, right: Any) -> Any:
    """One row per logged column: whether it is identical, and where it is not."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    columns = list(left.columns)
    if columns != list(right.columns):
        # Reported as a row rather than raised, so the file says what happened. A column
        # present in one arm and not the other is a difference in what was logged, which
        # is the same defect as a difference in what was logged for.
        rows.append({"column": "<columns>", "rows": len(left), "identical": False,
                     "largest_difference": float("nan"), "first_difference_row": -1,
                     "note": f"left {columns} against right {list(right.columns)}"})
        columns = [c for c in columns if c in right.columns]

    # A run that diverged stops early, so the two arms can differ in length. The common
    # prefix is still comparable and says where they parted; comparing the ragged tail
    # would raise instead of reporting.
    shared = min(len(left), len(right))
    if len(left) != len(right):
        rows.append({"column": "<rows>", "rows": shared, "identical": False,
                     "largest_difference": float(abs(len(left) - len(right))),
                     "first_difference_row": shared,
                     "note": f"left logged {len(left)} rows and right {len(right)}"})

    for column in columns:
        a = left[column].to_numpy()[:shared]
        b = right[column].to_numpy()[:shared]
        rows.append({"column": column, "rows": int(shared), "identical": _equal(a, b),
                     "largest_difference": _largest_difference(a, b),
                     "first_difference_row": _first_difference(a, b), "note": ""})
    return pd.DataFrame(rows)


@experiment(
    id="check.sketch.noninvasive",
    title="The trajectory sketch leaves every logged value bit-identical",
    paper=("app:sketch",),
    device=GPU,
    minutes=12,
    promotes=("sketch_noninvasive.csv",),
    tier=4,
    notes="Trains one registry run twice, with the observer and without, at the production "
          "sketch width. Set run= to check another key and steps= to shorten the budget.",
)
def sketch_noninvasive(ctx: Context) -> None:
    import pandas as pd

    from ..sketch.probe import TrajectoryRecorder
    from ..training import runs as registry
    from ..training import transformer as trainer

    key = str(ctx.option("run", "mod_wd1"))
    config = registry.get(key).replace(device=ctx.device)
    probe = SKETCH_PROBE

    if ctx.fast:
        # The smallest run that still exercises every branch this check is about: a real
        # task, a real optimiser, several logged steps and the same probe machinery. The
        # sketch width stays at the production value, because the width is the thing under
        # test and a narrower one would check something the article does not use.
        config = config.replace(p=17, max_steps=200, log_every=10)
        probe = 32
        ctx.note("fast", "p = 17, 200 steps, probe of 32; sketch width unchanged")
    steps = ctx.option("steps", None)
    if steps is not None:
        config = config.replace(max_steps=int(steps))

    ctx.config(run=key, sketch_dim=SKETCH_DIM, n_sketch=SKETCH_FAMILIES, n_probe=probe,
               device=ctx.device, **{f"run.{k}": v for k, v in config.to_dict().items()})

    print(f"  {key}: {config.max_steps} steps, bare")
    plain = trainer.train(config)
    print(f"  {key}: {config.max_steps} steps, with the sketch attached")
    recorder = TrajectoryRecorder(dim=SKETCH_DIM, n_sketch=SKETCH_FAMILIES, n_probe=probe,
                                  seed=ctx.seed_for(f"sketch:{key}"))
    probed = trainer.train(config, observer=recorder)

    frame = _compare_logs(plain.log, probed.log)
    frame.insert(0, "comparison", "bare against probed")
    moved = frame[~frame.identical]

    # A machine that cannot reproduce its own run cannot be asked whether the observer
    # moved one. Where a difference appears, the bare arm is trained a second time and the
    # two bare runs compared, so an overnight failure says which of the two it was instead
    # of leaving a night's work ambiguous. This costs a third run only when it is needed.
    control = None
    if len(moved):
        print("  a logged column moved; training the bare arm again to tell the observer "
              "from the machine")
        again = trainer.train(config)
        control = _compare_logs(plain.log, again.log)
        control.insert(0, "comparison", "bare against bare")
        frame = pd.concat([frame, control], ignore_index=True)
    reproducible = control is None or bool(control.identical.all())

    ctx.store.table("sketch_noninvasive.csv", frame)
    ctx.store.json("sketch_noninvasive.json", {
        "run": key,
        "steps": int(config.max_steps),
        "logged_rows": int(len(plain.log)),
        "sketch": recorder.metadata(),
        "device": ctx.device,
        "milestones_bare": {"t_mem": plain.t_mem, "t_gen": plain.t_gen},
        "milestones_probed": {"t_mem": probed.t_mem, "t_gen": probed.t_gen},
        "identical": bool(not len(moved)),
        "machine_reproduces_itself": reproducible,
    })

    ctx.note("columns_checked", int(len(frame[frame.comparison == "bare against probed"])))
    ctx.note("columns_moved", sorted(str(name) for name in moved.column))
    ctx.note("machine_reproduces_itself", reproducible)
    print(f"\n{len(plain.log)} logged row(s), sketch width "
          f"{SKETCH_DIM} x {SKETCH_FAMILIES}, probe {probe}")
    for record in frame.to_dict("records"):
        mark = "same" if record["identical"] else "MOVED"
        print(f"  [{mark:<5}] {record['comparison']:<20} {record['column']:<16} "
              f"largest difference {record['largest_difference']:.3e}")

    if len(moved):
        # The outputs are written first on purpose: this is the evidence, and a failure
        # that discards its own evidence has to be reproduced before it can be read.
        if not reproducible:
            raise AssertionError(
                f"two bare runs of {key} on {ctx.device} disagree with each other, so this "
                f"machine does not reproduce its own runs and the question this check asks "
                f"cannot be answered on it. Nothing has been shown about the sketch. See "
                f"sketch_noninvasive.csv for the columns and the rows they part at.")
        raise AssertionError(
            f"the sketch moved {len(moved)} logged column(s) of {key}: "
            f"{', '.join(str(name) for name in moved.column)}, first at row "
            f"{int(moved.first_difference_row.min())}. Two bare runs agree, so this is the "
            f"observer. It must not draw from the global torch stream: the split, the "
            f"initial weights and the mini-batch order all continue it, and one stray draw "
            f"changes the run this check claims to be watching.")

    print("\nevery logged value is bit-identical with the sketch attached.")


@experiment(
    id="check.sketch.cost",
    title="What the trajectory sketch costs, in seconds and in megabytes",
    paper=("app:compute",),
    device=GPU,
    minutes=6,
    promotes=("sketch_cost.json",),
    tier=4,
    notes="Appendix S. The archived measurement recorded device 'auto' and so cannot say "
          "whether it ran on a GPU; the resolved device is written here. A negative "
          "overhead is a bound, not a result.",
)
def sketch_cost(ctx: Context) -> None:
    from ..training import runs_perceptron as registry
    from ..training.perceptron import sketch_cost as measure

    steps = int(ctx.option("steps", COST_STEPS))
    repeats = int(ctx.option("repeats", COST_REPEATS))
    log_every = COST_LOG_EVERY

    if ctx.fast:
        # Two arms, one alternation, a hundred steps: enough to write the same file with
        # the same fields and no fewer branches taken.
        steps, repeats = 100, 2
        ctx.note("fast", "100 steps and one alternation; the timings mean nothing")

    # No snapshots and no spectral probes: they are logged work that neither arm is being
    # timed on, and including them would dilute the ratio the appendix quotes.
    config = registry.get("a_add").with_overrides({
        "max_steps": steps, "log_every": log_every, "obs_every": 0, "n_snapshots": 0,
        "device": ctx.device})

    ctx.config(run="a_add", steps=steps, log_every=log_every, repeats=repeats,
               sketch_dim=SKETCH_DIM, n_probe=SKETCH_PROBE, device=ctx.device,
               **{f"run.{k}": v for k, v in config.resolved().items()})

    print(f"  a_add: {repeats} alternating pair(s) of {steps} steps on {ctx.device}")
    record = measure(config, repeats=repeats, dim=SKETCH_DIM, n_probe=SKETCH_PROBE,
                     device=ctx.device)

    # The device the measurement resolved for itself must be the one the context resolved,
    # or the record names a machine the timings did not come from.
    if record["device"] != ctx.device:
        raise AssertionError(
            f"the measurement ran on {record['device']!r} and the context resolved "
            f"{ctx.device!r}. The record has to name the device the timings came from; "
            f"the archived one wrote 'auto' and cannot.")

    overhead = float(record["overhead_frac"])
    record["overhead_is_bound"] = overhead <= 0.0
    record["fast"] = bool(ctx.fast)
    ctx.store.json("sketch_cost.json", record)

    ctx.note("overhead_frac", overhead)
    ctx.note("resolved_device", record["device"])
    print(f"\n  bare    {record['seconds_bare']:.2f} s")
    print(f"  probed  {record['seconds_probed']:.2f} s")
    print(f"  overhead {overhead:+.1%}"
          + ("  -- negative, so a bound and not a result" if overhead <= 0 else ""))
    print(f"  storage {record['sketched_float32_MB']:.1f} MB sketched against "
          f"{record['full_float32_MB']:.1f} MB full, a factor of "
          f"{record['storage_ratio']:.0f}")
    print(f"  device  {record['device']}   model {record['model']}, "
          f"{record['n_params']} parameters")
    if record["model"] != "transformer":
        print("  note: appendix S describes this ratio at the transformer's logging "
              "stride; the two architectures have different parameter counts.")


#: The ranks appendix I tabulates, the window it uses, and the parameter count it runs at.
#: Sixty samples is the window the direct measurement of section 7.1 uses on a run logged
#: every ten steps, and 145,500 is the perceptron's parameter count, so the check is at the
#: geometry the article's numbers were taken at rather than at a convenient one.
SKETCH_RANKS: Tuple[int, ...] = (1, 2, 5, 10)
SKETCH_SAMPLES = 60
SKETCH_PARAMETERS = 145_500


@experiment(
    id="check.sketch.accuracy",
    title="What the trajectory sketch costs in accuracy, against the uncompressed statistic",
    paper=("app:sketch", "tab:sketch"),
    device=CPU,
    minutes=2,
    promotes=("sketch_accuracy.csv",),
    tier=4,
    notes="Appendix I's first check. Synthetic trajectories of known rank in the "
          "perceptron's parameter count, scored with and without the compression, so that "
          "the price of the sketch is separated from the price of a sixty-sample "
          "participation ratio.",
)
def sketch_accuracy(ctx: Context) -> None:
    import numpy as np
    import pandas as pd
    import torch

    from ..sketch.analysis import pr
    from ..sketch.countsketch import CountSketch

    ranks: Sequence[int] = SKETCH_RANKS
    samples, dimension = SKETCH_SAMPLES, SKETCH_PARAMETERS
    if ctx.fast:
        ranks, dimension = ranks[:2], 5_000

    ctx.config(ranks=list(ranks), samples=samples, parameters=dimension,
               sketch_dim=1024, n_sketch=2)
    ctx.declare_seeds("sketch_trajectory", "sketch_hashes")

    rows: List[Dict[str, Any]] = []
    for rank in ranks:
        # A trajectory of exactly `rank` directions: an orthonormal basis of the ambient
        # space carried by independent coefficients. Its participation ratio is the rank
        # only in expectation, which is the point of measuring the uncompressed value too.
        generator = ctx.rng(f"sketch_trajectory:{rank}")
        basis = np.linalg.qr(generator.standard_normal((dimension, rank)))[0]
        coefficients = generator.standard_normal((samples, rank))
        trajectory = coefficients @ basis.T

        sketch = CountSketch(dimension, dim=1024, n_sketch=2,
                             seed=int(ctx.seed_for("sketch_hashes")))
        compressed = np.stack([
            sketch.apply(torch.from_numpy(row.astype(np.float32))).reshape(-1).numpy()
            for row in trajectory])

        plain, sketched = pr(trajectory), pr(compressed)
        rows.append({"true_rank": rank, "samples": samples, "parameters": dimension,
                     "pr_uncompressed": plain, "pr_sketched": sketched,
                     "difference": sketched - plain})
        print(f"    rank {rank:>2}: {plain:.4f} uncompressed, {sketched:.4f} sketched",
              flush=True)

    frame = pd.DataFrame(rows)
    ctx.store.table("sketch_accuracy.csv", frame)
    ctx.note("worst_difference", float(frame.difference.abs().max()))
