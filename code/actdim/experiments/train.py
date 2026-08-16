"""The training campaigns of appendix O.

The only experiments that need a GPU. Both trainers run in float64, which a CPU will do
but slowly enough that a silent fallback wastes hours before anyone notices, so these
refuse a CPU unless ``--allow-cpu`` says otherwise.

Each writes its logs into ``runs/<id>/`` and promotes only what a figure or a table reads.
A sketched run also writes a ``.npz`` of the trajectory sketch, tens of megabytes each,
which stays in the untracked half.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..runtime import CPU, GPU, Context, experiment


def _train_transformer(ctx: Context, keys: Sequence[str], sketch: bool = False) -> List[dict]:
    from ..training import transformer as trainer

    rows = []
    for key in keys:
        print(f"  {key}")
        result = trainer.run(ctx, key, outdir=ctx.store.dir, sketch=sketch, progress=False)
        rows.append({"run": key, "t_mem": result.t_mem, "t_gen": result.t_gen,
                     "final_val_acc": getattr(result, "final_val_acc", None),
                     "n_params": getattr(result, "n_params", None),
                     "rows": getattr(result, "n_rows", None)})
    return rows


def _summary(ctx: Context, rows: List[dict], name: str = "milestones.csv") -> None:
    import pandas as pd

    ctx.store.table(name, pd.DataFrame(rows))


#: Every transformer run's training curve, promoted beside the milestones it is summarised
#: into. Only ``mod_wd1_train.csv`` used to reach ``data/``, and it got there through
#: ``grok.rank.dip`` rather than through the campaign that trained it, so the accuracy and
#: parameter-norm curves of the other eleven runs existed only in the untracked half. A
#: figure that puts the estimate under the loss curve it was computed from needs the curve,
#: and re-running a GPU campaign to recover a 900 KB CSV is not a repair anyone performs.
def _curve_promotes(keys: Sequence[str]) -> tuple:
    return ("milestones.csv",) + tuple(f"{key}_train.csv" for key in keys)


def _sketched_promotes() -> tuple:
    from ..training.runs import SKETCHED_RUNS

    return _curve_promotes(SKETCHED_RUNS)


def _extended_promotes() -> tuple:
    from ..training.runs import EXTENDED_RUNS

    return _curve_promotes(EXTENDED_RUNS)


_SKETCHED_PROMOTES = _sketched_promotes()
_EXTENDED_PROMOTES = _extended_promotes()


@experiment(
    id="train.transformer.sketched",
    title="Six transformer runs with the trajectory sketch attached",
    paper=("app:runs", "sec:direct", "app:dip"),
    device=GPU,
    minutes=24,
    promotes=_SKETCHED_PROMOTES,
    tier=2,
    notes="The runs section 7.2 measures. Each writes a 60-95 MB sketch into runs/. The "
          "training curves are promoted too: figure 6 draws the estimate under the "
          "validation accuracy of the run it was computed from, and only one of the six "
          "curves used to reach data/.",
)
def transformer_sketched(ctx: Context) -> None:
    from ..training import runs as registry

    keys = list(registry.SKETCHED_RUNS)
    if ctx.fast:
        keys = keys[:1]
        ctx.note("fast", "one run, truncated budget")
    _summary(ctx, _train_transformer(ctx, keys, sketch=True))


@experiment(
    id="train.transformer.extended",
    title="Seven 120,000-step reruns of modular addition",
    paper=("app:runs", "sec:grok-diagnostics", "sec:pairs"),
    device=GPU,
    minutes=150,
    promotes=_EXTENDED_PROMOTES,
    tier=2,
    notes="The logs the two diagnostics are computed on. Produced in the archived tree by "
          "a shell script, not by the planner that claims to plan them. The training "
          "curves are promoted beside the milestones, so the windowed estimate of "
          "figure 8 can be drawn against the run it came from.",
)
def transformer_extended(ctx: Context) -> None:
    from ..training import runs as registry

    keys = list(registry.EXTENDED_RUNS)
    if ctx.fast:
        keys = keys[:1]
    _summary(ctx, _train_transformer(ctx, keys))


@experiment(
    id="train.transformer.p211",
    title="Modular addition at p = 211 without weight decay",
    paper=("app:runs",),
    device=GPU,
    minutes=180,
    promotes=("milestones.csv",),
    tier=2,
    notes="Re-specified rather than reproduced: the archived log came from a notebook "
          "with a class-balanced split and float32. See the registry entry.",
)
def transformer_p211(ctx: Context) -> None:
    keys = ["p211_wd0"]
    _summary(ctx, _train_transformer(ctx, keys))


def _perceptron_group(ctx: Context, group: str, limit: int = 0) -> List[str]:
    from ..training import runs_perceptron as registry

    keys = list(registry.GROUPS[group])
    if ctx.fast and limit:
        keys = keys[:limit]
        ctx.note("fast", f"{len(keys)} of {len(registry.GROUPS[group])} runs")
    return keys


def _train_perceptron(ctx: Context, keys: Sequence[str], sketch: bool = False,
                      overrides: Dict[str, Any] = None) -> List[dict]:
    from ..training import perceptron as trainer

    rows = []
    for key in keys:
        print(f"  {key}")
        record = trainer.train_registered(ctx, key, overrides=dict(overrides or {}),
                                          sketch=sketch)
        rows.append({"run": key, **{k: record.get(k) for k in
                                    ("t_memorise", "t_grok", "final_val_acc",
                                     "final_train_acc", "diverged_at", "n_rows")}})
    return rows


@experiment(
    id="train.perceptron.arith",
    title="The quadratic perceptron on the arithmetic tasks",
    paper=("app:runs", "sec:pairs"),
    device=GPU,
    minutes=36,
    promotes=("milestones.csv",),
    tier=2,
)
def perceptron_arith(ctx: Context) -> None:
    _summary(ctx, _train_perceptron(ctx, _perceptron_group(ctx, "arith", limit=1)))


@experiment(
    id="train.perceptron.poly",
    title="The quadratic perceptron on the polynomial pairs",
    paper=("app:runs", "sec:pairs", "app:repr"),
    device=GPU,
    minutes=72,
    promotes=("milestones.csv", "g_p2_train.csv", "g_p2x_train.csv"),
    tier=2,
    notes="The two promoted logs are what fig_pairs draws; the rest stay in runs/.",
)
def perceptron_poly(ctx: Context) -> None:
    _summary(ctx, _train_perceptron(ctx, _perceptron_group(ctx, "poly", limit=2)))


# The sketched perceptron campaigns are three experiments rather than one because the
# trainer names its files after the run key: the same four runs at two batch sizes would
# overwrite each other in a single directory. Splitting them also keeps the comparison
# appendix J makes -- full batch against mini-batch -- readable from the run tree itself,
# which is what the archived metadata could not do: every configuration field in it was
# null, including the batch size, the one thing that differed.

@experiment(
    id="train.perceptron.sketched.fb",
    title="Four sketched perceptron runs at full batch",
    paper=("app:fb", "app:sketch"),
    device=GPU,
    minutes=12,
    promotes=("milestones.csv",),
    tier=2,
    notes="a_add, x_no_grok, g_p1, g_p1x. Not the article's 'top four rows of the "
          "inventory', which names a different set; see docs/errata.md item 11.",
)
def perceptron_sketched_fb(ctx: Context) -> None:
    rows = _train_perceptron(ctx, _perceptron_group(ctx, "sketched", limit=1), sketch=True)
    for row in rows:
        row["batch"] = "full"
    _summary(ctx, rows)


@experiment(
    id="train.perceptron.sketched.mb",
    title="The same four runs at mini-batches of 512",
    paper=("app:fb",),
    device=GPU,
    minutes=12,
    promotes=("milestones.csv",),
    tier=2,
    notes="The comparison arm of appendix J. Its milestones agree with the full-batch "
          "arm to within six logged steps, not the one or two the article states.",
)
def perceptron_sketched_mb(ctx: Context) -> None:
    rows = _train_perceptron(ctx, _perceptron_group(ctx, "sketched", limit=1),
                             sketch=True, overrides={"batch_size": 512})
    for row in rows:
        row["batch"] = "512"
    _summary(ctx, rows)


@experiment(
    id="train.perceptron.sketched.long",
    title="Two long sketched runs at a coarser logging stride",
    paper=("app:fb",),
    device=GPU,
    minutes=12,
    promotes=("milestones.csv",),
    tier=2,
    notes="150,000 steps logged every 50. This is the input to the window-length sweep "
          "of appendix J, whose archived sketches were not kept, so the committed "
          "pr_vs_window.csv cannot be regenerated without this run.",
)
def perceptron_sketched_long(ctx: Context) -> None:
    keys = ["a_add", "x_no_grok"][: 1 if ctx.fast else 2]
    rows = _train_perceptron(ctx, keys, sketch=True,
                             overrides={"max_steps": 150_000, "log_every": 50})
    _summary(ctx, rows)


#: What appendix Q and figure 12 read. The summary alone is not enough: panel (a) draws a
#: sharpness trace per rate and panel (b) one step-by-step loss log, and while those stayed
#: unpromoted the figure was drawn from the archived campaign while the table beside it came
#: from the current one -- which disagreed about which runs diverged.
def _eos_promotes() -> tuple:
    from ..training.eos import DEFAULT_LRS

    return (("eos_runs.csv",)
            + tuple(f"eos_lr{lr:g}_s1_sharp.csv" for lr in DEFAULT_LRS)
            + ("eos_lr2e+06_s1_train.csv",))


_EOS_PROMOTES = _eos_promotes()


@experiment(
    id="train.perceptron.eos",
    title="Full-batch descent at eight learning rates, logged every step",
    paper=("app:eos",),
    device=GPU,
    minutes=30,
    promotes=_EOS_PROMOTES,
    tier=2,
    notes="Appendix Q. Sharpness by power iteration every hundred steps, seeded from a "
          "stream that cannot move the trajectory. The seed-1 sharpness traces and the "
          "one step-by-step loss log figure 12 draws are promoted beside the summary, so "
          "that the figure and the table cannot come from different campaigns.",
)
def perceptron_eos(ctx: Context) -> None:
    from ..training import eos

    # The power-iteration start vector is seeded from a stream of its own, so that
    # turning the sharpness measurement on cannot move the trajectory it measures.
    ctx.declare_seeds("sharpness_start")

    if ctx.fast:
        # The smallest campaign that still crosses the edge: one rate below it, one
        # above, one seed, and enough steps for the sharpness to be measured twice.
        records = eos.campaign(ctx, lrs=(1e5, 3e6), seeds=(1,), steps=200, sharp_every=100)
    else:
        records = eos.campaign(ctx)

    ctx.store.table("eos_runs.csv", eos.campaign_table(records))
    ctx.note("diverged", sorted(r["key"] for r in records if r.get("diverged_at")))
