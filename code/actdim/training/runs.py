"""The run registry of appendix O: one named entry per row of the article's table.

Every training run the article draws on is defined here, in full, under the name the
article calls it by. The archived tree could not say that: it held seven configurations and
produced fourteen transformer rows from them by command-line overrides, a shell script from
a different machine, and one legacy notebook that was not part of the trainer at all. Three
of those rows could not be regenerated from anything committed, and none of them recorded
which overrides had produced it. A row of appendix O now names a key in this file.

**Seeds.** ``seed`` moves the train/validation split *and* the initial weights together,
because the task seeds one torch stream and the initialisation continues it. ``init_seed``
restarts the stream between the two, which is the only way to vary one axis at a time.
Which the archived runs used:

* the six sketched runs set ``seed`` alone (42, 43, 44), so their splits and their
  initialisations differ together -- ``mod_wd1_s43`` is not ``mod_wd1`` with new weights;
* the seven extended reruns set ``seed`` and ``init_seed`` to the same value, so the
  separation the field exists for was declared but never exercised;
* ``p211_wd0`` set ``seed`` alone, at 42.

**Two ``mod_wd1`` logs exist, and they are not the same series.** The re-trained log
produced with the trajectory sketch attached agrees with the earlier canonical log to 1e-14
for its first 198 rows, after which float64 rounding amplifies: they end 2.79 apart on the
parameter norm and 0.13 apart on validation accuracy. The consequence is one number.
Generalisation is at step **13,700** in the re-trained log and at **13,810** in the
canonical one. Appendix O quotes 13,700, and everything in section 7.2 -- the collapse
table, the windows, the figure -- is computed from the re-trained log, so 13,700 is the
figure this package reproduces. Memorisation agrees on both, as does every other milestone.
The two logs must not be described as one series or quoted from interchangeably. Only the
re-trained one is regenerable from this registry; the canonical log is an archived artefact
of a notebook that no longer exists in runnable form.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .transformer import BASE_COLUMNS, FULL_COLUMNS, TransformerConfig

# =============================================================================
# Transformer runs -- appendix O, architecture T. Fourteen entries, one per row.
# Nothing perceptron-specific belongs in this section.
# =============================================================================

# --- the six runs the trajectory sketch is attached to (section 7.2) ---------
# Produced by the archived ``active_rank/run_rank.py``, which drove the trainer with the
# probe attached. The three seed variants had no registry entry of their own there: they
# were ``--set seed=43`` on the command line, recorded nowhere but in shell history.

_SKETCHED: Dict[str, TransformerConfig] = {
    "mod_wd1": TransformerConfig(
        key="mod_wd1",
        description="Modular addition p=113, AdamW wd=1.0 -- generalises at 13,700",
        provenance="active_rank/run_rank.py mod_wd1; the log section 7.2 is built on",
        task="modular_addition", p=113, fraction=0.3,
        weight_decay=1.0, max_steps=20_000, log_every=10, seed=42,
        columns=BASE_COLUMNS,
    ),
    "mod_wd1_s43": TransformerConfig(
        key="mod_wd1_s43",
        description="Modular addition p=113, wd=1.0, second seed -- generalises at 6,660",
        provenance="active_rank/run_rank.py mod_wd1 --tag mod_wd1_s43 --set seed=43",
        task="modular_addition", p=113, fraction=0.3,
        weight_decay=1.0, max_steps=20_000, log_every=10, seed=43,
        columns=BASE_COLUMNS,
    ),
    "mod_wd1_s44": TransformerConfig(
        key="mod_wd1_s44",
        description="Modular addition p=113, wd=1.0, third seed -- generalises at 3,680",
        provenance="active_rank/run_rank.py mod_wd1 --tag mod_wd1_s44 --set seed=44",
        task="modular_addition", p=113, fraction=0.3,
        weight_decay=1.0, max_steps=20_000, log_every=10, seed=44,
        columns=BASE_COLUMNS,
    ),
    "mod_wd0": TransformerConfig(
        key="mod_wd0",
        description="Modular addition p=113, wd=0.0 -- the control, censored at 0.03",
        provenance="active_rank/run_rank.py mod_wd0",
        task="modular_addition", p=113, fraction=0.3,
        weight_decay=0.0, max_steps=20_000, log_every=10, seed=42,
        columns=BASE_COLUMNS,
    ),
    # The S_5 pair logs the gradient diagnostics and at a stride of 5, and it is the pair
    # that steps the optimiser twice. The weight decay of 0.2 and the task itself are the
    # two other departures from the source configuration appendix O records.
    "s5_wd1": TransformerConfig(
        key="s5_wd1",
        description="S_5 composition, AdamW wd=0.2 -- generalises at 6,735",
        provenance="active_rank/run_rank.py s5_wd1",
        task="symmetric_group", n=5, fraction=0.5,
        weight_decay=0.2, max_steps=15_000, log_every=5, seed=42,
        columns=FULL_COLUMNS, double_step=True,
    ),
    "s5_wd0": TransformerConfig(
        key="s5_wd0",
        description="S_5 composition, wd=0.0 -- the control, censored at 0.02",
        provenance="active_rank/run_rank.py s5_wd0",
        task="symmetric_group", n=5, fraction=0.5,
        weight_decay=0.0, max_steps=15_000, log_every=5, seed=42,
        columns=FULL_COLUMNS, double_step=True,
    ),
}

# --- the seven extended reruns, 120,000 steps (sections 7.1 and 7.3) --------
# These came from ``dimension_recovery/launch_extended.sh``, which is what produced the
# committed logs. Its sibling ``extend_runs.py`` plans the same campaign at 200,000 steps
# over twelve runs and calls the positive control ``grok_positive``, none of which matches
# the seven committed files: the budget is 120,000 and the control's log is
# ``grokpos_s0``. The shell script is the authority, and these entries follow it. Both set
# ``seed`` and ``init_seed`` to the same value, so the split and the initialisation still
# move together.

_EXTENDED: Dict[str, TransformerConfig] = {
    "grokpos_s0": TransformerConfig(
        key="grokpos_s0",
        description="Positive control at the long budget -- must still generalise",
        provenance="launch_extended.sh: mod_wd1 at 120k, seed 0",
        task="modular_addition", p=113, fraction=0.3,
        weight_decay=1.0, max_steps=120_000, log_every=10, seed=0, init_seed=0,
        columns=BASE_COLUMNS,
    ),
    "lowdata15_s0": TransformerConfig(
        key="lowdata15_s0",
        description="Training fraction 0.15 -- generalises at 110,940, well past the old budget",
        provenance="launch_extended.sh: mod_wd1 at 120k, fraction 0.15, seed 0",
        task="modular_addition", p=113, fraction=0.15,
        weight_decay=1.0, max_steps=120_000, log_every=10, seed=0, init_seed=0,
        columns=BASE_COLUMNS,
    ),
    "lowdata15_s1": TransformerConfig(
        key="lowdata15_s1",
        description="Training fraction 0.15, second seed -- censored at 0.94 and rising",
        provenance="launch_extended.sh: mod_wd1 at 120k, fraction 0.15, seed 1",
        task="modular_addition", p=113, fraction=0.15,
        weight_decay=1.0, max_steps=120_000, log_every=10, seed=1, init_seed=1,
        columns=BASE_COLUMNS,
    ),
    "lowdata15_s2": TransformerConfig(
        key="lowdata15_s2",
        description="Training fraction 0.15, third seed -- censored at 0.01",
        provenance="launch_extended.sh: mod_wd1 at 120k, fraction 0.15, seed 2",
        task="modular_addition", p=113, fraction=0.15,
        weight_decay=1.0, max_steps=120_000, log_every=10, seed=2, init_seed=2,
        columns=BASE_COLUMNS,
    ),
    "lowdata20_s0": TransformerConfig(
        key="lowdata20_s0",
        description="Training fraction 0.20 -- generalises at 39,600",
        provenance="launch_extended.sh: mod_wd1 at 120k, fraction 0.20, seed 0",
        task="modular_addition", p=113, fraction=0.20,
        weight_decay=1.0, max_steps=120_000, log_every=10, seed=0, init_seed=0,
        columns=BASE_COLUMNS,
    ),
    "wd0_s0": TransformerConfig(
        key="wd0_s0",
        description="No weight decay at the long budget -- censored at 0.09",
        provenance="launch_extended.sh: mod_wd0 at 120k, seed 0",
        task="modular_addition", p=113, fraction=0.3,
        weight_decay=0.0, max_steps=120_000, log_every=10, seed=0, init_seed=0,
        columns=BASE_COLUMNS,
    ),
    "wd0_s1": TransformerConfig(
        key="wd0_s1",
        description="No weight decay at the long budget, second seed -- censored at 0.07",
        provenance="launch_extended.sh: mod_wd0 at 120k, seed 1",
        task="modular_addition", p=113, fraction=0.3,
        weight_decay=0.0, max_steps=120_000, log_every=10, seed=1, init_seed=1,
        columns=BASE_COLUMNS,
    ),
}

# --- the legacy larger modulus ----------------------------------------------
# ``p211_wd0`` came from a Colab notebook outside the trainer entirely
# (``Grokking/modular_addition_grokking_colab/colab_p211_omnigrok_wd0_ablation.ipynb`` over
# ``prime_sweep_omnigrok.py``), which is why it is the one row nothing in the archived
# trainer could regenerate. The entry below is that notebook's configuration expressed in
# this package's terms, and three of its settings do not survive the translation:
#
#   * the split was balanced to 34 training examples per output class, not drawn uniformly.
#     34/211 is the training fraction that produced, which is the 0.16 appendix O reports,
#     but a uniform split of the same size is a different training set;
#   * the notebook trained in float32 with batches of 512, both kept below;
#   * it monitored accuracy on 2,048 validation examples where appendix O states 512. The
#     512 of the stated protocol is kept, since that is the definition every other row uses.
#
# Re-running this entry therefore reproduces the row's configuration, not its log. Nothing
# in the article reads the log itself; the row is quoted for its budget and its outcome.

_LEGACY: Dict[str, TransformerConfig] = {
    "p211_wd0": TransformerConfig(
        key="p211_wd0",
        description="Modular addition p=211, wd=0.0 at 200k steps -- censored at 0.04",
        provenance="legacy notebook colab_p211_omnigrok_wd0_ablation.ipynb, not the trainer",
        task="modular_addition", p=211,
        fraction=34 / 211,          # 0.1611: 34 training examples per output class
        weight_decay=0.0, max_steps=200_000, log_every=50, seed=42,
        batch_size=512, dtype="float32",
        columns=BASE_COLUMNS,
    ),
}

TRANSFORMER_RUNS: Dict[str, TransformerConfig] = {}
TRANSFORMER_RUNS.update(_SKETCHED)
TRANSFORMER_RUNS.update(_EXTENDED)
TRANSFORMER_RUNS.update(_LEGACY)

SKETCHED_RUNS: Tuple[str, ...] = tuple(_SKETCHED)
"""The six runs the trajectory sketch is attached to, grokking runs before their controls."""

EXTENDED_RUNS: Tuple[str, ...] = tuple(_EXTENDED)
"""The seven 120,000-step reruns."""

# =============================================================================
# Perceptron runs -- appendix O, architecture P. Added by the perceptron port;
# put nothing transformer-specific below this line.
# =============================================================================

PERCEPTRON_RUNS: Dict[str, Any] = {}


# =============================================================================
# The registry both architectures share.
# =============================================================================

RUNS: Dict[str, Any] = {}
RUNS.update(TRANSFORMER_RUNS)
RUNS.update(PERCEPTRON_RUNS)


def get(key: str) -> Any:
    """The configuration of one run, by the name appendix O calls it."""
    if key not in RUNS:
        raise KeyError(f"unknown run '{key}'. Available: {', '.join(sorted(RUNS))}")
    return RUNS[key]


def keys() -> List[str]:
    """Every registered run, in registration order."""
    return list(RUNS)


def transformer_keys() -> List[str]:
    """The fourteen transformer rows of appendix O: the six sketched, the seven extended,
    and the one legacy run."""
    return list(TRANSFORMER_RUNS)
