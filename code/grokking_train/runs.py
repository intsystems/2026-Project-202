"""Registry of the training runs behind the figures of ``icomp_article/grokking_en.tex``.

Each entry pins the task, architecture, optimizer and logging cadence of one CSV
in ``../grokking_analysis/grokking_logs/``, so ``train.py <key>`` regenerates that
log and ``reproduce_figures.py`` on the other side turns it back into figures.

The keys line up with ``grokking_analysis/experiments.py``: the figure ``s5_wd1``
is built from the log produced by the run ``s5_wd1``.
"""

from pathlib import Path

from grok.config import BASE_COLUMNS, BASELINE_COLUMNS, FULL_COLUMNS, RunConfig

LOG_DIR = Path(__file__).resolve().parent / "grokking_logs"
ANALYSIS_LOG_DIR = Path(__file__).resolve().parent.parent / "grokking_analysis" / "grokking_logs"


RUNS = {
    # --- Sec. 4.1, Fig. 1: modular addition (p = 113), Omnigrok 1L transformer,
    #     AdamW + mini-batches.  Observable: the weight norm ||w||_2. ---
    "mod_wd1": RunConfig(
        key="mod_wd1",
        description="Modular addition p=113, AdamW WD=1.0 -- grokks at step ~13810",
        task="modular_addition",
        p=113,
        fraction=0.3,
        weight_decay=1.0,
        max_steps=20000,
        log_every=10,
        columns=BASE_COLUMNS,
        csv="grokking_modular_addition_logs_to_flat_grokking_with_stochastic.csv",
    ),
    "mod_wd0": RunConfig(
        key="mod_wd0",
        description="Modular addition p=113, AdamW WD=0.0 -- control, never grokks",
        task="modular_addition",
        p=113,
        fraction=0.3,
        weight_decay=0.0,
        max_steps=20000,
        log_every=10,
        columns=BASE_COLUMNS,
        csv="grokking_modular_addition_logs_to_flat_grokking_with_stochastic_"
            "without_wight_decay.csv",
    ),

    # --- Sec. 4.2, Figs. 2-3: composition in the non-abelian group S_5.
    #     Logs the gradient diagnostics on top of the base columns. ---
    "s5_wd1": RunConfig(
        key="s5_wd1",
        description="S_5 composition, AdamW WD=0.2 -- grokks at step ~6735",
        task="symmetric_group",
        n=5,
        fraction=0.5,
        weight_decay=0.2,
        max_steps=15000,
        log_every=5,
        columns=FULL_COLUMNS,
        csv="grokking_modular_addition_logs_S_5_with_stochastic.csv",
        double_step=True,
    ),
    "s5_wd0": RunConfig(
        key="s5_wd0",
        description="S_5 composition, AdamW WD=0.0 -- control, eternal memorization",
        task="symmetric_group",
        n=5,
        fraction=0.5,
        weight_decay=0.0,
        max_steps=15000,
        log_every=5,
        columns=FULL_COLUMNS,
        csv="grokking_modular_addition_logs_S_5_with_stochastic_without_wight_decay.csv",
        double_step=True,
    ),

    # --- Appendix B: the full-batch baseline that motivated the switch to
    #     mini-batches.  Different architecture, no weight-norm column, and the
    #     published log was produced without a seed. ---
    "full_batch": RunConfig(
        key="full_batch",
        description="Full-batch AdamW baseline on p=97, stock encoder (App. B)",
        task="modular_addition",
        p=97,
        fraction=0.5,
        model="encoder",
        weight_decay=1.0,
        max_steps=15000,
        batch_size=None,
        val_batch_size=None,
        log_every=1,
        columns=BASELINE_COLUMNS,
        csv="grokking_modular_addition_logs.csv",
        dtype="float32",
        seed=None,
    ),

    # --- Templates, not article figures ------------------------------------
    "sn": RunConfig(
        key="sn",
        description="S_n composition template -- `--set n=6` and go",
        task="symmetric_group",
        n=5,
        fraction=0.5,
        weight_decay=1e-3,
        max_steps=15000,
        log_every=10,
        columns=FULL_COLUMNS,
        csv="grokking_s{n}_logs.csv",
    ),
    "sn_wd0": RunConfig(
        key="sn_wd0",
        description="S_n composition template, WD=0.0 -- the no-grokking control",
        task="symmetric_group",
        n=5,
        fraction=0.5,
        weight_decay=0.0,
        max_steps=15000,
        log_every=10,
        columns=FULL_COLUMNS,
        csv="grokking_s{n}_logs_without_weight_decay.csv",
    ),
}

ARTICLE_RUNS = ("mod_wd1", "mod_wd0", "s5_wd1", "s5_wd0", "full_batch")
"""The runs whose logs the article's figures are actually built from."""


def get(key):
    if key not in RUNS:
        raise KeyError(f"unknown run '{key}'. Available: {', '.join(RUNS)}")
    return RUNS[key]
