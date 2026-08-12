"""Registry of the figures reported in ``icomp_article/grokking_en.tex``.

Every entry pins the log file, the observation function (scalar metric) and the
sliding-window parameters used to produce one figure of the paper, so that
``reproduce_figures.py`` regenerates the article's images from the raw logs.
"""

from dataclasses import dataclass, field
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "grokking_logs"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


@dataclass(frozen=True)
class Figure:
    """One figure: a log file, an observable, and the estimator settings."""

    key: str
    kind: str                      # "panels" | "diagnostic" | "overview"
    csv: str
    description: str
    article_files: tuple = ()      # corresponding file(s) in icomp_article/images
    metric: str = "weight_norm"
    method: str = "mle"
    tau_selector: str = "fixed"
    window_size: int = 300
    step_size: int = 50
    include_last_window: bool = True
    smooth_window: int = 150
    required_columns: tuple = ("step", "train_acc", "val_acc")
    extra: dict = field(default_factory=dict)

    @property
    def csv_path(self):
        return LOG_DIR / self.csv


MOD_WD1_CSV = "grokking_modular_addition_logs_to_flat_grokking_with_stochastic.csv"
MOD_WD0_CSV = "grokking_modular_addition_logs_to_flat_grokking_with_stochastic_without_wight_decay.csv"
S5_WD1_CSV = "grokking_modular_addition_logs_S_5_with_stochastic.csv"
S5_WD0_CSV = "grokking_modular_addition_logs_S_5_with_stochastic_without_wight_decay.csv"
FULL_BATCH_CSV = "grokking_modular_addition_logs.csv"


FIGURES = {
    # --- Sec. 4.1, Fig. 1: modular addition (p = 113), Omnigrok 1L transformer,
    #     AdamW + mini-batches. Observable: the weight norm ||w||_2. ---
    "mod_wd1": Figure(
        key="mod_wd1",
        kind="panels",
        csv=MOD_WD1_CSV,
        description="Modular addition, WD=1.0 (grokking): dimension vs. acc / loss / norm",
        article_files=("mod_wd1_acc.png", "mod_wd1_loss.png", "mod_wd1_norm.png"),
    ),
    "mod_wd0": Figure(
        key="mod_wd0",
        kind="panels",
        csv=MOD_WD0_CSV,
        description="Modular addition, WD=0.0 (control, no grokking)",
        article_files=("mod_wd0_acc.png", "mod_wd0_loss.png", "mod_wd0_norm.png"),
    ),

    # --- Sec. 4.2, Fig. 2: composition in the non-abelian group S_5. ---
    "s5_wd1": Figure(
        key="s5_wd1",
        kind="panels",
        csv=S5_WD1_CSV,
        description="S_5 composition, WD=0.2 (grokking): dimension vs. acc / loss / norm",
        article_files=("s5_wd1_acc.png", "s5_wd1_loss.png", "s5_wd1_norm.png"),
    ),
    "s5_wd0": Figure(
        key="s5_wd0",
        kind="panels",
        csv=S5_WD0_CSV,
        description="S_5 composition, WD=0.0 (control, no grokking)",
        article_files=("s5_wd0_acc.png", "s5_wd0_loss.png", "s5_wd0_norm.png"),
    ),

    # --- Sec. 4.2, Fig. 3: the generic observable L_val recovers the algebraic
    #     complexity of the task (E ~ 4 for S_5). ---
    "s5_wd1_val_loss": Figure(
        key="s5_wd1_val_loss",
        kind="diagnostic",
        csv=S5_WD1_CSV,
        description="S_5, WD=0.2: dimension reconstructed from the validation loss",
        article_files=("s5_wd1_val_loss.pdf",),
        metric="val_loss",
        include_last_window=False,
    ),
    "s5_wd0_val_loss": Figure(
        key="s5_wd0_val_loss",
        kind="diagnostic",
        csv=S5_WD0_CSV,
        description="S_5, WD=0.0: dimension from validation loss (grows -- eternal memorization)",
        article_files=("s5_wd0_val_loss.pdf",),
        metric="val_loss",
        include_last_window=False,
    ),

    # --- Appendix B: full-batch GD baseline (p = 97, standard 1L transformer).
    #     Observable: the training loss, which degenerates as it approaches 0. ---
    "grokking_dimension": Figure(
        key="grokking_dimension",
        kind="diagnostic",
        csv=FULL_BATCH_CSV,
        description="Full-batch GD baseline: MLE dimension of the L_train series",
        article_files=("grokking_dimension.png",),
        metric="train_loss",
        window_size=1500,
        step_size=300,
        include_last_window=False,
    ),
    "grokking_accuracy": Figure(
        key="grokking_accuracy",
        kind="overview",
        csv=FULL_BATCH_CSV,
        description="Full-batch GD baseline: smoothed accuracy with t_mem / t_gen markers",
        article_files=("grokking_accuracy.png",),
        smooth_window=150,
    ),
}


def get(key):
    if key not in FIGURES:
        raise KeyError(f"unknown figure '{key}'. Available: {', '.join(FIGURES)}")
    return FIGURES[key]
