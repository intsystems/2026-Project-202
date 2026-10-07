"""Hyperparameter grids. PILOT is the candidate grid run on the pilot seed (99) only."""

PILOT = {
    # SimSiam
    "ss_base": dict(method="simsiam"),
    "ss_lr0.01": dict(method="simsiam", lr=0.01),
    "ss_lr0.2": dict(method="simsiam", lr=0.2),
    "ss_lr0.5": dict(method="simsiam", lr=0.5),
    "ss_wd1e-3": dict(method="simsiam", wd=1e-3),
    "ss_wd5e-3": dict(method="simsiam", wd=5e-3),
    "ss_wd2e-2": dict(method="simsiam", wd=2e-2),
    "ss_nopred": dict(method="simsiam", predictor=False),
    "ss_nosg": dict(method="simsiam", stopgrad=False),
    "ss_weak": dict(method="simsiam", aug="weak"),
    "ss_strong": dict(method="simsiam", aug="strong"),
    "ss_emb16": dict(method="simsiam", emb=16),
    "ss_projnobn": dict(method="simsiam", proj_bn=False),
    # BYOL
    "byol_base": dict(method="byol"),
    "byol_ema0.9": dict(method="byol", ema=0.9),
    "byol_nopred": dict(method="byol", predictor=False),
    "byol_strong": dict(method="byol", aug="strong"),
    # VICReg
    "vic_base": dict(method="vicreg", lr=0.01),
    "vic_lr0.05": dict(method="vicreg", lr=0.05),
    "vic_nocov": dict(method="vicreg", lr=0.01, vic_cov=0.0),
    "vic_novar": dict(method="vicreg", lr=0.01, vic_var=0.0),
    "vic_var5": dict(method="vicreg", lr=0.01, vic_var=5.0),
    "vic_cov10": dict(method="vicreg", lr=0.01, vic_cov=10.0),
    "vic_weak": dict(method="vicreg", lr=0.01, aug="weak"),
    # SimCLR
    "clr_t0.1": dict(method="simclr", temp=0.1),
    "clr_t0.5": dict(method="simclr", temp=0.5),
    "clr_t2": dict(method="simclr", temp=2.0),
    "clr_strong": dict(method="simclr", temp=0.2, aug="strong"),
    "clr_weak": dict(method="simclr", temp=0.2, aug="weak"),
}

# Main grid (frozen after the pilot on seed 99): the same 29 configurations, 3000 steps.
MAIN = {k: dict(v, steps=3000) for k, v in PILOT.items()}
CAL_SEEDS = (0, 1)
TEST_SEEDS = (10, 11, 12)
