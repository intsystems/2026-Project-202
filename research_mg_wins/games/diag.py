"""EXPLORATORY diagnosis on CAL seeds only (written after TEST_IN showed MG losing).

Why does MG fail to count cycles here? Attribution on CAL games (B in {inf, 1e5}):
  raw     the log as scored in main.py
  oracle  exact exogenous drift subtracted (attribution only, not a usable monitor)
  ma      within-window detrending: seg - moving average(2049) (a usable preprocessing)
Statistics: MG and the strongest scalar competitors. Output: results/diag_cal.csv.
Any preprocessing chosen here must be confirmed on fresh seeds (confirm.py), applied to MG
and to every scalar competitor alike.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def regen(spec):
    import sim as G
    rng = np.random.default_rng(spec["seed"])
    for attempt in range(1, 31):
        if spec.get("wide"):
            game = G.make_game(rng, spec["K"], int(rng.integers(0, 4)), scale_range=(0.3, 3.0), eps_max=0.3)
        else:
            game = G.make_game(rng, spec["K"], int(rng.integers(0, 3)))
        if G.linear_cycles(game) != spec["K"]:
            continue
        s = G.simulate(game, spec["learner"], {"alt": .05, "sim": .02, "pg": .1, "decay": .08}[spec["learner"]])
        if not G.resonant(G.subgame_freqs(s, game), tol=0.015, ratios=(1.0, 1.5, 2.0, 3.0)):
            break
    x = G.observe(game, s, spec["obs"], spec["B"], rng)
    return game, s, x


def one(spec):
    import sim as G
    import stats as ST
    import baselines as BL
    game, s, x = regen(spec)
    dr = G.drift(game)
    row = dict(seed=spec["seed"], K=spec["K"], B=float(spec["B"]), K_tr=game.K_tr,
               drift_rel=float(dr[G.BURN:].std() / max(s["u"][G.BURN:].std(), 1e-12)))
    for w in range(G.NWIN):
        a0 = G.BURN + w * G.W
        raw = x[a0:a0 + G.W]
        variants = {"raw": raw, "oracle": raw - dr[a0:a0 + G.W],
                    "ma": raw - uniform_filter1d(raw, 2049, mode="nearest")}
        for vn, seg in variants.items():
            row[f"w{w}_{vn}_acf"] = BL.acf_time(seg)
            row[f"w{w}_{vn}_MG"] = ST.estimate(seg, ST.CFG_MG).MG
            for nm, fn in (("corr_dim", ST.corr_dim), ("recurrence_rate", ST.recurrence_rate),
                           ("harmonic_count_0.01", lambda z: ST.harmonic_count(z, 0.01)),
                           ("self_repeat_long", lambda z: BL.self_repeat(z, 20, 2000)),
                           ("spectral_entropy", BL.spectral_entropy),
                           ("peak_count_0.02", lambda z: BL.peak_count(z, 0.02))):
                try:
                    row[f"w{w}_{vn}_{nm}"] = float(fn(seg))
                except Exception:  # noqa: BLE001
                    row[f"w{w}_{vn}_{nm}"] = float("nan")
    return row


if __name__ == "__main__":
    import main as MN
    specs = [s for s in MN.specs() if s["arm"] == "CAL" and s["B"] != 1e4]
    out = HERE / "results" / "diag_cal.jsonl"
    with Pool(2) as p, open(out, "w") as fh:
        for r in p.imap_unordered(one, specs):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
    print("done")
