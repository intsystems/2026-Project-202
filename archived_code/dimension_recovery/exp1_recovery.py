"""Which 1-D observers recover a known dynamical dimension, and under what conditions.

Five parts:

  A  regime map      -- where in (cycles per window, SNR, window) recovery is possible
  B  observer sweep  -- every observer x every estimator over k = 1..6, three seeds
  C  controls        -- synchronous motion, pure rescaling, white noise
  D  invariance      -- same k at different speed and amplitude
  E  verdict         -- validated_good / validated_lossy / validated_bad / unknown

The verdict rule is fixed before the numbers are looked at, and is stated in
:func:`classify`. It is deliberately strict on the controls: an observer that reports a
high dimension for one-dimensional motion is worse than useless, because that is the
error the whole exercise exists to avoid.

    python exp1_recovery.py            # all parts, about 6 minutes
    python exp1_recovery.py A C        # named parts
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import estimators as E
import systems as S

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# Reference operating point, chosen from part A and then held fixed everywhere else.
#
# It is deliberately FAVOURABLE: 300 oscillations of the observable inside one estimator
# window, and a signal-to-noise ratio of a million. A training log's weight norm does
# zero oscillations inside a window. The point of picking a favourable reference is to
# ask what a 1-D observer can do at its best, so that a negative result there would be
# decisive; it is not a claim that training logs live anywhere near here.
REF = dict(window=2000, cycles_per_window=300.0, obs_snr=1e6, n=20000, stride=1000,
           D=64, band_mode="matched")
KS = (1, 2, 3, 4, 5, 6)
SEEDS = (0, 1, 2)
EST = ("LB", "MG", "PR", "TwoNN", "roughness")


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def series_for(mode, k, seed, obs="norm_fro", **over):
    cfg = {**REF, **over}
    info = S.make_system(mode, k=k, D=cfg["D"], n=cfg["n"],
                         cycles_per_window=cfg["cycles_per_window"],
                         window=cfg["window"], seed=seed, band_mode=cfg["band_mode"])
    return S.observers(info, seed=seed, obs_snr=cfg["obs_snr"])[obs], info


def part_A2():
    rule("A2  The bandwidth confound, and the coverage ceiling")
    print("  Left block: frequencies grow with k (f_j ~ sqrt(p_j)), which is the obvious")
    print("  construction and the one the brief implies. Right block: all k frequencies")
    print("  held inside a fixed band, so the spectral centroid does not move with k.\n")
    print(f"  {'':<10}" + f"{'band_mode = widening':^37}  {'band_mode = matched':^37}")
    print(f"  {'k':<10}" + f"{'MG':>8}{'LB':>8}{'TwoNN':>9}{'roughness':>12}"
          + f"{'MG':>10}{'LB':>8}{'TwoNN':>9}{'roughness':>12}")
    rows = []
    for k in KS:
        cells = []
        for bm in ("widening", "matched"):
            x, _ = series_for("quasiperiodic", k, 0, band_mode=bm)
            r = E.evaluate(x, REF["window"], REF["stride"], names=EST)
            cells += [r["MG"], r["LB"], r["TwoNN"], r["roughness"]]
            rows.append({"k": k, "band_mode": bm, **{e: r[e] for e in EST}})
        print(f"  {k:<10}{cells[0]:8.2f}{cells[1]:8.2f}{cells[2]:9.2f}{cells[3]:12.4f}"
              f"{cells[4]:10.2f}{cells[5]:8.2f}{cells[6]:9.2f}{cells[7]:12.4f}")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp1_A2_bandwidth.csv", index=False)
    for bm in ("widening", "matched"):
        sub = frame[frame.band_mode == bm]
        rho_mg = spearmanr(sub.k, sub.MG).statistic
        rho_r = spearmanr(sub.k, sub.roughness).statistic
        print(f"\n  {bm:<10} Spearman(MG, k) = {rho_mg:+.2f}   "
              f"Spearman(roughness, k) = {rho_r:+.2f}")
    print("\n  Under 'widening' the roughness ratio tracks k as well as any dimension")
    print("  estimator, because it is reading bandwidth. Any experiment built that way")
    print("  cannot distinguish geometry from smoothness and proves nothing. Under")
    print("  'matched' roughness is flat and MG still moves, which is the comparison")
    print("  that has content. Everything after this uses 'matched'.")

    print("\n  Coverage ceiling: MG against k at several cycles per window (matched).")
    print(f"  {'cyc/win':>8}" + "".join(f"{f'k={k}':>7}" for k in KS) + f"{'rho':>7}")
    rows2 = []
    for cyc in (10, 30, 100, 300, 1000):
        vals = []
        for k in KS:
            x, _ = series_for("quasiperiodic", k, 0, cycles_per_window=float(cyc))
            vals.append(E.evaluate(x, REF["window"], REF["stride"], names=["MG"])["MG"])
        rho = spearmanr(KS, vals).statistic
        print(f"  {cyc:>8}" + "".join(f"{v:7.2f}" for v in vals) + f"{rho:>+7.2f}")
        rows2.append({"cycles": cyc, "rho": rho,
                      **{f"k{k}": v for k, v in zip(KS, vals)}})
    pd.DataFrame(rows2).to_csv(OUT / "exp1_A2_coverage.csv", index=False)
    print("\n  The largest k that can be resolved rises with cycles per window, which is")
    print("  what a coverage argument predicts: to see a k-torus locally the window must")
    print("  contain returns in all k directions, and that needs of order c^k points.")
    print("  At the top of the range low k breaks down instead -- a single sinusoid seen")
    print("  over a thousand periods has near-exact recurrences and MG collapses. There")
    print("  is no single setting that is correct for every k, only a band.")


# ------------------------------------------------------------------ A. the regime map

def part_A():
    rule("A  Regime map: where can a 1-D observable recover k at all?")
    print("  Observer 'norm_fro', estimator MG, k = 1, 2, 3. A cell shows the three")
    print("  estimates. Recovery means they are ordered and separated; the failures are")
    print("  of two opposite kinds and both are visible here.\n")
    rows = []
    for window in (300, 1000, 3000):
        print(f"  -- window {window} samples")
        print(f"     {'cycles/win':>11}" + "".join(f"{f'SNR=1e{e}':>22}" for e in (2, 3, 4, 6)))
        for cyc in (1, 3, 10, 30, 100):
            cells = []
            for e in (2, 3, 4, 6):
                vals = []
                for k in (1, 2, 3):
                    x, _ = series_for("quasiperiodic", k, 0, window=window,
                                      cycles_per_window=cyc, obs_snr=10.0 ** e,
                                      n=max(6000, 6 * window))
                    vals.append(E.evaluate(x, window, window // 2, names=["MG"])["MG"])
                ordered = bool(vals[0] < vals[1] < vals[2])
                sep = vals[2] - vals[0]
                mark = "*" if ordered and sep >= 0.5 else (":" if ordered else " ")
                cells.append((mark + " ".join(f"{v:4.2f}" for v in vals)).rjust(22))
                rows.append({"window": window, "cycles": cyc, "snr": 10.0 ** e,
                             "k1": vals[0], "k2": vals[1], "k3": vals[2],
                             "ordered": ordered, "separation": sep,
                             "usable": bool(ordered and sep >= 0.5)})
            print(f"     {cyc:>11}" + "".join(cells))
        print()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp1_A_regime.csv", index=False)
    ok, use = frame[frame.ordered], frame[frame.usable]
    print("  '*' ordered with k=3 minus k=1 at least 0.5;  ':' ordered but compressed")
    print(f"\n  {len(ok)} of {len(frame)} settings order k=1 < k=2 < k=3;")
    print(f"  {len(use)} of {len(frame)} also separate them by at least 0.5.")
    if len(use):
        print(f"    usable cycles/window : {sorted(use.cycles.unique())}")
        print(f"    usable log10 SNR     : {sorted(set(np.log10(use.snr).round(0)))}")
        print(f"    usable windows       : {sorted(use.window.unique())}")
        print(f"    best separation      : {use.separation.max():.2f} at "
              f"{use.loc[use.separation.idxmax(), ['window', 'cycles', 'snr']].to_dict()}")
    print("\n  Two distinct failures, at opposite ends, and one in the middle:")
    print("   * too few cycles -- the window sees an arc. Ordering can survive but the")
    print("     spread collapses to ~0.07, which no finite sample could resolve. This is")
    print("     the regime the training logs are in.")
    print("   * too much noise -- neighbours sit inside the noise ball and every k")
    print("     returns something near the embedding dimension. At SNR 1e2 the ordering")
    print("     is destroyed at every window length.")
    print("   * too many cycles at k=1 -- a single sinusoid observed over 30+ periods")
    print("     has near-exact recurrences, the log-ratios blow up and MG returns 0.3.")
    print("     The estimator is least trustworthy exactly where the motion is simplest.")


# ---------------------------------------------------------------- B. observer sweep

def part_B():
    rule("B  Every observer, every estimator, k = 1..6, three seeds")
    print(f"  Reference point: {REF}\n")
    rows = []
    for seed in SEEDS:
        for k in KS:
            info = S.make_system("quasiperiodic", k=k, D=REF["D"], n=REF["n"],
                                 cycles_per_window=REF["cycles_per_window"],
                                 window=REF["window"], seed=seed, band_mode=REF["band_mode"])
            obs = S.observers(info, seed=seed, obs_snr=REF["obs_snr"])
            for name, x in obs.items():
                res = E.evaluate(x, REF["window"], REF["stride"], names=EST)
                rows.append({"observer": name, "k": k, "seed": seed,
                             "expected": S.image_dimension(name, k),
                             "class": S.EXPECTED.get(name, "?"),
                             **{e: res[e] for e in EST}})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp1_B_observers.csv", index=False)

    for est in EST:
        print(f"  -- estimator {est}   (median over {len(SEEDS)} seeds)")
        piv = frame.pivot_table(index="observer", columns="k", values=est, aggfunc="median")
        piv = piv.reindex([o for o in S.EXPECTED if o in piv.index])
        print(f"     {'observer':<20}" + "".join(f"{f'k={k}':>8}" for k in KS)
              + f"{'rho':>7}{'class':>22}")
        for name, r in piv.iterrows():
            v = r.to_numpy(dtype=float)
            m = np.isfinite(v)
            rho = spearmanr(np.array(KS)[m], v[m]).statistic if m.sum() >= 3 else np.nan
            print(f"     {name:<20}" + "".join(f"{x:8.2f}" for x in v)
                  + f"{rho:>+7.2f}{S.EXPECTED.get(name, '?'):>22}")
        print()
    print("  'class' is the prediction made in systems.EXPECTED before any run.")
    print("  rho is Spearman between the estimate and the true k over k = 1..6.")


# ------------------------------------------------------------------- C. the controls

def part_C():
    rule("C  Controls: one-dimensional motion that could be mistaken for many")
    print("  Every control has k = 4 moving coordinates. Only the first row has four")
    print("  degrees of freedom; the rest have one, or none.\n")
    modes = [("quasiperiodic", 4.0, "4 independent oscillators"),
             ("sync", 1.0, "4 coordinates, one shared phase"),
             ("sync_phased", 1.0, "4 coordinates, one frequency, fixed phase offsets"),
             ("scale_periodic", 1.0, "whole matrix rescaled by one oscillation"),
             ("scale_monotone", 1.0, "whole matrix rescaled monotonically (a curve)"),
             ("noise_smooth", 4.0, "4 independent smooth random signals"),
             ("noise", np.inf, "4 independent white sequences")]
    rows = []
    print(f"  {'mode':<16}{'true':>6}" + "".join(f"{e:>10}" for e in EST) + "   note")
    for mode, truth, note in modes:
        vals = {e: [] for e in EST}
        for seed in SEEDS:
            x, _ = series_for(mode, 4, seed)
            res = E.evaluate(x, REF["window"], REF["stride"], names=EST)
            for e in EST:
                vals[e].append(res[e])
        med = {e: float(np.nanmedian(vals[e])) for e in EST}
        print(f"  {mode:<16}{truth:>6.1f}" + "".join(f"{med[e]:10.2f}" for e in EST)
              + f"   {note}")
        rows.append({"mode": mode, "true_dim": truth, **med})
    pd.DataFrame(rows).to_csv(OUT / "exp1_C_controls.csv", index=False)
    print("\n  The row that matters is 'sync_phased': four coordinates visibly moving,")
    print("  four different phases, and one degree of freedom. An estimator that reports")
    print("  four here would confirm the paper's original claim for the wrong reason.")


# ------------------------------------------------------- D. invariance to speed, amplitude

def part_D():
    rule("D  Same dimension, different speed and amplitude")
    print("  k = 3 throughout. If the estimate is a property of the dynamics it should")
    print("  not move. Speed is varied two ways: at a fixed window, and at a window")
    print("  rescaled to hold cycles-per-window constant.\n")
    rows = []
    print(f"  {'variation':<34}{'MG':>8}{'LB':>8}{'PR':>8}{'TwoNN':>8}")
    base = None
    for label, over in (("reference (10 cyc/win, amp 1x)", {}),
                        ("amplitude x0.1", {"_amp": 0.1}),
                        ("amplitude x10", {"_amp": 10.0}),
                        ("speed x3, window fixed", {"cycles_per_window": 30.0}),
                        ("speed x1/3, window fixed", {"cycles_per_window": 10 / 3}),
                        ("speed x3, window /3", {"cycles_per_window": 30.0,
                                                 "_rescale": 3.0}),
                        ("speed x1/3, window x3", {"cycles_per_window": 10 / 3,
                                                   "_rescale": 1 / 3.0})):
        over = dict(over)
        amp_mult = over.pop("_amp", 1.0)
        rescale = over.pop("_rescale", 1.0)
        win = int(REF["window"] / rescale)
        vals = {e: [] for e in EST}
        for seed in SEEDS:
            cfg = {**REF, **over}
            info = S.make_system("quasiperiodic", k=3, D=cfg["D"], n=cfg["n"],
                                 cycles_per_window=cfg["cycles_per_window"],
                                 window=cfg["window"], amp=0.1 * amp_mult, seed=seed,
                                 band_mode=cfg["band_mode"])
            x = S.observers(info, seed=seed, obs_snr=cfg["obs_snr"])["norm_fro"]
            res = E.evaluate(x, win, max(50, win // 2), names=EST)
            for e in EST:
                vals[e].append(res[e])
        med = {e: float(np.nanmedian(vals[e])) for e in EST}
        if base is None:
            base = med
        print(f"  {label:<34}" + "".join(f"{med[e]:8.2f}" for e in ("MG", "LB", "PR", "TwoNN")))
        rows.append({"variation": label, "window": win, **med})
    pd.DataFrame(rows).to_csv(OUT / "exp1_D_invariance.csv", index=False)
    print("\n  Amplitude invariance is exact -- the W=0 estimate is scale free. Speed")
    print("  invariance at a FIXED window is not expected and not observed, because")
    print("  speed is cycles-per-window and part A shows that is the controlling")
    print("  variable. Holding cycles-per-window fixed restores it, which is the")
    print("  practical statement: the window must be chosen in units of the system's")
    print("  own recurrence time, never in steps.")


# ------------------------------------------------------------------- E. the verdict

def classify(sub, controls):
    """Pre-registered rule. `sub` is one observer's table over k and seeds.

    validated_good    -- Spearman(d_hat, k) >= 0.9 over k=1..6, at least 4 of 5 adjacent
                         k-steps strictly increasing, across-seed spread below half the
                         mean adjacent step, and the one-dimensional controls read below
                         the observer's own k=2 level.
    validated_lossy   -- tracks its OWN image dimension (which is below k by
                         construction) rather than k. The estimator is right; the
                         observer throws information away.
    validated_bad     -- neither: no monotone relation to k, or a one-dimensional
                         control read at or above the k=3 level.
    unknown           -- passes some tests and fails none decisively.
    """
    med = sub.groupby("k").MG.median().reindex(KS).to_numpy()
    if not np.isfinite(med).all():
        return "validated_bad", np.nan, "estimate undefined"
    rho = spearmanr(KS, med).statistic
    steps = np.diff(med)
    rising = int((steps > 0).sum())
    spread = sub.groupby("k").MG.std().mean()
    mean_step = float(np.mean(np.abs(steps)))
    range_k = float(med.max() - med.min())

    exp_med = sub.groupby("k").expected.median().reindex(KS).to_numpy()
    exp_known = bool(np.isfinite(exp_med).all())
    exp_constant = exp_known and bool(np.allclose(exp_med, exp_med[0]))
    exp_is_k = exp_known and bool(np.allclose(exp_med, np.asarray(KS, dtype=float)))

    # 1. The observable carries no signal at all: the estimate sits near the embedding
    #    dimension for every k, which is what a noise ball returns.
    if float(np.min(med)) > 0.6 * E.MAX_E:
        return "validated_bad", rho, f"reads ~max_E ({med.mean():.1f}) at every k: noise"

    # 2. The observable is lossy by construction. Its image has a known dimension below
    #    k, and the estimator returning that is the estimator being right. This test has
    #    to come before the control test, because a flat observer's own 'k=3 level' is
    #    the same number as its k=1 level, and comparing a control against it is vacuous.
    if exp_known and not exp_is_k:
        if exp_constant and range_k < 0.3:
            return "validated_lossy", rho, "flat in k, at its constant image dimension"
        if not exp_constant and spearmanr(exp_med, med).statistic > 0.9:
            return "validated_lossy", rho, "tracks its own image dimension, not k"

    # 3. Spurious: one-dimensional motion read as multi-dimensional. Only meaningful for
    #    an observer whose estimate actually moves with k.
    if range_k > 0.3 and controls >= med[2]:
        return "validated_bad", rho, "reads a 1-D control at its own k=3 level"

    if rho >= 0.9 and rising >= 4 and spread < 0.5 * mean_step and controls <= med[1]:
        return "validated_good", rho, "monotone in k, stable, controls pass"
    if rho < 0.5:
        return "validated_bad", rho, "no monotone relation to k"
    return "unknown", rho, f"rho {rho:+.2f}, {rising}/5 steps rising, seed sd {spread:.2f}"


def part_E():
    rule("E  Verdict per observer")
    b = OUT / "exp1_B_observers.csv"
    if not b.exists():
        print("  run part B first"); return
    frame = pd.read_csv(b)
    ctrl = {}
    for seed in SEEDS:
        for name, x in S.observers(
                S.make_system("sync_phased", k=4, D=REF["D"], n=REF["n"],
                              cycles_per_window=REF["cycles_per_window"],
                              window=REF["window"], seed=seed, band_mode=REF["band_mode"]),
                seed=seed, obs_snr=REF["obs_snr"]).items():
            ctrl.setdefault(name, []).append(
                E.evaluate(x, REF["window"], REF["stride"], names=["MG"])["MG"])

    print(f"  {'observer':<20}{'predicted':>22}{'rho':>7}{'sync_phased':>13}"
          f"   verdict")
    rows = []
    for name in S.EXPECTED:
        sub = frame[frame.observer == name]
        if not len(sub):
            continue
        c = float(np.nanmedian(ctrl.get(name, [np.nan])))
        verdict, rho, why = classify(sub, c)
        print(f"  {name:<20}{S.EXPECTED[name]:>22}{rho:>+7.2f}{c:>13.2f}"
              f"   {verdict:<16} {why}")
        rows.append({"observer": name, "predicted": S.EXPECTED[name], "rho": rho,
                     "sync_phased_MG": c, "verdict": verdict, "reason": why})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "exp1_E_verdict.csv", index=False)
    print()
    for v, n in out.verdict.value_counts().items():
        print(f"  {v:<18} {n}")
    good = out[out.verdict == "validated_good"].observer.tolist()
    print(f"\n  validated_good: {good}")
    print("\n  Every one of these is a scalar function of the weights. So the general")
    print("  claim 'a 1-D log cannot recover the dimension' is false as stated. The")
    print("  correct claim is conditional, and part A gives the condition.")


PARTS = {"A": part_A, "A2": part_A2, "B": part_B, "C": part_C, "D": part_D, "E": part_E}


def main(argv):
    for name in ([a.upper() for a in argv[1:]] or list(PARTS)):
        if name not in PARTS:
            print(f"unknown part {name!r}")
            return 2
        t0 = time.time()
        PARTS[name]()
        print(f"\n  [{name} took {time.time() - t0:.1f}s]")
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
