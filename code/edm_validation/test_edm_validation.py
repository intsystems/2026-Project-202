"""Known-answer calibration for the surrogate and forecast machinery.

Nothing in this campaign is applied to project data before it has produced the right
answer on systems whose answer is known. Deterministic chaos must be detected; linearly
correlated noise must not be; a straight line must not masquerade as either.

    python test_edm_validation.py
"""

import numpy as np

from forecast import delay_embed, recurrence_stats, simplex_skill, skill_vs_horizon
from surrogates import iaaft, phase_randomised, surrogate_test


# --- reference systems -----------------------------------------------------

def logistic_map(n=3000, r=3.9, x0=0.4, burn_in=200):
    """Deterministic chaos on a 1-D attractor -- the driver used in poisoned_batch."""
    x, out = x0, []
    for _ in range(n + burn_in):
        x = r * x * (1 - x)
        out.append(x)
    return np.array(out[burn_in:])


def ar1(n=3000, phi=0.95, seed=0):
    """Linearly correlated noise: smooth, strongly autocorrelated, no determinism."""
    rng = np.random.default_rng(seed)
    x, out = 0.0, []
    for _ in range(n):
        x = phi * x + rng.normal()
        out.append(x)
    return np.array(out)


def lorenz_x(n=6000, dt=0.01, s=10.0, r=28.0, b=8 / 3, burn_in=1000):
    state, out = np.array([1.0, 1.0, 1.0]), []
    for _ in range(n + burn_in):
        x, y, z = state
        state = state + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        out.append(state[0])
    return np.array(out[burn_in:])


def _report(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


# --- surrogate machinery ---------------------------------------------------

def test_surrogates_preserve_what_they_should():
    """FT preserves the spectrum; IAAFT preserves the spectrum *and* the values."""
    rng = np.random.default_rng(0)
    x = lorenz_x(3000)

    ft = phase_randomised(x, rng)
    spec_err = (np.linalg.norm(np.abs(np.fft.rfft(ft)) - np.abs(np.fft.rfft(x)))
                / np.linalg.norm(np.abs(np.fft.rfft(x))))
    ok_ft = spec_err < 1e-8

    sur = iaaft(x, rng)
    same_values = np.allclose(np.sort(sur), np.sort(x))
    spec_err2 = (np.linalg.norm(np.abs(np.fft.rfft(sur)) - np.abs(np.fft.rfft(x)))
                 / np.linalg.norm(np.abs(np.fft.rfft(x))))
    ok_iaaft = same_values and spec_err2 < 0.05

    assert _report("FT preserves spectrum", ok_ft, f"rel err {spec_err:.2e}")
    assert _report("IAAFT preserves values + spectrum", ok_iaaft,
                   f"identical multiset={same_values}, spec err {spec_err2:.2e}")


def test_surrogate_test_rejects_chaos_and_spares_linear_noise():
    """The decisive calibration: the null must be rejected only where it is false."""
    def skill(series):
        return simplex_skill(series, E=3, tau=1, horizon=1)

    chaos = surrogate_test(logistic_map(2000), skill, n_surrogates=39, seed=1)
    noise = surrogate_test(ar1(2000), skill, n_surrogates=39, seed=1)

    ok_chaos = chaos.p_value <= 0.05
    ok_noise = noise.p_value > 0.05
    print(f"      logistic map : {chaos}")
    print(f"      AR(1)        : {noise}")
    assert _report("rejects the null on deterministic chaos", ok_chaos,
                   f"p={chaos.p_value:.3f}")
    assert _report("does NOT reject on linearly correlated noise", ok_noise,
                   f"p={noise.p_value:.3f}")


# --- forecast machinery ----------------------------------------------------

def test_simplex_skill_ranks_systems_correctly():
    """High skill on chaos and on a sinusoid, low on white noise."""
    rng = np.random.default_rng(0)
    values = {
        "logistic": simplex_skill(logistic_map(2000), E=3),
        "sine": simplex_skill(np.sin(np.linspace(0, 120 * np.pi, 2000)), E=3),
        "white": simplex_skill(rng.normal(size=2000), E=3),
    }
    print("      skills:", {k: round(v, 3) for k, v in values.items()})
    ok = values["logistic"] > 0.9 and values["sine"] > 0.9 and abs(values["white"]) < 0.3
    assert _report("simplex ranks chaos/periodic above white noise", ok, str(
        {k: round(v, 3) for k, v in values.items()}))


def test_skill_decays_for_chaos_not_for_periodic():
    """The signature that separates chaos from a limit cycle.

    The horizons have to reach far enough for the Lyapunov divergence to matter. With
    ``N`` points on a 1-D map the typical neighbour separation is ~1/N, so error reaches
    the attractor scale only around ``h ~ ln(N)/lambda`` -- about 15 steps at N=3000,
    lambda~0.5. Testing at h=8 (the first attempt) measured nothing: skill was still
    0.95, correctly.
    """
    horizons = (1, 8, 16, 24, 32)
    chaos = skill_vs_horizon(logistic_map(3000), E=3, horizons=horizons)
    periodic = skill_vs_horizon(np.sin(np.linspace(0, 120 * np.pi, 3000)), E=3,
                                horizons=horizons)
    print("      chaos   :", {k: round(v, 3) for k, v in chaos.items()})
    print("      periodic:", {k: round(v, 3) for k, v in periodic.items()})
    ok = chaos[32] < chaos[1] - 0.2 and periodic[32] > 0.8
    assert _report("skill decays with horizon for chaos, not for a cycle", ok,
                   f"chaos {chaos[1]:.2f}->{chaos[32]:.2f}, periodic {periodic[32]:.2f}")


def test_recurrence_separates_attractor_from_transient():
    """The precondition check itself must be calibrated."""
    attractor = recurrence_stats(lorenz_x(4000), E=5, tau=5)
    ramp = recurrence_stats(np.linspace(0, 1, 4000) ** 2, E=5, tau=5)
    print("      Lorenz   profile:", {k: round(v, 4) for k, v in attractor["profile"].items()})
    print("      monotone profile:", {k: round(v, 4) for k, v in ramp["profile"].items()})
    ok = attractor["ratio"] > 0.4 and ramp["ratio"] < 0.15
    assert _report("recurrence separates an attractor from a monotone transient", ok,
                   f"Lorenz ratio {attractor['ratio']:.3f}, ramp {ramp['ratio']:.3f}")


def test_delay_embed_shape():
    v, idx = delay_embed(np.arange(10.0), E=3, tau=2)
    ok = v.shape == (6, 3) and np.allclose(v[0], [4, 2, 0]) and idx[0] == 4
    assert _report("delay_embed indexes backwards in time", ok, f"shape {v.shape}")


def coupled_logistic(n, b_xy, b_yx, rx=3.8, ry=3.5, burn=500):
    """Sugihara et al. (2012), eq. 1. ``b_yx`` is the X->Y strength, ``b_xy`` the Y->X."""
    x, y, X, Y = 0.4, 0.2, [], []
    for i in range(n + burn):
        xn = x * (rx - rx * x - b_xy * y)
        yn = y * (ry - ry * y - b_yx * x)
        x, y = min(max(xn, 1e-9), 1.0), min(max(yn, 1e-9), 1.0)
        if i >= burn:
            X.append(x)
            Y.append(y)
    return np.array(X), np.array(Y)


def test_cross_map_recovers_the_direction_of_coupling():
    """The direction convention, calibrated on a system whose answer is known.

    "Y xmap X" -- embedding Y and predicting X -- is evidence that X drives Y. If that
    convention were inverted in `ccm.py`, every causal claim in the report would invert
    with it, so it is checked here rather than assumed.

    Note what does *not* discriminate: the false direction still reaches rho ~ 0.57,
    because Y is a function of X and a smooth function of a variable is partly
    predictable from it. Convergence with library size is the discriminator, which is
    why `ccm_test` requires it alongside the surrogate null.
    """
    from ccm import ccm_convergence

    X, Y = coupled_logistic(3000, b_xy=0.0, b_yx=0.32)   # X drives Y, not the reverse
    sizes = (20, 40, 80, 160, 320, 640, 1600, 2900)
    true_dir = ccm_convergence(Y, X, E=3, tau=1, library_sizes=sizes)
    false_dir = ccm_convergence(X, Y, E=3, tau=1, library_sizes=sizes)

    gain_true = true_dir[2900] - true_dir[20]
    gain_false = false_dir[2900] - false_dir[20]
    print(f"      Y xmap X (true) : {true_dir[20]:.3f} -> {true_dir[2900]:.3f}")
    print(f"      X xmap Y (false): {false_dir[20]:.3f} -> {false_dir[2900]:.3f}")

    ok = true_dir[2900] > 0.9 and gain_true > 0.3 and gain_false < 0.1
    assert _report("cross mapping recovers the direction of coupling", ok,
                   f"true gain {gain_true:+.3f}, false gain {gain_false:+.3f}")


if __name__ == "__main__":
    checks = [
        test_delay_embed_shape,
        test_cross_map_recovers_the_direction_of_coupling,
        test_surrogates_preserve_what_they_should,
        test_simplex_skill_ranks_systems_correctly,
        test_skill_decays_for_chaos_not_for_periodic,
        test_recurrence_separates_attractor_from_transient,
        test_surrogate_test_rejects_chaos_and_spares_linear_noise,
    ]
    results = []
    for check in checks:
        print(f"\n--- {check.__name__} ---")
        try:
            check()
            results.append(True)
        except AssertionError:
            # _report has already printed the FAIL line with the numbers.
            results.append(False)
        except Exception as exc:                             # noqa: BLE001
            print(f"FAIL  {check.__name__} raised: {exc}")
            results.append(False)
    print(f"\n{sum(results)}/{len(results)} calibration checks passed")
    raise SystemExit(0 if all(results) else 1)
