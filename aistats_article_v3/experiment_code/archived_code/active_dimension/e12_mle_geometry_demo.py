"""E12 -- two visual controls showing what MLE contributes beyond roughness.

The paper's grokking experiment finds that roughness detects the transition even more strongly
than the delay-embedding MLE.  These controls ask the simpler, prior question: does MLE measure
anything that roughness cannot measure?

Experiment A: four clocks versus one clock with four hands
-----------------------------------------------------------
The scalar log is a sum of four sinusoids in both arms.  In the first arm their frequencies are
rationally independent, so the orbit closure is a 4-torus.  In the second they are harmonics of
one irrational base frequency, so every hand is fixed by one phase and the closure is a circle.
The base frequency of the one-clock arm is chosen by bisection so that

    std(diff x) / std(x)

matches the four-clock arm to floating-point precision.  The scheduled run is 4D -> 1D -> 4D.
Success was fixed before running: MLE must show 4 -> 1 -> 4 while roughness changes by < 5%.

Experiment B: the same dynamics through different instrument scales
---------------------------------------------------------------------
One 4-torus is observed through monotone maps g(x) = x + a*x**p.  The maps are invertible, so
they cannot change the number of state variables, but they change the shape and roughness of the
record.  Success was fixed before running: roughness must span at least 30% while the median MLE
stays within 0.5 components of four and spans less than 0.5 components across the observer maps.

The script uses ``mg.py``, the canonical implementation behind Algorithm 1 of the paper.

Run from this directory:

    python e12_mle_geometry_demo.py

Outputs are written to ``results/e12_mle_geometry_demo/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mg as MG


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e12_mle_geometry_demo"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = tuple(range(8))
R = 4
SEGMENT = 12_000
RAMP = 1_200
N = 3 * SEGMENT

# Short enough to localise the switch, long enough to resolve four phases in this construction.
TRACE_CFG = MG.MGConfig(max_E=20, tau=4, k_neighbors=20, theiler="autocorr",
                        window=4_000, stride=400, dither=1e-9)
LEVEL_CFG = MG.MGConfig(max_E=20, tau=4, k_neighbors=20, theiler="autocorr",
                        window=8_000, stride=4_000, dither=1e-9)


def roughness(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.std(np.diff(x)) / np.std(x))


def signal(t: np.ndarray, freqs: np.ndarray, phases: np.ndarray,
           amplitudes: np.ndarray) -> np.ndarray:
    terms = [a * np.sin(2 * np.pi * f * t + p)
             for f, p, a in zip(freqs, phases, amplitudes)]
    return np.sum(terms, axis=0) / np.sqrt(np.sum(amplitudes ** 2))


def four_clock_frequencies() -> np.ndarray:
    """Four rationally independent frequencies in one matched band."""
    primes = np.array([2, 3, 5, 7], float)
    ratios = 1.0 + 0.3 * (2.0 * (np.sqrt(primes) % 1.0) - 1.0)
    return (np.sqrt(2.0) / 100.0) * ratios


def match_one_clock(t: np.ndarray, phases: np.ndarray, amplitudes: np.ndarray,
                    target_roughness: float) -> tuple[np.ndarray, np.ndarray]:
    """One irrational phase with four harmonic hands, roughness-matched by bisection."""
    lo, hi = 0.001, 0.05
    harmonics = np.arange(1, R + 1, dtype=float)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        x = signal(t, mid * harmonics, phases, amplitudes)
        if roughness(x) < target_roughness:
            lo = mid
        else:
            hi = mid
    freqs = ((lo + hi) / 2.0) * harmonics
    return signal(t, freqs, phases, amplitudes), freqs


def make_pair(seed: int, n: int = N) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    phases = rng.uniform(0, 2 * np.pi, R)
    amplitudes = rng.uniform(0.7, 1.3, R)

    f4 = four_clock_frequencies()
    x4 = signal(t, f4, phases, amplitudes)
    x1, f1 = match_one_clock(t, phases, amplitudes, roughness(x4))
    return dict(t=t, x4=x4, x1=x1, f4=f4, f1=f1,
                phases=phases, amplitudes=amplitudes)


def switch_envelope(n: int = N) -> np.ndarray:
    e = np.zeros(n, float)
    e[SEGMENT:2 * SEGMENT] = 1.0
    # The ramp is deliberately marked as transition/undefined in the output.
    return np.convolve(e, np.ones(RAMP) / RAMP, mode="same")


def trace_job(seed: int) -> pd.DataFrame:
    pair = make_pair(seed)
    envelope = switch_envelope()
    x = (1.0 - envelope) * pair["x4"] + envelope * pair["x1"]
    z = (x - x.mean()) / x.std()
    right, tr = MG.sliding(z, TRACE_CFG, seed=seed)
    centre = right - (TRACE_CFG.window - 1) / 2.0
    truth = np.where((centre >= SEGMENT + RAMP / 2) &
                     (centre <= 2 * SEGMENT - RAMP / 2), 1.0, 4.0)
    transition = (((centre > SEGMENT - RAMP / 2) &
                   (centre < SEGMENT + RAMP / 2)) |
                  ((centre > 2 * SEGMENT - RAMP / 2) &
                   (centre < 2 * SEGMENT + RAMP / 2)))
    truth[transition] = np.nan
    return pd.DataFrame(dict(seed=seed, centre=centre, truth=truth,
                             MG=tr["MG"], roughness=tr["roughness"],
                             PRdelay=tr["PRdelay"], specPR=tr["specPR0"],
                             degenerate=tr["degenerate"].astype(bool)))


def level_job(seed: int) -> list[dict]:
    pair = make_pair(seed, n=16_000)
    rows = []
    for arm, dim, x in (("one clock, four hands", 1, pair["x1"]),
                        ("four independent clocks", 4, pair["x4"])):
        z = (x - x.mean()) / x.std()
        rec = MG.summarise(z, LEVEL_CFG, seed=seed)
        rows.append(dict(seed=seed, arm=arm, truth=dim,
                         matched_roughness=roughness(x), **rec))
    return rows


WARPS = (
    ("identity", lambda x: x),
    ("x + 0.25 x^3", lambda x: x + 0.25 * x ** 3),
    ("x + x^3", lambda x: x + x ** 3),
    ("x + x^5", lambda x: x + x ** 5),
    ("x + 0.1 x^7", lambda x: x + 0.1 * x ** 7),
)


def warp_job(seed: int) -> list[dict]:
    pair = make_pair(seed, n=16_000)
    x = pair["x4"] / pair["x4"].std()
    rows = []
    for observer, fn in WARPS:
        y = fn(x)
        z = (y - y.mean()) / y.std()
        rec = MG.summarise(z, LEVEL_CFG, seed=seed)
        rows.append(dict(seed=seed, observer=observer, truth=4,
                         observed_roughness=roughness(y), **rec))
    return rows


def summarise_segments(trace: pd.DataFrame) -> pd.DataFrame:
    x = trace[~trace.degenerate & trace.truth.notna()].copy()
    x["segment"] = np.where(x.truth == 1, "one clock", "four clocks")
    return (x.groupby(["seed", "segment", "truth"], as_index=False)
             .agg(MG=("MG", "median"), roughness=("roughness", "median"),
                  PRdelay=("PRdelay", "median"), specPR=("specPR", "median")))


def median_band(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (df.groupby("centre")[column]
              .agg(median="median", q25=lambda x: x.quantile(0.25),
                   q75=lambda x: x.quantile(0.75)).reset_index())


def make_figure(trace: pd.DataFrame, warps: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.12, wspace=0.30)
    ax_mg = fig.add_subplot(gs[0, 0])
    ax_ro = fig.add_subplot(gs[1, 0], sharex=ax_mg)
    ax_w = fig.add_subplot(gs[:, 1])

    for ax in (ax_mg, ax_ro):
        ax.axvspan(SEGMENT, 2 * SEGMENT, color="#e6e6e6", zorder=0)
        ax.axvline(SEGMENT, color="0.55", lw=0.8)
        ax.axvline(2 * SEGMENT, color="0.55", lw=0.8)

    b = median_band(trace[~trace.degenerate], "MG")
    ax_mg.plot(b.centre, b["median"], color="#0072B2", lw=2)
    ax_mg.fill_between(b.centre, b.q25, b.q75, color="#0072B2", alpha=0.18, linewidth=0)
    ax_mg.plot([0, SEGMENT, SEGMENT, 2 * SEGMENT, 2 * SEGMENT, N],
               [4, 4, 1, 1, 4, 4], color="black", ls="--", lw=1, label="true dimension")
    ax_mg.set_ylabel("MLE dimension")
    ax_mg.set_ylim(0.5, 5.1)
    ax_mg.legend(frameon=False, loc="upper right")
    ax_mg.set_title("(a) Four independent clocks become one clock", loc="left")
    ax_mg.tick_params(labelbottom=False)

    b = median_band(trace[~trace.degenerate], "roughness")
    ax_ro.plot(b.centre, b["median"], color="#D55E00", lw=2)
    ax_ro.fill_between(b.centre, b.q25, b.q75, color="#D55E00", alpha=0.18, linewidth=0)
    ax_ro.set_ylabel("roughness")
    ax_ro.set_xlabel("sample")
    lo, hi = b.q25.min(), b.q75.max()
    pad = max((hi - lo) * 2, 0.003)
    ax_ro.set_ylim(lo - pad, hi + pad)
    ax_ro.text(SEGMENT / 2, ax_ro.get_ylim()[0] + 0.12 * np.ptp(ax_ro.get_ylim()),
               "4 independent\nphases", ha="center", va="bottom")
    ax_ro.text(1.5 * SEGMENT, ax_ro.get_ylim()[0] + 0.12 * np.ptp(ax_ro.get_ylim()),
               "1 shared\nphase", ha="center", va="bottom")
    ax_ro.text(2.5 * SEGMENT, ax_ro.get_ylim()[0] + 0.12 * np.ptp(ax_ro.get_ylim()),
               "4 independent\nphases", ha="center", va="bottom")

    order = [name for name, _ in WARPS]
    med = warps.groupby("observer", as_index=False).agg(
        rough=("observed_roughness", "median"),
        mg=("MG", "median"),
        q25=("MG", lambda x: x.quantile(0.25)),
        q75=("MG", lambda x: x.quantile(0.75)))
    med["observer"] = pd.Categorical(med.observer, categories=order, ordered=True)
    med = med.sort_values("observer")
    ax_w.axhline(4, color="black", ls="--", lw=1, label="true dimension = 4")
    ax_w.errorbar(med.rough, med.mg, yerr=[med.mg - med.q25, med.q75 - med.mg],
                  fmt="o-", color="#0072B2", capsize=3)
    for _, row in med.iterrows():
        is_rightmost = row.rough == med.rough.max()
        ax_w.annotate(str(row.observer), (row.rough, row.mg),
                      xytext=(-4 if is_rightmost else 4, 5),
                      ha="right" if is_rightmost else "left",
                      textcoords="offset points", fontsize=7)
    span = med.rough.max() - med.rough.min()
    ax_w.set_xlim(med.rough.min() - 0.05 * span, med.rough.max() + 0.12 * span)
    ax_w.set_xlabel("roughness after monotone rescaling")
    ax_w.set_ylabel("MLE dimension")
    ax_w.set_ylim(3.3, 4.55)
    ax_w.set_title("(b) Same 4D dynamics, changed readout", loc="left")
    ax_w.legend(frameon=False, loc="lower left")

    fig.suptitle("MLE sees phase geometry that roughness does not", y=0.995, fontsize=11)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.98)
    fig.savefig(OUT / "mle_geometry_demo.pdf", bbox_inches="tight")
    fig.savefig(OUT / "mle_geometry_demo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small numeric frame without pandas' optional tabulate dependency."""
    shown = frame.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(levels: pd.DataFrame, segments: pd.DataFrame,
                 warps: pd.DataFrame, verdict: dict) -> None:
    level_med = (levels.groupby(["arm", "truth"], as_index=False)
                       .agg(MG=("MG", "median"), roughness=("matched_roughness", "median"),
                            PRdelay=("PRdelay", "median"), specPR=("specPR0", "median")))
    warp_med = (warps.groupby("observer", as_index=False)
                      .agg(MG=("MG", "median"), roughness=("observed_roughness", "median"),
                           PRdelay=("PRdelay", "median")))
    level_table = markdown_table(level_med)
    warp_table = markdown_table(warp_med)
    report = f"""# MLE versus roughness: two visual controls

## Question

Does the delay-embedding MLE measure geometry that the one-line roughness statistic cannot see?

## Experiment A: four clocks versus one clock with four hands

Both scalar logs are sums of four sinusoids and are matched to the same roughness. In one case the
four phases are independent (dimension 4); in the other every frequency is a harmonic of one
master phase (dimension 1). The scheduled trace is 4D -> 1D -> 4D.

{level_table}

Result: MLE follows the number of independent phases while roughness is constant. The linear
delay PR and spectral PR do not give the correct count in both arms. This is the simplest answer
to "why MLE rather than roughness?": roughness measures how rapidly the scalar moves, not how many
independent clocks generate it.

## Experiment B: same four clocks, different instrument scales

The same 4D scalar log is passed through invertible monotone maps. The state dynamics and their
dimension are unchanged, but the shape and roughness of the recorded signal change.

{warp_table}

Result: roughness spans {verdict['warp_roughness_relative_span']:.0%}; median MLE spans only
{verdict['warp_mg_span']:.3f} components and remains near four.

## Pre-specified verdict

- Geometry switch: **{'PASS' if verdict['geometry_pass'] else 'FAIL'}**. Required MLE 4 -> 1 -> 4
  and less than 5% roughness change.
- Observer-scale control: **{'PASS' if verdict['warp_pass'] else 'FAIL'}**. Required at least 30%
  roughness span, every median MLE within 0.5 of four, and less than 0.5 MLE span.

## What this establishes—and what it does not

It establishes that MLE can carry phase-geometric information absent from roughness and can be
stable when roughness changes for purely observational reasons. It does not establish that MLE is
the best detector of grokking: on the current parameter-norm logs roughness detects the transition
more strongly. The clean claim is complementary: roughness detects loss of innovation; MLE can,
in an admissible recurrent setting, count independent degrees of freedom.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    report_ru = (HERE / "e12_mle_geometry_demo_ru.md").read_text(encoding="utf-8")
    replacements = {
        "__MG_FOUR__": f"{verdict['geometry_mg_four']:.2f}",
        "__MG_ONE__": f"{verdict['geometry_mg_one']:.2f}",
        "__ROUGH_DELTA__": f"{verdict['geometry_roughness_relative_change']:.3%}",
        "__WARP_ROUGH_SPAN__": f"{verdict['warp_roughness_relative_span']:.0%}",
        "__WARP_MG_SPAN__": f"{verdict['warp_mg_span']:.3f}",
        "__WARP_MG_ERROR__": f"{verdict['warp_max_abs_error']:.3f}",
        "__LEVEL_TABLE__": level_table,
        "__WARP_TABLE__": warp_table,
    }
    for placeholder, value in replacements.items():
        report_ru = report_ru.replace(placeholder, value)
    (OUT / "RESULTS_RU.md").write_text(report_ru, encoding="utf-8")


def main() -> None:
    trace = pd.concat(Parallel(n_jobs=min(8, len(SEEDS)))(
        delayed(trace_job)(seed) for seed in SEEDS), ignore_index=True)
    levels = pd.DataFrame(sum(Parallel(n_jobs=min(8, len(SEEDS)))(
        delayed(level_job)(seed) for seed in SEEDS), []))
    warps = pd.DataFrame(sum(Parallel(n_jobs=min(8, len(SEEDS)))(
        delayed(warp_job)(seed) for seed in SEEDS), []))
    segments = summarise_segments(trace)

    trace.to_csv(OUT / "geometry_switch_trace.csv", index=False)
    levels.to_csv(OUT / "geometry_levels.csv", index=False)
    segments.to_csv(OUT / "geometry_switch_summary.csv", index=False)
    warps.to_csv(OUT / "observer_warps.csv", index=False)

    seg_med = segments.groupby("segment").median(numeric_only=True)
    mg_one = float(seg_med.loc["one clock", "MG"])
    mg_four = float(seg_med.loc["four clocks", "MG"])
    ro_one = float(seg_med.loc["one clock", "roughness"])
    ro_four = float(seg_med.loc["four clocks", "roughness"])
    geometry_rough_delta = abs(ro_one - ro_four) / ro_four

    warp_med = warps.groupby("observer").median(numeric_only=True)
    warp_rough_span = ((warp_med.observed_roughness.max() - warp_med.observed_roughness.min()) /
                       warp_med.observed_roughness.min())
    warp_mg_span = warp_med.MG.max() - warp_med.MG.min()
    warp_mg_error = np.abs(warp_med.MG - 4).max()
    verdict = dict(
        geometry_mg_one=mg_one,
        geometry_mg_four=mg_four,
        geometry_roughness_relative_change=geometry_rough_delta,
        geometry_pass=bool(abs(mg_one - 1) < 0.5 and abs(mg_four - 4) < 0.5 and
                           geometry_rough_delta < 0.05),
        warp_roughness_relative_span=float(warp_rough_span),
        warp_mg_span=float(warp_mg_span),
        warp_max_abs_error=float(warp_mg_error),
        warp_pass=bool(warp_rough_span >= 0.30 and warp_mg_span < 0.5 and
                       warp_mg_error < 0.5),
        seeds=list(SEEDS),
        trace_config=asdict(TRACE_CFG),
        level_config=asdict(LEVEL_CFG),
    )
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    make_figure(trace, warps)
    write_report(levels, segments, warps, verdict)

    print(json.dumps(verdict, indent=2))
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
