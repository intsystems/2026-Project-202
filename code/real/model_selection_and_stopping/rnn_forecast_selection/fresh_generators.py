"""Generate forecast windows with the shared FORCE recurrent generator."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
from synthetic.poster_examples.four_tone_rnn.generator import init_multi, target_spec, train_multi

TASKS = {
    "T1": ("torus", 1), "T2": ("torus", 2), "T3": ("torus", 3), "T4": ("torus", 4),
    "H2": ("harmonic", 2), "H4": ("harmonic", 4), "M4": ("mixed", 4), "chaos": None,
}
START, SPLIT, END = 4096, 6144, 6176


def record(seed, arm):
    task = TASKS[arm]
    n = 256
    if task is None:
        recurrent, feedback, state = init_multi(seed, n, 1, gain=1.5)
        readout = np.zeros((n, 1))
    else:
        spec = target_spec(*task, seed)
        recurrent, feedback, readout, state, _ = train_multi(seed, n, spec, 40000, gain=1.2)

    weights = recurrent + feedback @ readout.T
    for _ in range(5000 + START):
        state = .9 * state + .1 * (weights @ np.tanh(state))

    activity = np.empty(END - START)
    for i in range(len(activity)):
        rates = np.tanh(state)
        activity[i] = rates[2]
        state = .9 * state + .1 * (weights @ rates)
    return pd.DataFrame({"sample": np.arange(START, END), "neuron_2": activity})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", nargs="*", type=int, default=[681])
    parser.add_argument("--arms", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=CODE / "runs" / "rnn_forecast_selection")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            for arm in args.arms:
                path = args.output / f"fresh_seed{seed}__{arm}.csv"
                record(seed, arm).to_csv(path, index=False, float_format="%.17g")
                print(path, flush=True)


if __name__ == "__main__":
    main()
