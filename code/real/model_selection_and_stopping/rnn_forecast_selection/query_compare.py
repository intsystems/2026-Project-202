"""Score MG and rolling validation on bundled or newly generated forecast windows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from forecast import forecast
from fresh_generators import END, SPLIT, START
from query_budget import CODE, CONFIG, sampled
from actdim.estimator.mle import estimate

LOGS = Path(__file__).resolve().parent / "logs"
MODELS = ("last", "daily", "mean", "ridge8", "ridge48", "knn8", "knn48", "rff48")
LOSSES = [f"loss_{name}" for name in MODELS]
HORIZON = END - SPLIT


def fit_rule():
    data = pd.read_csv(LOGS / "mg_calibration.csv", float_precision="round_trip")
    losses = data[LOSSES].to_numpy(copy=True)
    flat = data["std"] < 1e-10
    losses[flat] = data.loc[flat, "loss_last"].fillna(0).to_numpy()[:, None]
    rule = make_pipeline(
        SimpleImputer(add_indicator=True),
        DecisionTreeRegressor(max_depth=2, min_samples_leaf=8, random_state=42),
    )
    return rule.fit(data[["MG"]].to_numpy(), losses)


def validation_choice(prefix):
    origins = range(len(prefix) - 4 * HORIZON, len(prefix), HORIZON)
    scores = [
        np.mean([
            np.mean((forecast(prefix[:o], HORIZON, model) - prefix[o:o + HORIZON]) ** 2)
            for o in origins
        ])
        for model in MODELS
    ]
    return MODELS[np.argmin(scores)]


def compare(activity, rule):
    prefix, continuation = np.split(activity, [SPLIT - START])
    methods = ("MG", "MG_q64", "MG_q128", "MG_q256", "validation")
    if prefix.std() < 1e-10:
        choices = dict.fromkeys(methods, "last")
    else:
        full = estimate(prefix, CONFIG, seed=123)
        scores = [np.nan if full.degenerate else full.MG] + [sampled(prefix, q) for q in (64, 128, 256)]
        choices = {
            name: MODELS[index]
            for name, index in zip(methods, rule.predict(np.asarray(scores)[:, None]).argmin(axis=1))
        }
        choices["validation"] = validation_choice(prefix)
    variance = max(prefix.var(), 1e-12)
    for method, model in choices.items():
        prediction = forecast(prefix, HORIZON, model)
        yield method, model, np.mean((prediction - continuation) ** 2) / variance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", nargs="*", type=int, help="Score all available seeds by default.")
    parser.add_argument("--input", type=Path, default=LOGS)
    parser.add_argument("--output", type=Path, default=CODE / "runs" / "rnn_forecast_selection" / "comparisons.csv")
    args = parser.parse_args()

    records = []
    with threadpool_limits(limits=1):
        rule = fit_rule()
        for path in sorted(args.input.glob("fresh_seed*__*.csv")):
            seed_text, arm = path.stem.removeprefix("fresh_seed").split("__")
            seed = int(seed_text)
            if args.seeds and seed not in args.seeds:
                continue
            data = pd.read_csv(path, float_precision="round_trip")
            if not np.array_equal(data["sample"], np.arange(START, END)):
                parser.error(f"Expected samples {START} through {END - 1} in {path}.")
            for method, model, error in compare(data["neuron_2"].to_numpy(), rule):
                records.append(dict(seed=seed, arm=arm, method=method, model=model, error=error))
    if not records:
        parser.error(f"No matching forecast windows in {args.input}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(records)
    scores.to_csv(args.output, index=False)
    print(scores.groupby(["seed", "method"]).error.mean().groupby("method").agg(["mean", "std"]))


if __name__ == "__main__":
    main()
