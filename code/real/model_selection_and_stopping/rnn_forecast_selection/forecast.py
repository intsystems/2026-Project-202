"""Fit and evaluate the forecasters used in the single-neuron benchmark."""

import numpy as np
from sklearn.kernel_approximation import RBFSampler


def forecast(prefix, horizon, model):
    """Return a direct forecast fitted only on the observed prefix."""
    if model == "last":
        return np.full(horizon, prefix[-1])
    if model == "mean":
        return np.full(horizon, prefix.mean())
    if model == "daily":
        return np.resize(prefix[-314:], horizon)

    lag = int(model.lstrip("abcdefghijklmnopqrstuvwxyz"))
    mean = prefix.mean()
    sd = max(prefix.std(), 1e-12)
    z = (prefix - mean) / sd
    windows = np.lib.stride_tricks.sliding_window_view(z, lag + horizon)
    inputs = windows[:, :lag]
    targets = windows[:, lag:]
    query = z[-lag:][None]
    if model.startswith("knn"):
        distances = np.sum((inputs - query) ** 2, axis=1)
        neighbors = np.argsort(distances)[:5]
        prediction = targets[neighbors].mean(axis=0)
    else:
        if model.startswith("rff"):
            features = RBFSampler(gamma=1 / lag, n_components=64, random_state=42)
            inputs = features.fit_transform(inputs)
            query = features.transform(query)
        inputs = np.column_stack([inputs, np.ones(len(inputs))])
        query = np.column_stack([query, np.ones(len(query))])
        penalty = np.eye(inputs.shape[1])
        penalty[-1, -1] = 1e-8
        weights = np.linalg.solve(inputs.T @ inputs + penalty, inputs.T @ targets)
        prediction = (query @ weights)[0]
    return np.clip(prediction, -10, 10) * sd + mean
