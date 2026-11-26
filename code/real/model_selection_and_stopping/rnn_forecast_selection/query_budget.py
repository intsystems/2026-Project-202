"""Estimate MG from a subset of query vectors and all eligible neighbors."""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.embedding import reconstruct

CONFIG = EstimatorConfig(max_E=20, tau="acorr", k_neighbors=20, theiler="autocorr", theiler_cap=150)


def sampled(x, q=128, backend="blocked"):
    if q < 2:
        raise ValueError("At least two queries are required.")
    rec = reconstruct(x, CONFIG, seed=123)
    if not rec.usable:
        return np.nan
    points = rec.points
    n, k, exclusion = len(points), CONFIG.k_neighbors, rec.theiler
    if n < k + 2 * exclusion + 1:
        return np.nan

    queries = np.linspace(0, n - 1, min(n, q), dtype=int)
    if backend == "tree":
        distances, indices = KDTree(points).query(points[queries], k=k + 2 * exclusion + 1)
        excluded = np.abs(indices - queries[:, None]) <= exclusion
        order = np.argsort(excluded, axis=1, kind="stable")[:, :k]
        distances = np.take_along_axis(distances, order, axis=1)
    else:
        chunks = []
        for start in range(0, len(queries), 64):
            indices = queries[start:start + 64]
            squared = cdist(points[indices], points, metric="sqeuclidean")
            squared[np.abs(indices[:, None] - np.arange(n)) <= exclusion] = np.inf
            nearest = np.partition(squared, k - 1, axis=1)[:, :k]
            chunks.append(np.sqrt(np.sort(nearest, axis=1)))
        distances = np.concatenate(chunks)

    distances = np.maximum(distances, CONFIG.floor_distance)
    sums = np.log(distances[:, -1:] / distances[:, :-1]).sum(axis=1)
    distance_fraction = np.mean(distances <= CONFIG.floor_distance * 1.000001)
    sum_fraction = np.mean(sums <= CONFIG.floor_ratio_sum)
    if max(distance_fraction, sum_fraction) > CONFIG.degenerate_fraction:
        return np.nan
    return (len(queries) * (k - 1) - 1) / np.maximum(sums, CONFIG.floor_ratio_sum).sum()
