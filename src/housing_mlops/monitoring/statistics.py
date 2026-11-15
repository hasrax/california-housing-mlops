from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_1samp

PSI_EPSILON = 1e-4


def _as_1d_float(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return array


def observed_fractions(values, bin_edges) -> np.ndarray:
    array = _as_1d_float(values)
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must contain at least two values")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")
    open_edges = edges.copy()
    open_edges[0] = -np.inf
    open_edges[-1] = np.inf
    counts, _ = np.histogram(array, bins=open_edges)
    return counts / counts.sum()


def population_stability_index(
    reference_fractions, observed, *, epsilon: float = PSI_EPSILON
) -> float:
    reference = np.asarray(reference_fractions, dtype=float)
    actual = np.asarray(observed, dtype=float)
    if reference.shape != actual.shape:
        raise ValueError("reference and observed fractions must have the same shape")
    if np.any(reference < 0) or np.any(actual < 0):
        raise ValueError("fractions must be non-negative")
    reference = np.clip(reference, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    reference = reference / reference.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - reference) * np.log(actual / reference)))


def psi_from_reference(reference_fractions, bin_edges, values) -> float:
    return population_stability_index(reference_fractions, observed_fractions(values, bin_edges))


def reference_cdf(quantiles, grid) -> Callable[[np.ndarray], np.ndarray]:
    xs = np.asarray(quantiles, dtype=float)
    ps = np.asarray(grid, dtype=float)
    if xs.shape != ps.shape or xs.ndim != 1 or xs.size < 2:
        raise ValueError("quantiles and grid must be one-dimensional with the same length >= 2")
    if np.any(np.diff(xs) < 0) or np.any(np.diff(ps) < 0):
        raise ValueError("quantiles and grid must be non-decreasing")
    unique_xs, first_index = np.unique(xs, return_index=True)
    last_index = np.array([np.flatnonzero(xs == x)[-1] for x in unique_xs])
    p_first = ps[first_index]
    p_last = ps[last_index]
    n = unique_xs.size

    def cdf(x):
        points = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(unique_xs, points, side="right") - 1
        result = np.empty_like(points)
        below = idx < 0
        at_or_above_last = idx >= n - 1
        middle = ~below & ~at_or_above_last
        result[below] = 0.0
        result[at_or_above_last] = p_last[-1]
        i = idx[middle]
        x0, x1 = unique_xs[i], unique_xs[i + 1]
        p0, p1 = p_last[i], p_first[i + 1]
        t = (points[middle] - x0) / (x1 - x0)
        result[middle] = p0 + t * (p1 - p0)
        return result

    return cdf


@dataclass(frozen=True)
class KsResult:
    statistic: float
    p_value: float


def ks_against_reference(quantiles, grid, values) -> KsResult:
    array = _as_1d_float(values)
    result = ks_1samp(array, reference_cdf(quantiles, grid))
    return KsResult(statistic=float(result.statistic), p_value=float(result.pvalue))
