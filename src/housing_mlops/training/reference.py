from typing import Any

import numpy as np
import pandas as pd

from housing_mlops.data.schema import FEATURE_COLUMNS

QUANTILE_GRID = np.linspace(0.0, 1.0, 101)
N_BINS = 10

FeatureReference = dict[str, Any]
ReferenceStatistics = dict[str, FeatureReference]


def feature_reference(values: pd.Series | np.ndarray) -> FeatureReference:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("feature_reference expects a non-empty one-dimensional array")
    if np.isnan(array).any():
        raise ValueError("feature_reference does not accept NaN values")

    edges = np.unique(np.quantile(array, np.linspace(0.0, 1.0, N_BINS + 1)))
    if edges.size < 2:
        edges = np.array([edges[0], edges[0] + 1.0])
    counts, _ = np.histogram(array, bins=edges)

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "quantiles": [float(q) for q in np.quantile(array, QUANTILE_GRID)],
        "bin_edges": [float(e) for e in edges],
        "bin_fractions": [float(c) for c in counts / counts.sum()],
    }


def compute_reference_statistics(frame: pd.DataFrame) -> ReferenceStatistics:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing feature columns: {missing}")
    return {column: feature_reference(frame[column]) for column in FEATURE_COLUMNS}
