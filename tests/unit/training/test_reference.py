import numpy as np
import pytest

from housing_mlops.data.schema import FEATURE_COLUMNS
from housing_mlops.training.reference import (
    N_BINS,
    QUANTILE_GRID,
    compute_reference_statistics,
    feature_reference,
)


def test_reference_covers_every_feature(housing_frame):
    stats = compute_reference_statistics(housing_frame)
    assert list(stats) == FEATURE_COLUMNS


def test_feature_reference_fields(housing_frame):
    values = housing_frame["MedInc"]
    ref = feature_reference(values)
    assert ref["count"] == len(values)
    assert ref["mean"] == pytest.approx(values.mean())
    assert ref["std"] == pytest.approx(values.std(ddof=0))
    assert ref["min"] == pytest.approx(values.min())
    assert ref["max"] == pytest.approx(values.max())


def test_quantiles_are_monotonic_and_span_min_to_max(housing_frame):
    ref = feature_reference(housing_frame["AveRooms"])
    quantiles = np.array(ref["quantiles"])
    assert len(quantiles) == len(QUANTILE_GRID) == 101
    assert np.all(np.diff(quantiles) >= 0)
    assert quantiles[0] == pytest.approx(ref["min"])
    assert quantiles[-1] == pytest.approx(ref["max"])


def test_bins_partition_the_data(housing_frame):
    ref = feature_reference(housing_frame["Population"])
    edges = np.array(ref["bin_edges"])
    fractions = np.array(ref["bin_fractions"])
    assert len(edges) == N_BINS + 1
    assert len(fractions) == N_BINS
    assert np.all(np.diff(edges) > 0)
    assert fractions.sum() == pytest.approx(1.0)
    assert np.all(fractions > 0)


def test_constant_feature_gets_a_single_bin():
    ref = feature_reference(np.full(50, 3.0))
    assert ref["bin_edges"] == [3.0, 4.0]
    assert ref["bin_fractions"] == [1.0]
    assert ref["std"] == 0.0


def test_discrete_feature_with_duplicate_quantiles_has_unique_edges():
    ref = feature_reference(np.repeat([1.0, 2.0, 3.0], [80, 15, 5]))
    assert ref["bin_edges"] == sorted(set(ref["bin_edges"]))
    assert sum(ref["bin_fractions"]) == pytest.approx(1.0)


def test_nan_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        feature_reference(np.array([1.0, np.nan]))


def test_empty_or_2d_input_is_rejected():
    with pytest.raises(ValueError):
        feature_reference(np.array([]))
    with pytest.raises(ValueError):
        feature_reference(np.ones((3, 2)))


def test_missing_feature_column_is_rejected(housing_frame):
    with pytest.raises(ValueError, match="missing feature columns"):
        compute_reference_statistics(housing_frame.drop(columns=["Latitude"]))


def test_statistics_are_json_native_types(housing_frame):
    ref = feature_reference(housing_frame["MedInc"])
    assert isinstance(ref["count"], int)
    assert all(isinstance(ref[k], float) for k in ("mean", "std", "min", "max"))
    assert all(isinstance(q, float) for q in ref["quantiles"])
