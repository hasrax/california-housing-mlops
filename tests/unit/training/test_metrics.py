import math

import numpy as np
import pytest

from housing_mlops.training.metrics import (
    LOWER_IS_BETTER,
    METRIC_NAMES,
    PRIMARY_METRIC,
    is_better,
    regression_metrics,
)


def test_perfect_prediction():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_known_values():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert metrics["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["r2"] == pytest.approx(1 - 4 / 2)


def test_constant_prediction_has_non_positive_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert regression_metrics(y, np.full_like(y, y.mean()))["r2"] == pytest.approx(0.0)
    assert regression_metrics(y, np.full_like(y, 10.0))["r2"] < 0


def test_metric_names_match_output():
    metrics = regression_metrics([1.0, 2.0], [1.0, 2.0])
    assert set(metrics) == set(METRIC_NAMES) == set(LOWER_IS_BETTER)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_metrics([1.0, 2.0], [1.0])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        regression_metrics([], [])


def test_is_better_respects_direction():
    assert PRIMARY_METRIC == "rmse"
    assert is_better(0.4, 0.5, "rmse")
    assert not is_better(0.5, 0.5, "rmse")
    assert is_better(0.9, 0.8, "r2")
    assert not is_better(0.7, 0.8, "r2")
