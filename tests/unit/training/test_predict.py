import numpy as np
import pytest

from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.training.pipelines import GRADIENT_BOOSTING, build_pipeline
from housing_mlops.training.predict import predict_frame


@pytest.fixture
def fitted(housing_frame):
    pipeline = build_pipeline(GRADIENT_BOOSTING, seed=0, max_iter=20)
    return pipeline.fit(housing_frame[FEATURE_COLUMNS], housing_frame[TARGET_COLUMN])


def test_returns_float_vector(fitted, housing_frame):
    predictions = predict_frame(fitted, housing_frame)
    assert predictions.shape == (len(housing_frame),)
    assert predictions.dtype == np.float64


def test_column_order_does_not_matter(fitted, housing_frame):
    shuffled = housing_frame[list(reversed(housing_frame.columns))]
    np.testing.assert_array_equal(
        predict_frame(fitted, shuffled), predict_frame(fitted, housing_frame)
    )


def test_extra_columns_are_ignored(fitted, housing_frame):
    with_extra = housing_frame.assign(noise=1.0, **{TARGET_COLUMN: 0.0})
    np.testing.assert_array_equal(
        predict_frame(fitted, with_extra), predict_frame(fitted, housing_frame)
    )


def test_missing_feature_raises(fitted, housing_frame):
    with pytest.raises(ValueError, match="missing feature columns"):
        predict_frame(fitted, housing_frame.drop(columns=["AveOccup"]))
