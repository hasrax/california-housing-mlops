import numpy as np
import pytest

from housing_mlops.data import FEATURE_COLUMNS, dataframe_sha256
from housing_mlops.training import BASELINE, GRADIENT_BOOSTING, predict_frame, train_model

pytestmark = pytest.mark.dataset

THRESHOLDS = {
    BASELINE: {"rmse_max": 0.80, "r2_min": 0.55},
    GRADIENT_BOOSTING: {"rmse_max": 0.55, "r2_min": 0.75},
}


@pytest.mark.parametrize("name", list(THRESHOLDS))
def test_test_set_metrics_meet_minimum_quality(trained, name):
    metrics = trained[name].metadata.metrics["test"]
    assert metrics["rmse"] <= THRESHOLDS[name]["rmse_max"], metrics
    assert metrics["r2"] >= THRESHOLDS[name]["r2_min"], metrics


def test_gradient_boosting_beats_baseline(trained):
    for split_name in ("validation", "test"):
        gb = trained[GRADIENT_BOOSTING].metadata.metrics[split_name]["rmse"]
        base = trained[BASELINE].metadata.metrics[split_name]["rmse"]
        assert gb < base, (split_name, gb, base)


def test_validation_and_test_metrics_are_close(trained):
    for name, result in trained.items():
        validation = result.metadata.metrics["validation"]["rmse"]
        test = result.metadata.metrics["test"]["rmse"]
        assert abs(validation - test) < 0.1, (name, validation, test)


@pytest.mark.parametrize("name", [BASELINE, GRADIENT_BOOSTING])
def test_higher_income_never_lowers_prediction(trained, real_split, name):
    sample = real_split.test.head(1000)
    richer = sample.assign(MedInc=sample["MedInc"] + 1.0)
    before = predict_frame(trained[name].pipeline, sample)
    after = predict_frame(trained[name].pipeline, richer)
    assert np.all(after >= before - 1e-9)
    assert after.mean() > before.mean()


def test_predictions_are_invariant_to_column_order_and_extra_columns(trained, real_split):
    pipeline = trained[GRADIENT_BOOSTING].pipeline
    sample = real_split.test.head(500)
    expected = predict_frame(pipeline, sample)
    shuffled = sample[list(reversed(sample.columns))].assign(extra="x")
    np.testing.assert_array_equal(predict_frame(pipeline, shuffled), expected)


def test_batch_and_single_row_predictions_agree(trained, real_split):
    pipeline = trained[GRADIENT_BOOSTING].pipeline
    sample = real_split.test.head(50)
    batch = predict_frame(pipeline, sample)
    singles = np.array([predict_frame(pipeline, sample.iloc[[i]])[0] for i in range(len(sample))])
    np.testing.assert_allclose(singles, batch, rtol=1e-12, atol=1e-12)


def test_predictions_are_finite_and_mostly_plausible(trained, real_split):
    for name, result in trained.items():
        predictions = predict_frame(result.pipeline, real_split.test)
        assert np.isfinite(predictions).all(), name
        low, median, high = np.quantile(predictions, [0.01, 0.5, 0.99])
        assert low > 0.0 and high < 6.0, (name, low, high)
        assert 1.0 < median < 3.5, (name, median)


def test_served_model_never_predicts_outside_the_target_range(trained, real_split):
    predictions = predict_frame(trained[GRADIENT_BOOSTING].pipeline, real_split.test)
    assert predictions.min() > 0.0
    assert predictions.max() < 10.0


def test_retraining_with_same_seed_is_reproducible(trained, real_split, tmp_path):
    again = train_model(
        real_split,
        model_name=GRADIENT_BOOSTING,
        seed=42,
        model_dir=tmp_path / "models",
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="model-quality",
    )
    assert again.metadata.metrics == trained[GRADIENT_BOOSTING].metadata.metrics
    np.testing.assert_array_equal(
        predict_frame(again.pipeline, real_split.test.head(200)),
        predict_frame(trained[GRADIENT_BOOSTING].pipeline, real_split.test.head(200)),
    )


def test_metadata_reference_matches_training_data(trained, real_split):
    metadata = trained[GRADIENT_BOOSTING].metadata
    assert metadata.training_rows == len(real_split.train) == 14448
    assert metadata.training_data_hash == dataframe_sha256(real_split.train)
    for column in FEATURE_COLUMNS:
        ref = metadata.reference_statistics[column]
        assert ref["count"] == 14448
        assert ref["mean"] == pytest.approx(real_split.train[column].mean())
