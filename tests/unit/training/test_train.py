import re

import mlflow
import numpy as np
import pytest

from housing_mlops.data.hashing import dataframe_sha256
from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.training.metadata import METADATA_FILENAME, MODEL_FILENAME
from housing_mlops.training.metrics import METRIC_NAMES
from housing_mlops.training.pipelines import BASELINE, GRADIENT_BOOSTING
from housing_mlops.training.predict import predict_frame
from housing_mlops.training.registry import load_model
from housing_mlops.training.train import (
    EVALUATION_SPLITS,
    evaluate,
    resolve_tracking_uri,
    train_model,
)

EXPERIMENT = "unit-tests"


@pytest.fixture
def baseline_result(housing_split, tmp_path, tracking_uri):
    return train_model(
        housing_split,
        model_name=BASELINE,
        seed=42,
        model_dir=tmp_path / "models",
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT,
    )


def test_artifacts_are_written(baseline_result, tmp_path):
    artifact_dir = baseline_result.artifact_dir
    assert artifact_dir.parent.parent == tmp_path / "models"
    assert artifact_dir.parent.name == BASELINE
    assert (artifact_dir / MODEL_FILENAME).exists()
    assert (artifact_dir / METADATA_FILENAME).exists()


def test_metadata_describes_the_training(baseline_result, housing_split):
    metadata = baseline_result.metadata
    assert metadata.model_name == BASELINE
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", metadata.version)
    assert metadata.seed == 42
    assert metadata.params == {"alpha": 1.0}
    assert metadata.feature_columns == FEATURE_COLUMNS
    assert metadata.target_column == TARGET_COLUMN
    assert metadata.training_rows == len(housing_split.train)
    assert metadata.training_data_hash == dataframe_sha256(housing_split.train)
    assert set(metadata.metrics) == set(EVALUATION_SPLITS)
    assert all(set(m) == set(METRIC_NAMES) for m in metadata.metrics.values())
    assert list(metadata.reference_statistics) == FEATURE_COLUMNS
    assert metadata.reference_statistics["MedInc"]["count"] == len(housing_split.train)
    assert metadata.mlflow_run_id
    assert metadata.mlflow_experiment == EXPERIMENT
    assert "scikit-learn" in metadata.library_versions


def test_mlflow_run_records_params_metrics_and_artifacts(baseline_result, tracking_uri):
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    assert len(runs) == 1
    run = runs.iloc[0]
    metadata = baseline_result.metadata
    assert run["run_id"] == metadata.mlflow_run_id
    assert run["params.model_name"] == BASELINE
    assert run["params.alpha"] == "1.0"
    assert run["metrics.validation_rmse"] == pytest.approx(metadata.metrics["validation"]["rmse"])
    assert run["metrics.test_r2"] == pytest.approx(metadata.metrics["test"]["r2"])
    assert run["tags.training_data_hash"] == metadata.training_data_hash

    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(metadata.mlflow_run_id)}
    assert {METADATA_FILENAME, MODEL_FILENAME} <= artifacts


def test_saved_model_reproduces_in_memory_predictions(baseline_result, housing_split):
    loaded = load_model(baseline_result.artifact_dir)
    assert loaded.metadata == baseline_result.metadata
    np.testing.assert_allclose(
        predict_frame(loaded.pipeline, housing_split.test),
        predict_frame(baseline_result.pipeline, housing_split.test),
    )


def test_metadata_metrics_match_evaluate(baseline_result, housing_split):
    assert evaluate(baseline_result.pipeline, housing_split) == baseline_result.metadata.metrics


def test_gradient_boosting_trains_with_param_override(housing_split, tmp_path, tracking_uri):
    result = train_model(
        housing_split,
        model_name=GRADIENT_BOOSTING,
        seed=1,
        model_dir=tmp_path / "models",
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT,
        params={"max_iter": 15},
    )
    assert result.metadata.params["max_iter"] == 15
    assert result.pipeline.named_steps["model"].max_iter == 15
    assert result.metadata.metrics["test"]["rmse"] > 0


def test_same_seed_gives_identical_models(housing_split, tmp_path, tracking_uri):
    kwargs = dict(
        model_name=GRADIENT_BOOSTING,
        seed=7,
        model_dir=tmp_path / "models",
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT,
        params={"max_iter": 15},
    )
    first = train_model(housing_split, **kwargs)
    second = train_model(housing_split, **kwargs)
    assert first.metadata.version != second.metadata.version
    assert first.metadata.metrics == second.metadata.metrics
    np.testing.assert_array_equal(
        predict_frame(first.pipeline, housing_split.test),
        predict_frame(second.pipeline, housing_split.test),
    )


def test_unknown_model_fails_before_any_run_is_created(housing_split, tmp_path, tracking_uri):
    with pytest.raises(ValueError, match="Unknown model"):
        train_model(
            housing_split,
            model_name="nope",
            seed=0,
            model_dir=tmp_path / "models",
            tracking_uri=tracking_uri,
            experiment_name=EXPERIMENT,
        )
    assert not (tmp_path / "mlruns").exists()
    assert not (tmp_path / "models").exists()


def test_resolve_tracking_uri(tmp_path):
    assert resolve_tracking_uri("http://mlflow:5000") == "http://mlflow:5000"
    resolved = resolve_tracking_uri(str(tmp_path / "mlruns"))
    assert resolved.startswith("sqlite:///")
    assert resolved.endswith("/mlruns/mlflow.db")
    assert (tmp_path / "mlruns").is_dir()


def test_artifacts_are_stored_next_to_the_tracking_database(baseline_result, tmp_path):
    client = mlflow.MlflowClient(resolve_tracking_uri(str(tmp_path / "mlruns")))
    run = client.get_run(baseline_result.metadata.mlflow_run_id)
    assert run.info.artifact_uri.startswith((tmp_path / "mlruns" / "artifacts").as_uri())
    assert (tmp_path / "mlruns" / "mlflow.db").exists()
