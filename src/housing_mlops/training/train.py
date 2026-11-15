import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy
import pandas
import sklearn
from sklearn.pipeline import Pipeline

from housing_mlops.data.hashing import dataframe_sha256
from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.data.split import DatasetSplit
from housing_mlops.training.metadata import MODEL_FILENAME, ModelMetadata, new_version
from housing_mlops.training.metrics import regression_metrics
from housing_mlops.training.pipelines import DEFAULT_PARAMS, build_pipeline
from housing_mlops.training.predict import predict_frame
from housing_mlops.training.reference import compute_reference_statistics
from housing_mlops.training.registry import save_model, version_dir

log = logging.getLogger(__name__)

EVALUATION_SPLITS: tuple[str, ...] = ("validation", "test")


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    metadata: ModelMetadata
    artifact_dir: Path


TRACKING_DB_FILENAME = "mlflow.db"
ARTIFACTS_DIRNAME = "artifacts"


def resolve_tracking_uri(uri: str) -> str:
    if "://" in uri:
        return uri
    root = Path(uri).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / TRACKING_DB_FILENAME).as_posix()}"


def configure_tracking(tracking_uri: str, experiment_name: str) -> str:
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_location = None
        if "://" not in tracking_uri:
            artifact_location = (Path(tracking_uri).resolve() / ARTIFACTS_DIRNAME).as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    return mlflow.set_experiment(experiment_name).experiment_id


def library_versions() -> dict[str, str]:
    return {
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "mlflow": mlflow.__version__,
    }


def evaluate(pipeline: Pipeline, split: DatasetSplit) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name in EVALUATION_SPLITS:
        frame = getattr(split, name)
        predictions = predict_frame(pipeline, frame)
        results[name] = regression_metrics(frame[TARGET_COLUMN], predictions)
    return results


def train_model(
    split: DatasetSplit,
    *,
    model_name: str,
    seed: int,
    model_dir: Path,
    tracking_uri: str,
    experiment_name: str,
    params: dict[str, Any] | None = None,
) -> TrainingResult:
    params = {**DEFAULT_PARAMS.get(model_name, {}), **(params or {})}
    pipeline = build_pipeline(model_name, seed=seed, **params)
    version = new_version()

    configure_tracking(tracking_uri, experiment_name)

    with mlflow.start_run(run_name=f"{model_name}-{version}") as run:
        log.info("Training %s version %s (run %s)", model_name, version, run.info.run_id)
        pipeline.fit(split.train[FEATURE_COLUMNS], split.train[TARGET_COLUMN])

        metrics = evaluate(pipeline, split)
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            created_at=datetime.now(UTC).isoformat(),
            seed=seed,
            params=params,
            feature_columns=list(FEATURE_COLUMNS),
            target_column=TARGET_COLUMN,
            training_data_hash=dataframe_sha256(split.train),
            training_rows=len(split.train),
            metrics=metrics,
            reference_statistics=compute_reference_statistics(split.train),
            mlflow_run_id=run.info.run_id,
            mlflow_experiment=experiment_name,
            library_versions=library_versions(),
        )

        artifact_dir = save_model(pipeline, metadata, version_dir(model_dir, model_name, version))

        mlflow.log_params({"model_name": model_name, "seed": seed, **params})
        mlflow.log_metrics(
            {f"{split_name}_{k}": v for split_name, m in metrics.items() for k, v in m.items()}
        )
        mlflow.set_tags({"version": version, "training_data_hash": metadata.training_data_hash})
        mlflow.log_dict(metadata.to_dict(), "metadata.json")
        mlflow.log_artifact(str(artifact_dir / MODEL_FILENAME))

    for split_name, values in metrics.items():
        log.info("%s %s: %s", model_name, split_name, values)
    return TrainingResult(pipeline=pipeline, metadata=metadata, artifact_dir=artifact_dir)
