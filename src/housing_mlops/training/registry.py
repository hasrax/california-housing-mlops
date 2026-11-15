import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from housing_mlops.training.metadata import METADATA_FILENAME, MODEL_FILENAME, ModelMetadata

CURRENT_DIRNAME = "current"


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    metadata: ModelMetadata
    source_dir: Path


def version_dir(model_dir: Path, model_name: str, version: str) -> Path:
    return model_dir / model_name / version


def current_dir(model_dir: Path) -> Path:
    return model_dir / CURRENT_DIRNAME


def save_model(pipeline: Pipeline, metadata: ModelMetadata, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target_dir / MODEL_FILENAME)
    metadata.save(target_dir / METADATA_FILENAME)
    return target_dir


def load_model(source_dir: Path) -> LoadedModel:
    model_path = source_dir / MODEL_FILENAME
    metadata_path = source_dir / METADATA_FILENAME
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"No model artifacts in {source_dir}")
    return LoadedModel(
        pipeline=joblib.load(model_path),
        metadata=ModelMetadata.load(metadata_path),
        source_dir=source_dir,
    )


def set_current(source_dir: Path, model_dir: Path) -> Path:
    target = current_dir(model_dir)
    staging = model_dir / f".{CURRENT_DIRNAME}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source_dir, staging)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    return target


def load_current(model_dir: Path) -> LoadedModel:
    return load_model(current_dir(model_dir))


def list_versions(model_dir: Path, model_name: str) -> list[str]:
    root = model_dir / model_name
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / METADATA_FILENAME).exists())
