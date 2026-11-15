import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from housing_mlops.training.reference import ReferenceStatistics

METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.joblib"


def new_version(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:6]}"


@dataclass
class ModelMetadata:
    model_name: str
    version: str
    created_at: str
    seed: int
    params: dict[str, Any]
    feature_columns: list[str]
    target_column: str
    training_data_hash: str
    training_rows: int
    metrics: dict[str, dict[str, float]]
    reference_statistics: ReferenceStatistics
    mlflow_run_id: str | None = None
    mlflow_experiment: str | None = None
    library_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelMetadata":
        return cls(**payload)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ModelMetadata":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
