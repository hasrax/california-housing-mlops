import json
import re
from datetime import UTC, datetime

from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.training.metadata import ModelMetadata, new_version
from housing_mlops.training.reference import compute_reference_statistics
from tests.conftest import make_housing_frame


def sample_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_name="baseline",
        version="20260101-000000-abc123",
        created_at="2026-01-01T00:00:00+00:00",
        seed=42,
        params={"alpha": 1.0},
        feature_columns=list(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
        training_data_hash="deadbeef",
        training_rows=200,
        metrics={
            "validation": {"rmse": 0.5, "mae": 0.4, "r2": 0.6},
            "test": {"rmse": 0.6, "mae": 0.45, "r2": 0.55},
        },
        reference_statistics=compute_reference_statistics(make_housing_frame()),
    )


def test_version_format_and_uniqueness():
    fixed = datetime(2026, 11, 14, 12, 30, 15, tzinfo=UTC)
    first, second = new_version(fixed), new_version(fixed)
    assert re.fullmatch(r"20261114-123015-[0-9a-f]{6}", first)
    assert first != second


def test_dict_roundtrip():
    metadata = sample_metadata()
    assert ModelMetadata.from_dict(metadata.to_dict()) == metadata


def test_save_and_load_roundtrip(tmp_path):
    metadata = sample_metadata()
    path = metadata.save(tmp_path / "nested" / "metadata.json")
    assert path.exists()
    assert ModelMetadata.load(path) == metadata


def test_saved_file_is_sorted_readable_json(tmp_path):
    path = sample_metadata().save(tmp_path / "metadata.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload) == sorted(payload)
    assert payload["mlflow_run_id"] is None
    assert payload["library_versions"] == {}
