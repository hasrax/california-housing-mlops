import json
import re
from datetime import UTC, datetime

from housing_mlops.training.metadata import ModelMetadata, new_version


def sample_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_name="baseline",
        version="20260101-000000-abc123",
        created_at="2026-01-01T00:00:00+00:00",
        seed=42,
        params={"alpha": 1.0},
        feature_columns=["MedInc"],
        target_column="MedHouseVal",
        training_data_hash="deadbeef",
        training_rows=10,
        metrics={"validation": {"rmse": 0.5}, "test": {"rmse": 0.6}},
        reference_statistics={"MedInc": {"count": 10, "mean": 3.0}},
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
