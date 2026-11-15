import numpy as np
import pytest

from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.training.pipelines import BASELINE, build_pipeline
from housing_mlops.training.registry import (
    current_dir,
    list_versions,
    load_current,
    load_model,
    save_model,
    set_current,
    version_dir,
)
from tests.unit.training.test_metadata import sample_metadata


@pytest.fixture
def fitted_pipeline(housing_frame):
    pipeline = build_pipeline(BASELINE, seed=0)
    return pipeline.fit(housing_frame[FEATURE_COLUMNS], housing_frame[TARGET_COLUMN])


def test_save_then_load_roundtrip(tmp_path, fitted_pipeline, housing_frame):
    metadata = sample_metadata()
    target = save_model(fitted_pipeline, metadata, version_dir(tmp_path, "baseline", "v1"))
    loaded = load_model(target)
    assert loaded.metadata == metadata
    assert loaded.source_dir == target
    np.testing.assert_allclose(
        loaded.pipeline.predict(housing_frame[FEATURE_COLUMNS]),
        fitted_pipeline.predict(housing_frame[FEATURE_COLUMNS]),
    )


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError):
        load_current(tmp_path)


def test_set_current_replaces_previous(tmp_path, fitted_pipeline):
    first = save_model(fitted_pipeline, sample_metadata(), version_dir(tmp_path, "baseline", "v1"))
    second_metadata = sample_metadata()
    second_metadata.version = "v2"
    second = save_model(fitted_pipeline, second_metadata, version_dir(tmp_path, "baseline", "v2"))

    set_current(first, tmp_path)
    assert load_current(tmp_path).metadata.version == sample_metadata().version
    set_current(second, tmp_path)
    assert load_current(tmp_path).metadata.version == "v2"
    assert current_dir(tmp_path).is_dir()
    assert not (tmp_path / ".current.staging").exists()


def test_list_versions_sorted_and_ignores_incomplete_dirs(tmp_path, fitted_pipeline):
    for version in ("20260102-000000-bbb", "20260101-000000-aaa"):
        save_model(fitted_pipeline, sample_metadata(), version_dir(tmp_path, "baseline", version))
    (tmp_path / "baseline" / "broken").mkdir()
    assert list_versions(tmp_path, "baseline") == [
        "20260101-000000-aaa",
        "20260102-000000-bbb",
    ]
    assert list_versions(tmp_path, "unknown") == []
