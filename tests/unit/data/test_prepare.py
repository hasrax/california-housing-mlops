import pytest

from housing_mlops.data.prepare import prepare_dataset, processed_dir
from housing_mlops.data.schema import SchemaErrors
from housing_mlops.data.split import SPLIT_NAMES, read_splits


def test_prepare_writes_raw_and_processed_files(tmp_path, fetcher, housing_frame):
    split = prepare_dataset(tmp_path, seed=42, fetcher=fetcher)
    assert sum(split.sizes().values()) == len(housing_frame)
    for name in SPLIT_NAMES:
        assert (processed_dir(tmp_path) / f"{name}.parquet").exists()
    assert read_splits(processed_dir(tmp_path)).sizes() == split.sizes()


def test_prepare_is_reproducible_across_runs(tmp_path, fetcher):
    first = prepare_dataset(tmp_path, seed=42, fetcher=fetcher)
    second = prepare_dataset(tmp_path, seed=42, fetcher=fetcher)
    assert list(first.test.index) == list(second.test.index)
    assert fetcher.calls == 1


def test_prepare_rejects_invalid_data_before_splitting(tmp_path, housing_frame):
    broken = housing_frame.copy()
    broken.loc[0, "MedInc"] = -5.0

    with pytest.raises(SchemaErrors):
        prepare_dataset(tmp_path, seed=42, fetcher=lambda cache_dir: broken)
    assert not processed_dir(tmp_path).exists()
