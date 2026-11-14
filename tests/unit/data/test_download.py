import pandas as pd
import sklearn.datasets

from housing_mlops.data.download import fetch_from_sklearn, load_raw_dataset, raw_dataset_path
from housing_mlops.data.schema import ALL_COLUMNS


def test_first_load_fetches_and_caches(tmp_path, fetcher, housing_frame):
    frame = load_raw_dataset(tmp_path, fetcher=fetcher)
    assert fetcher.calls == 1
    assert raw_dataset_path(tmp_path).exists()
    pd.testing.assert_frame_equal(frame, housing_frame)


def test_second_load_uses_cache_without_fetching(tmp_path, fetcher, housing_frame):
    load_raw_dataset(tmp_path, fetcher=fetcher)
    cached = load_raw_dataset(tmp_path, fetcher=fetcher)
    assert fetcher.calls == 1
    pd.testing.assert_frame_equal(cached, housing_frame)


def test_force_refetches_even_when_cached(tmp_path, fetcher):
    load_raw_dataset(tmp_path, fetcher=fetcher)
    load_raw_dataset(tmp_path, fetcher=fetcher, force=True)
    assert fetcher.calls == 2


def test_cache_directory_passed_to_fetcher_lives_under_raw(tmp_path, housing_frame):
    seen = []

    def spy(cache_dir):
        seen.append(cache_dir)
        return housing_frame

    load_raw_dataset(tmp_path, fetcher=spy)
    assert seen[0].parent == raw_dataset_path(tmp_path).parent


def test_sklearn_fetch_uses_cache_dir_and_keeps_only_known_columns(
    tmp_path, housing_frame, monkeypatch
):
    seen = {}

    class FakeBunch:
        frame = housing_frame.assign(Unexpected=0.0)[["Unexpected", *reversed(ALL_COLUMNS)]]

    def fake_fetch(*, data_home, as_frame):
        seen["data_home"] = data_home
        seen["as_frame"] = as_frame
        return FakeBunch()

    monkeypatch.setattr(sklearn.datasets, "fetch_california_housing", fake_fetch)
    cache_dir = tmp_path / "raw" / "sklearn_cache"

    frame = fetch_from_sklearn(cache_dir)

    assert seen == {"data_home": str(cache_dir), "as_frame": True}
    assert cache_dir.is_dir()
    assert list(frame.columns) == ALL_COLUMNS
    pd.testing.assert_frame_equal(frame, housing_frame)
