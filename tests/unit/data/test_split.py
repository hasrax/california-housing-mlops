import pandas as pd
import pytest

from housing_mlops.data.split import read_splits, split_dataset, write_splits


def test_split_sizes_match_fractions(housing_frame):
    split = split_dataset(housing_frame, seed=42)
    sizes = split.sizes()
    assert sum(sizes.values()) == len(housing_frame)
    assert sizes["test"] == pytest.approx(0.2 * len(housing_frame), abs=1)
    assert sizes["validation"] == pytest.approx(0.1 * len(housing_frame), abs=1)
    assert sizes["train"] == pytest.approx(0.7 * len(housing_frame), abs=2)


def test_splits_are_disjoint_and_cover_all_rows(housing_frame):
    split = split_dataset(housing_frame, seed=42)
    train, val, test = set(split.train.index), set(split.validation.index), set(split.test.index)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(housing_frame.index)


def test_same_seed_gives_identical_split(housing_frame):
    first = split_dataset(housing_frame, seed=7)
    second = split_dataset(housing_frame, seed=7)
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_different_seed_gives_different_split(housing_frame):
    first = split_dataset(housing_frame, seed=1)
    second = split_dataset(housing_frame, seed=2)
    assert list(first.test.index) != list(second.test.index)


def test_split_preserves_columns(housing_frame):
    split = split_dataset(housing_frame, seed=0)
    for name in ("train", "validation", "test"):
        assert list(getattr(split, name).columns) == list(housing_frame.columns)


@pytest.mark.parametrize(
    ("test_fraction", "validation_fraction"),
    [(0.0, 0.1), (1.0, 0.1), (0.2, 0.0), (0.6, 0.4), (0.9, 0.5)],
)
def test_invalid_fractions_raise(housing_frame, test_fraction, validation_fraction):
    with pytest.raises(ValueError):
        split_dataset(
            housing_frame,
            seed=0,
            test_fraction=test_fraction,
            validation_fraction=validation_fraction,
        )


def test_write_then_read_roundtrip(housing_frame, tmp_path):
    split = split_dataset(housing_frame, seed=3)
    paths = write_splits(split, tmp_path / "processed")
    assert all(path.exists() for path in paths.values())
    loaded = read_splits(tmp_path / "processed")
    pd.testing.assert_frame_equal(loaded.train, split.train)
    pd.testing.assert_frame_equal(loaded.validation, split.validation)
    pd.testing.assert_frame_equal(loaded.test, split.test)
