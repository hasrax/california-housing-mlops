from pathlib import Path

import pytest

from housing_mlops.data import (
    ALL_COLUMNS,
    load_raw_dataset,
    raw_dataset_path,
    split_dataset,
    validate_housing_frame,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

pytestmark = pytest.mark.dataset


def refuse_to_fetch(cache_dir):
    raise AssertionError(
        f"Cached dataset not found at {raw_dataset_path(DATA_DIR)}. "
        "Run `python scripts/download_data.py` first."
    )


@pytest.fixture(scope="module")
def real_frame():
    return load_raw_dataset(DATA_DIR, fetcher=refuse_to_fetch)


def test_real_dataset_has_expected_shape(real_frame):
    assert real_frame.shape == (20640, 9)
    assert list(real_frame.columns) == ALL_COLUMNS


def test_real_dataset_passes_schema(real_frame):
    validate_housing_frame(real_frame)


def test_real_dataset_split_sizes_with_default_seed(real_frame):
    split = split_dataset(real_frame, seed=42)
    assert split.sizes() == {"train": 14448, "validation": 2064, "test": 4128}
