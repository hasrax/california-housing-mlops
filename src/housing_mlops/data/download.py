import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from housing_mlops.data.schema import ALL_COLUMNS

RAW_FILENAME = "california_housing.parquet"
SKLEARN_CACHE_DIRNAME = "sklearn_cache"

Fetcher = Callable[[Path], pd.DataFrame]

log = logging.getLogger(__name__)


def raw_dataset_path(data_dir: Path) -> Path:
    return data_dir / "raw" / RAW_FILENAME


def fetch_from_sklearn(cache_dir: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    cache_dir.mkdir(parents=True, exist_ok=True)
    bunch = fetch_california_housing(data_home=str(cache_dir), as_frame=True)
    return bunch.frame[ALL_COLUMNS].copy()


def load_raw_dataset(
    data_dir: Path,
    *,
    force: bool = False,
    fetcher: Fetcher = fetch_from_sklearn,
) -> pd.DataFrame:
    raw_path = raw_dataset_path(data_dir)
    if raw_path.exists() and not force:
        log.info("Loading cached dataset from %s", raw_path)
        return pd.read_parquet(raw_path)

    log.info("Fetching dataset (force=%s)", force)
    frame = fetcher(raw_path.parent / SKLEARN_CACHE_DIRNAME)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(raw_path, index=False)
    log.info("Cached %d rows to %s", len(frame), raw_path)
    return frame
