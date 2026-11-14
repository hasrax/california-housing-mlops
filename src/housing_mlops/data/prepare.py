import logging
from pathlib import Path

from housing_mlops.data.download import Fetcher, fetch_from_sklearn, load_raw_dataset
from housing_mlops.data.hashing import dataframe_sha256
from housing_mlops.data.schema import validate_housing_frame
from housing_mlops.data.split import DatasetSplit, split_dataset, write_splits

log = logging.getLogger(__name__)


def processed_dir(data_dir: Path) -> Path:
    return data_dir / "processed"


def prepare_dataset(
    data_dir: Path,
    *,
    seed: int,
    force: bool = False,
    fetcher: Fetcher = fetch_from_sklearn,
) -> DatasetSplit:
    raw = load_raw_dataset(data_dir, force=force, fetcher=fetcher)
    validated = validate_housing_frame(raw)
    log.info("Validated %d rows, data hash %s", len(validated), dataframe_sha256(validated))

    split = split_dataset(validated, seed=seed)
    paths = write_splits(split, processed_dir(data_dir))
    for name, size in split.sizes().items():
        log.info("%s: %d rows -> %s", name, size, paths[name])
    return split
