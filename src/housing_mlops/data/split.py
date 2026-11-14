from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in SPLIT_NAMES}


def split_dataset(
    frame: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> DatasetSplit:
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("test_fraction and validation_fraction must be strictly between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be less than 1")

    remainder, test = train_test_split(
        frame, test_size=test_fraction, random_state=seed, shuffle=True
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        remainder, test_size=relative_validation, random_state=seed, shuffle=True
    )
    return DatasetSplit(train=train, validation=validation, test=test)


def split_path(processed_dir: Path, name: str) -> Path:
    return processed_dir / f"{name}.parquet"


def write_splits(split: DatasetSplit, processed_dir: Path) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        path = split_path(processed_dir, name)
        getattr(split, name).to_parquet(path, index=True)
        paths[name] = path
    return paths


def read_splits(processed_dir: Path) -> DatasetSplit:
    frames = {name: pd.read_parquet(split_path(processed_dir, name)) for name in SPLIT_NAMES}
    return DatasetSplit(**frames)
