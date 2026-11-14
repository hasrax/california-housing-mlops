import numpy as np
import pandas as pd
import pytest

from housing_mlops.config import Settings
from housing_mlops.data.schema import ALL_COLUMNS, TARGET_COLUMN


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RANDOM_SEED", raising=False)
    return Settings(_env_file=None, data_dir=tmp_path / "data", model_dir=tmp_path / "models")


def make_housing_frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rooms = rng.uniform(3.0, 8.0, n_rows)
    frame = pd.DataFrame(
        {
            "MedInc": rng.uniform(0.5, 15.0, n_rows),
            "HouseAge": rng.integers(1, 52, n_rows).astype(float),
            "AveRooms": rooms,
            "AveBedrms": rooms * rng.uniform(0.1, 0.4, n_rows),
            "Population": rng.integers(3, 5000, n_rows).astype(float),
            "AveOccup": rng.uniform(1.0, 6.0, n_rows),
            "Latitude": rng.uniform(32.6, 41.9, n_rows),
            "Longitude": rng.uniform(-124.3, -114.4, n_rows),
        }
    )
    frame[TARGET_COLUMN] = np.clip(0.3 * frame["MedInc"] + rng.normal(0, 0.3, n_rows), 0.15, 5.0)
    return frame[ALL_COLUMNS]


@pytest.fixture
def housing_frame() -> pd.DataFrame:
    return make_housing_frame()


class CountingFetcher:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = 0

    def __call__(self, cache_dir):
        self.calls += 1
        return self.frame.copy()


@pytest.fixture
def fetcher(housing_frame) -> CountingFetcher:
    return CountingFetcher(housing_frame)
