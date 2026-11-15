from pathlib import Path

import pytest

from housing_mlops.data import processed_dir, read_splits
from housing_mlops.training import MODEL_NAMES, train_model

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEED = 42


@pytest.fixture(scope="session")
def real_split():
    processed = processed_dir(DATA_DIR)
    if not processed.exists():
        raise AssertionError(
            f"Processed dataset not found at {processed}. Run `python scripts/download_data.py`."
        )
    return read_splits(processed)


@pytest.fixture(scope="session")
def trained(real_split, tmp_path_factory):
    root = tmp_path_factory.mktemp("model-quality")
    return {
        name: train_model(
            real_split,
            model_name=name,
            seed=SEED,
            model_dir=root / "models",
            tracking_uri=str(root / "mlruns"),
            experiment_name="model-quality",
        )
        for name in MODEL_NAMES
    }
