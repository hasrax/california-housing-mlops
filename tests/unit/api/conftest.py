import pytest
from fastapi.testclient import TestClient

from housing_mlops.api.main import create_app
from housing_mlops.config import Settings
from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.training import BASELINE, build_pipeline, save_model, set_current, version_dir
from tests.unit.training.test_metadata import sample_metadata

VALID_FEATURES = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


@pytest.fixture
def model_dir(tmp_path, housing_frame):
    pipeline = build_pipeline(BASELINE, seed=0)
    pipeline.fit(housing_frame[FEATURE_COLUMNS], housing_frame[TARGET_COLUMN])
    metadata = sample_metadata()
    metadata.feature_columns = list(FEATURE_COLUMNS)
    root = tmp_path / "models"
    saved = save_model(pipeline, metadata, version_dir(root, BASELINE, metadata.version))
    set_current(saved, root)
    return root


@pytest.fixture
def api_settings(tmp_path, model_dir):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        model_dir=model_dir,
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def client(api_settings):
    with TestClient(create_app(api_settings)) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        model_dir=tmp_path / "empty-models",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client
