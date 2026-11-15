import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from housing_mlops.api.db import Base, PredictionLog, make_engine
from housing_mlops.api.main import create_app
from housing_mlops.config import Settings
from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.training import BASELINE, build_pipeline, save_model, set_current, version_dir
from tests.unit.api.conftest import VALID_FEATURES
from tests.unit.training.test_metadata import sample_metadata

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url or not url.startswith("postgresql"):
        raise AssertionError(
            "Integration tests need DATABASE_URL pointing at PostgreSQL, "
            "for example postgresql+psycopg://housing:housing@localhost:5432/housing"
        )
    return url


@pytest.fixture
def clean_database(database_url):
    engine = make_engine(database_url)
    Base.metadata.drop_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def client(tmp_path, housing_frame, database_url, clean_database):
    pipeline = build_pipeline(BASELINE, seed=0)
    pipeline.fit(housing_frame[FEATURE_COLUMNS], housing_frame[TARGET_COLUMN])
    root = tmp_path / "models"
    set_current(
        save_model(pipeline, sample_metadata(), version_dir(root, BASELINE, "v1")),
        root,
    )
    settings = Settings(_env_file=None, database_url=database_url, model_dir=root)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_reports_database_ok(client):
    assert client.get("/health").json()["database"] == "ok"


def test_predictions_are_persisted_in_postgres(client, clean_database):
    responses = [client.post("/predict", json=VALID_FEATURES) for _ in range(25)]
    assert all(r.status_code == 200 for r in responses)

    with clean_database.connect() as connection:
        count = connection.scalar(select(func.count()).select_from(PredictionLog))
        distinct = connection.scalar(select(func.count(func.distinct(PredictionLog.request_id))))
        assert count == distinct == 25

        row = connection.execute(select(PredictionLog).limit(1)).mappings().one()
        assert row["created_at"].tzinfo is not None
        assert row["model_version"] == sample_metadata().version
        assert row["MedInc"] == pytest.approx(VALID_FEATURES["MedInc"])


def test_created_at_is_timestamptz_and_request_id_is_unique(client, clean_database):
    with clean_database.connect() as connection:
        data_type = connection.scalar(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'predictions' AND column_name = 'created_at'"
            )
        )
        assert data_type == "timestamp with time zone"
        indexes = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE tablename = 'predictions'")
        ).scalars()
        assert any("UNIQUE" in d and "request_id" in d for d in indexes)


def test_invalid_payload_leaves_no_row(client, clean_database):
    assert client.post("/predict", json={**VALID_FEATURES, "MedInc": -1}).status_code == 422
    with clean_database.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(PredictionLog)) == 0
