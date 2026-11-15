from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

import housing_mlops.dashboard.app as dashboard
from housing_mlops.api.db import (
    PredictionLog,
    init_db,
    make_engine,
    make_session_factory,
    register_model_version,
)
from housing_mlops.data.schema import FEATURE_COLUMNS
from housing_mlops.monitoring import run_drift_job
from tests.conftest import make_housing_frame
from tests.unit.training.test_metadata import sample_metadata

APP_PATH = Path(dashboard.__file__)
NOW = datetime(2026, 11, 15, 9, 0, tzinfo=UTC)


@pytest.fixture
def database_url(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'dash.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    engine = make_engine(url)
    init_db(engine)
    engine.dispose()
    return url


def run_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    return app


def test_empty_database_renders_hints_without_errors(database_url):
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "California Housing MLOps"
    assert len(app.info) == 3
    assert app.metric[0].value == "0"
    assert app.metric[1].value == "none"


def test_populated_database_renders_charts_and_tables(database_url):
    engine = make_engine(database_url)
    metadata = sample_metadata()
    frame = make_housing_frame(200, seed=3)
    with make_session_factory(engine)() as session:
        register_model_version(session, metadata)
        for offset, (_, row) in enumerate(frame.iterrows()):
            session.add(
                PredictionLog(
                    request_id=str(uuid4()),
                    created_at=NOW + timedelta(seconds=offset),
                    model_name=metadata.model_name,
                    model_version=metadata.version,
                    prediction=2.0,
                    latency_ms=1.0,
                    **{name: float(row[name]) for name in FEATURE_COLUMNS},
                )
            )
        session.commit()
        first = run_drift_job(session, metadata, min_requests=50, now=NOW + timedelta(minutes=5))
        frame["MedInc"] *= 2.0
        frame["HouseAge"] *= 0.4
        for offset, (_, row) in enumerate(frame.iterrows()):
            session.add(
                PredictionLog(
                    request_id=str(uuid4()),
                    created_at=NOW + timedelta(minutes=6, seconds=offset),
                    model_name=metadata.model_name,
                    model_version=metadata.version,
                    prediction=3.0,
                    latency_ms=1.0,
                    **{name: float(row[name]) for name in FEATURE_COLUMNS},
                )
            )
        session.commit()
        second = run_drift_job(session, metadata, min_requests=50, now=NOW + timedelta(minutes=15))
    engine.dispose()
    assert first is not None and second is not None and second.drifted

    app = run_app()
    assert not app.exception
    assert app.metric[0].value == "400"
    assert app.metric[1].value == metadata.version
    assert app.metric[2].value == "2"
    assert app.metric[3].value == "drift detected"
    assert len(app.dataframe) == 2
    assert len(app.multiselect) == 1
    charts = app.get("plotly_chart")
    assert len(charts) == 3
