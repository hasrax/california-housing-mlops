import pytest
from sqlalchemy import inspect

from housing_mlops.api.db import (
    PredictionLog,
    check_connection,
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)
from housing_mlops.data.schema import FEATURE_COLUMNS


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    yield engine
    engine.dispose()


def test_init_db_creates_predictions_table(engine):
    init_db(engine)
    columns = {c["name"] for c in inspect(engine).get_columns(PredictionLog.__tablename__)}
    assert set(FEATURE_COLUMNS) <= columns
    assert {"request_id", "created_at", "model_version", "prediction", "latency_ms"} <= columns


def test_init_db_creates_monitoring_tables(engine):
    init_db(engine)
    assert {"predictions", "drift_reports", "model_versions"} <= set(
        inspect(engine).get_table_names()
    )


def test_check_connection_true_for_working_engine(engine):
    assert check_connection(engine)


def test_check_connection_false_for_unreachable_database(tmp_path):
    broken = make_engine(f"sqlite:///{(tmp_path / 'missing' / 'x.db').as_posix()}")
    assert not check_connection(broken)
    broken.dispose()


def test_session_scope_closes_session(engine):
    generator = session_scope(make_session_factory(engine))
    session = next(generator)
    assert session.is_active
    generator.close()
    assert session.get_bind() is engine
