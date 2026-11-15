from sqlalchemy import select

from housing_mlops.api.db import (
    ModelVersion,
    init_db,
    make_engine,
    make_session_factory,
    register_model_version,
)
from tests.unit.training.test_metadata import sample_metadata


def versions(session):
    return session.scalars(select(ModelVersion).order_by(ModelVersion.id)).all()


def test_startup_registers_the_served_model_as_current(client):
    with client.app.state.session_factory() as session:
        rows = versions(session)
    assert len(rows) == 1
    row = rows[0]
    metadata = sample_metadata()
    assert row.version == metadata.version
    assert row.model_name == metadata.model_name
    assert row.test_rmse == metadata.metrics["test"]["rmse"]
    assert row.is_current is True
    assert row.source == "startup"
    assert row.registered_at is not None


def test_register_is_idempotent_and_switches_current(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'v.db').as_posix()}")
    init_db(engine)
    first = sample_metadata()
    second = sample_metadata()
    second.version = "v2"
    with make_session_factory(engine)() as session:
        register_model_version(session, first)
        register_model_version(session, first)
        assert [r.version for r in versions(session)] == [first.version]

        register_model_version(session, second, source="promotion")
        rows = {r.version: r for r in versions(session)}
        assert rows[first.version].is_current is False
        assert rows["v2"].is_current is True
        assert rows["v2"].source == "promotion"

        register_model_version(session, first)
        rows = {r.version: r for r in versions(session)}
        assert rows[first.version].is_current is True
        assert rows["v2"].is_current is False
    engine.dispose()
