from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from housing_mlops.api.db import (
    DriftReport,
    PredictionLog,
    init_db,
    make_engine,
    make_session_factory,
)
from housing_mlops.data.schema import FEATURE_COLUMNS
from housing_mlops.monitoring.job import latest_report, recent_requests, run_drift_job
from housing_mlops.training.reference import compute_reference_statistics
from tests.conftest import make_housing_frame
from tests.unit.training.test_metadata import sample_metadata

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'job.db').as_posix()}")
    init_db(engine)
    with make_session_factory(engine)() as session:
        yield session
    engine.dispose()


@pytest.fixture
def metadata():
    item = sample_metadata()
    item.reference_statistics = compute_reference_statistics(make_housing_frame(5000, seed=1))
    item.feature_columns = list(FEATURE_COLUMNS)
    return item


def insert_requests(session, frame, version, *, start, step=timedelta(seconds=1)):
    for offset, (_, row) in enumerate(frame.iterrows()):
        session.add(
            PredictionLog(
                request_id=str(uuid4()),
                created_at=start + offset * step,
                model_name="baseline",
                model_version=version,
                prediction=1.0,
                latency_ms=1.0,
                **{name: float(row[name]) for name in FEATURE_COLUMNS},
            )
        )
    session.commit()


def test_too_few_requests_stores_nothing(session, metadata):
    insert_requests(session, make_housing_frame(30, seed=2), metadata.version, start=NOW)
    assert run_drift_job(session, metadata, min_requests=50, now=NOW + timedelta(minutes=5)) is None
    assert session.query(DriftReport).count() == 0


def test_report_is_stored_for_same_distribution(session, metadata):
    insert_requests(session, make_housing_frame(300, seed=2), metadata.version, start=NOW)
    report = run_drift_job(session, metadata, min_requests=100, now=NOW + timedelta(minutes=10))
    assert report is not None
    assert report.model_version == metadata.version
    assert report.recent_count == 300
    assert report.n_drifted == 0
    assert report.drifted is False
    assert report.window_end == NOW + timedelta(minutes=10)
    assert len(report.details["features"]) == len(FEATURE_COLUMNS)
    assert latest_report(session).id == report.id


def test_drifted_requests_produce_a_drifted_report(session, metadata):
    frame = make_housing_frame(300, seed=3)
    frame["MedInc"] *= 1.5
    frame["HouseAge"] *= 0.5
    insert_requests(session, frame, metadata.version, start=NOW)
    report = run_drift_job(session, metadata, min_requests=100, now=NOW + timedelta(minutes=10))
    assert report.drifted is True
    assert set(report.details["drifted_features"]) >= {"MedInc", "HouseAge"}


def test_only_requests_in_window_and_for_this_version_are_used(session, metadata):
    insert_requests(session, make_housing_frame(120, seed=4), metadata.version, start=NOW)
    insert_requests(session, make_housing_frame(500, seed=5), "other-version", start=NOW)
    insert_requests(
        session,
        make_housing_frame(500, seed=6),
        metadata.version,
        start=NOW - timedelta(days=3),
    )
    recent = recent_requests(
        session, metadata.version, NOW - timedelta(hours=1), NOW + timedelta(hours=1)
    )
    assert len(recent) == 120
    report = run_drift_job(
        session,
        metadata,
        window=timedelta(hours=1),
        min_requests=100,
        now=NOW + timedelta(minutes=5),
    )
    assert report.recent_count == 120


def test_latest_report_filters_by_version(session, metadata):
    insert_requests(session, make_housing_frame(150, seed=7), metadata.version, start=NOW)
    first = run_drift_job(session, metadata, min_requests=100, now=NOW + timedelta(minutes=5))
    second = run_drift_job(session, metadata, min_requests=100, now=NOW + timedelta(minutes=6))
    assert latest_report(session, metadata.version).id == second.id
    assert latest_report(session, "nope") is None
    assert first.id != second.id
