import pytest
from sqlalchemy import func, select

from housing_mlops.api.db import DriftReport, ModelVersion
from housing_mlops.monitoring import run_drift_job
from housing_mlops.training import load_current
from tests.integration.test_api_postgres import clean_database, client, database_url  # noqa: F401
from tests.unit.api.conftest import VALID_FEATURES

pytestmark = pytest.mark.integration


def test_drift_job_stores_a_report_in_postgres(client, clean_database, tmp_path):  # noqa: F811
    for _ in range(30):
        assert client.post("/predict", json=VALID_FEATURES).status_code == 200

    model = load_current(tmp_path / "models")
    with client.app.state.session_factory() as session:
        report = run_drift_job(session, model.metadata, min_requests=10)
        assert report is not None
        stored = session.get(DriftReport, report.id)
        assert stored.recent_count == 30
        assert isinstance(stored.details, dict)
        assert len(stored.details["features"]) == 8
        assert stored.created_at.tzinfo is not None

        current = session.scalar(select(ModelVersion).where(ModelVersion.is_current.is_(True)))
        assert current.version == model.metadata.version
        assert session.scalar(select(func.count()).select_from(ModelVersion)) == 1
