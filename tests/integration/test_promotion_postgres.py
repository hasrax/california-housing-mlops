import pytest
from sqlalchemy import select

from housing_mlops.api.db import ModelVersion, PromotionDecision
from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.retraining import gate, promote, record_decision
from housing_mlops.training import (
    BASELINE,
    LoadedModel,
    build_pipeline,
    load_current,
    save_model,
    version_dir,
)
from tests.conftest import make_housing_frame
from tests.integration.test_api_postgres import clean_database, client, database_url  # noqa: F401
from tests.unit.training.test_metadata import sample_metadata

pytestmark = pytest.mark.integration


def test_promotion_is_recorded_in_postgres(client, clean_database, tmp_path, housing_frame):  # noqa: F811
    model_dir = tmp_path / "models"
    champion = load_current(model_dir)

    pipeline = build_pipeline(BASELINE, seed=1, alpha=0.1)
    pipeline.fit(housing_frame[FEATURE_COLUMNS], housing_frame[TARGET_COLUMN])
    metadata = sample_metadata()
    metadata.version = "v2"
    candidate = LoadedModel(
        pipeline, metadata, save_model(pipeline, metadata, version_dir(model_dir, BASELINE, "v2"))
    )

    test_frame = make_housing_frame(300, seed=9)
    with client.app.state.session_factory() as session:
        decision = gate(candidate, champion, test_frame)
        record_decision(session, decision, drift_report_id=None)
        if decision.promoted:
            promote(candidate, model_dir, session)

        stored = session.scalars(select(PromotionDecision)).one()
        assert stored.candidate_version == "v2"
        assert stored.champion_version == champion.metadata.version
        assert stored.created_at.tzinfo is not None

        versions = {r.version: r for r in session.scalars(select(ModelVersion)).all()}
        if decision.promoted:
            assert versions["v2"].is_current is True
            assert versions["v2"].source == "promotion"
            assert versions[champion.metadata.version].is_current is False
            assert load_current(model_dir).metadata.version == "v2"
        else:
            assert "v2" not in versions
            assert load_current(model_dir).metadata.version == champion.metadata.version
