from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from housing_mlops.api.db import (
    ModelVersion,
    PromotionDecision,
    init_db,
    make_engine,
    make_session_factory,
)
from housing_mlops.monitoring import run_drift_job
from housing_mlops.retraining.retrain import (
    CANDIDATE_PARAMS,
    decide_trigger,
    load_champion,
    retrain_if_drifted,
)
from housing_mlops.training import (
    BASELINE,
    GRADIENT_BOOSTING,
    list_versions,
    load_current,
    train_model,
)
from tests.conftest import make_housing_frame
from tests.unit.monitoring.test_job import insert_requests

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
EXPERIMENT = "retraining-tests"


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'retrain.db').as_posix()}")
    init_db(engine)
    with make_session_factory(engine)() as session:
        yield session
    engine.dispose()


@pytest.fixture
def model_dir(tmp_path):
    return tmp_path / "models"


@pytest.fixture
def common(model_dir, tracking_uri):
    return {
        "model_dir": model_dir,
        "tracking_uri": tracking_uri,
        "experiment_name": EXPERIMENT,
        "seed": 42,
        "model_name": BASELINE,
    }


def install_champion(housing_split, model_dir, tracking_uri, **params):
    result = train_model(
        housing_split,
        model_name=BASELINE,
        seed=42,
        model_dir=model_dir,
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT,
        params=params,
    )
    from housing_mlops.training import set_current

    set_current(result.artifact_dir, model_dir)
    return result


def drifted_report(session, metadata, drifted: bool):
    frame = make_housing_frame(200, seed=5)
    if drifted:
        frame["MedInc"] *= 2.0
        frame["HouseAge"] *= 0.4
    insert_requests(session, frame, metadata.version, start=NOW)
    report = run_drift_job(session, metadata, min_requests=50, now=NOW + timedelta(minutes=5))
    assert report.drifted is drifted
    return report


def test_no_champion_trains_and_promotes(session, housing_split, common, model_dir):
    assert load_champion(model_dir) is None
    outcome = retrain_if_drifted(session, housing_split, **common)
    assert outcome.triggered is True
    assert "no champion" in outcome.reason
    assert outcome.promoted is True
    assert outcome.gate.champion_version is None
    assert load_current(model_dir).metadata.version == outcome.gate.candidate_version
    assert session.scalar(select(func.count()).select_from(PromotionDecision)) == 1
    assert session.scalar(select(ModelVersion.source)) == "promotion"


def test_without_a_drift_report_nothing_is_trained(session, housing_split, common, model_dir):
    install_champion(housing_split, model_dir, common["tracking_uri"])
    outcome = retrain_if_drifted(session, housing_split, **common)
    assert outcome.triggered is False
    assert "no drift report" in outcome.reason
    assert outcome.gate is None
    assert len(list_versions(model_dir, BASELINE)) == 1
    assert session.scalar(select(func.count()).select_from(PromotionDecision)) == 0


def test_stable_report_does_not_trigger(session, housing_split, common, model_dir):
    champion = install_champion(housing_split, model_dir, common["tracking_uri"])
    report = drifted_report(session, champion.metadata, drifted=False)
    outcome = retrain_if_drifted(session, housing_split, **common)
    assert outcome.triggered is False
    assert outcome.drift_report_id == report.id
    assert "not drifted" in outcome.reason


def test_drifted_report_triggers_and_better_candidate_is_promoted(
    session, housing_split, common, model_dir
):
    champion = install_champion(housing_split, model_dir, common["tracking_uri"], alpha=1e6)
    report = drifted_report(session, champion.metadata, drifted=True)

    outcome = retrain_if_drifted(session, housing_split, params={"alpha": 1.0}, **common)

    assert outcome.triggered is True
    assert outcome.drift_report_id == report.id
    assert outcome.champion_version == champion.metadata.version
    assert outcome.promoted is True
    assert load_current(model_dir).metadata.version == outcome.gate.candidate_version
    decision = session.scalars(select(PromotionDecision)).one()
    assert decision.drift_report_id == report.id
    assert decision.promoted is True
    current = session.scalar(select(ModelVersion).where(ModelVersion.is_current.is_(True)))
    assert current.version == outcome.gate.candidate_version
    assert current.source == "promotion"


def test_drifted_report_triggers_but_worse_candidate_is_rejected(
    session, housing_split, common, model_dir
):
    champion = install_champion(housing_split, model_dir, common["tracking_uri"], alpha=1.0)
    drifted_report(session, champion.metadata, drifted=True)

    outcome = retrain_if_drifted(session, housing_split, params={"alpha": 1e6}, **common)

    assert outcome.triggered is True
    assert outcome.promoted is False
    assert outcome.promoted_dir is None
    assert load_current(model_dir).metadata.version == champion.metadata.version
    decision = session.scalars(select(PromotionDecision)).one()
    assert decision.promoted is False
    assert len(list_versions(model_dir, BASELINE)) == 2


def test_force_retrains_without_any_report(session, housing_split, common, model_dir):
    install_champion(housing_split, model_dir, common["tracking_uri"])
    outcome = retrain_if_drifted(session, housing_split, force=True, **common)
    assert outcome.triggered is True
    assert outcome.reason == "forced retraining"
    assert outcome.drift_report_id is None
    assert outcome.gate is not None
    assert outcome.promoted is False


def test_candidate_uses_stronger_defaults_for_gradient_boosting(
    session, housing_split, common, model_dir
):
    outcome = retrain_if_drifted(
        session,
        housing_split,
        **{**common, "model_name": GRADIENT_BOOSTING},
        params={"max_iter": 10},
    )
    metadata = load_current(model_dir).metadata
    assert metadata.model_name == GRADIENT_BOOSTING
    assert metadata.params["max_iter"] == 10
    assert metadata.params["learning_rate"] == CANDIDATE_PARAMS[GRADIENT_BOOSTING]["learning_rate"]
    assert outcome.promoted is True


def test_decide_trigger_rules():
    assert decide_trigger(None, None, force=False) == (
        True,
        "no champion model; training the first one",
    )
    assert decide_trigger(None, None, force=True) == (True, "forced retraining")


def test_outcome_to_dict_is_json_friendly(session, housing_split, common):
    payload = retrain_if_drifted(session, housing_split, **common).to_dict()
    assert payload["triggered"] is True
    assert payload["promoted"] is True
    assert payload["gate"]["champion_version"] is None
    assert payload["promoted_dir"].endswith("current")
