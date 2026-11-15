import pytest
from sqlalchemy import select

from housing_mlops.api.db import (
    ModelVersion,
    PromotionDecision,
    init_db,
    make_engine,
    make_session_factory,
)
from housing_mlops.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from housing_mlops.retraining.promotion import (
    gate,
    promote,
    record_decision,
    relative_improvement,
    score,
)
from housing_mlops.training import (
    BASELINE,
    LoadedModel,
    build_pipeline,
    load_current,
    save_model,
    version_dir,
)
from tests.unit.training.test_metadata import sample_metadata


def fitted_model(split, version, tmp_path, **params) -> LoadedModel:
    pipeline = build_pipeline(BASELINE, seed=0, **params)
    pipeline.fit(split.train[FEATURE_COLUMNS], split.train[TARGET_COLUMN])
    metadata = sample_metadata()
    metadata.version = version
    target = save_model(pipeline, metadata, version_dir(tmp_path / "models", BASELINE, version))
    return LoadedModel(pipeline, metadata, target)


@pytest.fixture
def strong(housing_split, tmp_path):
    return fitted_model(housing_split, "strong", tmp_path, alpha=1.0)


@pytest.fixture
def weak(housing_split, tmp_path):
    return fitted_model(housing_split, "weak", tmp_path, alpha=1e6)


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'gate.db').as_posix()}")
    init_db(engine)
    with make_session_factory(engine)() as session:
        yield session
    engine.dispose()


def test_strong_model_scores_better_than_weak(strong, weak, housing_split):
    assert score(strong, housing_split.test) < score(weak, housing_split.test)
    assert score(strong, housing_split.test, "r2") > score(weak, housing_split.test, "r2")


def test_better_candidate_is_promoted(strong, weak, housing_split):
    result = gate(strong, weak, housing_split.test)
    assert result.promoted is True
    assert result.candidate_version == "strong"
    assert result.champion_version == "weak"
    assert result.improvement > 0
    assert "promoted" in result.reason


def test_worse_candidate_is_rejected(strong, weak, housing_split):
    result = gate(weak, strong, housing_split.test)
    assert result.promoted is False
    assert result.improvement < 0
    assert "rejected" in result.reason


def test_identical_candidate_is_rejected(strong, housing_split):
    result = gate(strong, strong, housing_split.test)
    assert result.promoted is False
    assert result.improvement == pytest.approx(0.0)


def test_no_champion_promotes_candidate(strong, housing_split):
    result = gate(strong, None, housing_split.test)
    assert result.promoted is True
    assert result.champion_version is None
    assert result.champion_value is None
    assert result.improvement is None


def test_min_improvement_blocks_small_gains(strong, weak, housing_split):
    small = gate(strong, weak, housing_split.test, min_improvement=0.0)
    blocked = gate(strong, weak, housing_split.test, min_improvement=0.99)
    assert small.promoted is True
    assert blocked.promoted is False
    assert blocked.improvement == small.improvement


def test_negative_min_improvement_is_rejected(strong, weak, housing_split):
    with pytest.raises(ValueError):
        gate(strong, weak, housing_split.test, min_improvement=-0.1)


def test_gate_respects_metric_direction(strong, weak, housing_split):
    by_r2 = gate(strong, weak, housing_split.test, metric="r2")
    assert by_r2.metric == "r2"
    assert by_r2.promoted is True
    assert by_r2.candidate_value > by_r2.champion_value


@pytest.mark.parametrize(
    ("candidate", "champion", "metric", "expected"),
    [
        (0.4, 0.5, "rmse", 0.2),
        (0.6, 0.5, "rmse", -0.2),
        (0.9, 0.8, "r2", 0.125),
        (0.5, 0.0, "rmse", -0.5),
    ],
)
def test_relative_improvement(candidate, champion, metric, expected):
    assert relative_improvement(candidate, champion, metric) == pytest.approx(expected)


def test_record_decision_persists_row(session, strong, weak, housing_split):
    result = gate(strong, weak, housing_split.test)
    row = record_decision(session, result, drift_report_id=7)
    stored = session.get(PromotionDecision, row.id)
    assert stored.candidate_version == "strong"
    assert stored.champion_version == "weak"
    assert stored.promoted is True
    assert stored.drift_report_id == 7
    assert stored.candidate_value == pytest.approx(result.candidate_value)
    assert stored.created_at is not None


def test_promote_sets_current_and_registers_version(session, strong, weak, tmp_path):
    model_dir = tmp_path / "models"
    promote(weak, model_dir, session)
    promote(strong, model_dir, session)
    assert load_current(model_dir).metadata.version == "strong"
    rows = {r.version: r for r in session.scalars(select(ModelVersion)).all()}
    assert rows["strong"].is_current is True
    assert rows["strong"].source == "promotion"
    assert rows["weak"].is_current is False


def test_promote_without_session_only_touches_disk(strong, tmp_path):
    target = promote(strong, tmp_path / "models")
    assert target == tmp_path / "models" / "current"
    assert load_current(tmp_path / "models").metadata.version == "strong"
