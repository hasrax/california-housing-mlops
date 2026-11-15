import pytest
from sklearn.pipeline import Pipeline

from housing_mlops.training.pipelines import (
    BASELINE,
    DEFAULT_PARAMS,
    GRADIENT_BOOSTING,
    MODEL_NAMES,
    MONOTONIC_FEATURES,
    build_pipeline,
)


def test_registry_lists_both_models():
    assert set(MODEL_NAMES) == {BASELINE, GRADIENT_BOOSTING}
    assert set(DEFAULT_PARAMS) == set(MODEL_NAMES)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_builders_return_pipeline_ending_in_model_step(name):
    pipeline = build_pipeline(name, seed=0)
    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps[-1][0] == "model"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("random_forest", seed=0)


def test_params_override_defaults():
    pipeline = build_pipeline(GRADIENT_BOOSTING, seed=0, max_iter=7)
    assert pipeline.named_steps["model"].max_iter == 7
    assert (
        pipeline.named_steps["model"].learning_rate
        == DEFAULT_PARAMS[GRADIENT_BOOSTING]["learning_rate"]
    )


def test_gradient_boosting_is_monotonic_in_income():
    model = build_pipeline(GRADIENT_BOOSTING, seed=0).named_steps["model"]
    assert model.monotonic_cst == MONOTONIC_FEATURES == {"MedInc": 1}


def test_seed_is_applied_to_gradient_boosting():
    assert build_pipeline(GRADIENT_BOOSTING, seed=123).named_steps["model"].random_state == 123


def test_baseline_scales_before_fitting():
    assert [name for name, _ in build_pipeline(BASELINE, seed=0).steps] == ["scale", "model"]
