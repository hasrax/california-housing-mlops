from collections.abc import Callable
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE = "baseline"
GRADIENT_BOOSTING = "gradient_boosting"

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    BASELINE: {"alpha": 1.0},
    GRADIENT_BOOSTING: {
        "learning_rate": 0.1,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
}

MONOTONIC_FEATURES: dict[str, int] = {"MedInc": 1}


def build_baseline(seed: int, **params: Any) -> Pipeline:
    merged = {**DEFAULT_PARAMS[BASELINE], **params}
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(**merged))])


def build_gradient_boosting(seed: int, **params: Any) -> Pipeline:
    merged = {**DEFAULT_PARAMS[GRADIENT_BOOSTING], **params}
    model = HistGradientBoostingRegressor(
        random_state=seed,
        monotonic_cst=MONOTONIC_FEATURES,
        **merged,
    )
    return Pipeline([("model", model)])


PipelineBuilder = Callable[..., Pipeline]

MODEL_BUILDERS: dict[str, PipelineBuilder] = {
    BASELINE: build_baseline,
    GRADIENT_BOOSTING: build_gradient_boosting,
}

MODEL_NAMES: tuple[str, ...] = tuple(MODEL_BUILDERS)


def build_pipeline(model_name: str, *, seed: int, **params: Any) -> Pipeline:
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {MODEL_NAMES}") from exc
    return builder(seed=seed, **params)
