import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

METRIC_NAMES: tuple[str, ...] = ("rmse", "mae", "r2")
PRIMARY_METRIC = "rmse"
LOWER_IS_BETTER: dict[str, bool] = {"rmse": True, "mae": True, "r2": False}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def is_better(candidate: float, champion: float, metric: str = PRIMARY_METRIC) -> bool:
    if LOWER_IS_BETTER[metric]:
        return candidate < champion
    return candidate > champion
