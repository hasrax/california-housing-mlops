from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from housing_mlops.data.schema import FEATURE_COLUMNS
from housing_mlops.monitoring.statistics import ks_against_reference, psi_from_reference
from housing_mlops.training.reference import QUANTILE_GRID, ReferenceStatistics


@dataclass(frozen=True)
class DriftThresholds:
    psi: float = 0.2
    ks_statistic: float = 0.1
    ks_p_value: float = 0.01
    min_drifted_features: int = 2

    def feature_drifted(self, psi: float, ks_statistic: float, ks_p_value: float) -> bool:
        ks_drift = ks_statistic >= self.ks_statistic and ks_p_value < self.ks_p_value
        return psi >= self.psi or ks_drift

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    reference_mean: float
    recent_mean: float
    recent_count: int
    drifted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftResult:
    features: list[FeatureDrift]
    recent_count: int
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    @property
    def n_drifted(self) -> int:
        return len(self.drifted_features)

    @property
    def drifted(self) -> bool:
        return self.n_drifted >= self.thresholds.min_drifted_features

    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_count": self.recent_count,
            "n_drifted": self.n_drifted,
            "drifted": self.drifted,
            "drifted_features": self.drifted_features,
            "thresholds": self.thresholds.to_dict(),
            "features": [f.to_dict() for f in self.features],
        }


def feature_drift(
    name: str, reference: dict[str, Any], values, thresholds: DriftThresholds
) -> FeatureDrift:
    array = np.asarray(values, dtype=float)
    psi = psi_from_reference(reference["bin_fractions"], reference["bin_edges"], array)
    ks = ks_against_reference(reference["quantiles"], QUANTILE_GRID, array)
    return FeatureDrift(
        feature=name,
        psi=psi,
        ks_statistic=ks.statistic,
        ks_p_value=ks.p_value,
        reference_mean=float(reference["mean"]),
        recent_mean=float(array.mean()),
        recent_count=int(array.size),
        drifted=thresholds.feature_drifted(psi, ks.statistic, ks.p_value),
    )


def compute_drift(
    reference_statistics: ReferenceStatistics,
    recent: pd.DataFrame,
    thresholds: DriftThresholds | None = None,
) -> DriftResult:
    thresholds = thresholds or DriftThresholds()
    missing = [c for c in FEATURE_COLUMNS if c not in recent.columns]
    if missing:
        raise ValueError(f"recent frame is missing feature columns: {missing}")
    if recent.empty:
        raise ValueError("recent frame must not be empty")
    features = [
        feature_drift(name, reference_statistics[name], recent[name].to_numpy(), thresholds)
        for name in FEATURE_COLUMNS
    ]
    return DriftResult(features=features, recent_count=len(recent), thresholds=thresholds)
