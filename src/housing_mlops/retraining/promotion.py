import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from housing_mlops.api.db import PromotionDecision, register_model_version
from housing_mlops.data.schema import TARGET_COLUMN
from housing_mlops.training import (
    LOWER_IS_BETTER,
    PRIMARY_METRIC,
    LoadedModel,
    is_better,
    predict_frame,
    regression_metrics,
    set_current,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    candidate_version: str
    champion_version: str | None
    metric: str
    candidate_value: float
    champion_value: float | None
    improvement: float | None
    promoted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score(model: LoadedModel, frame: pd.DataFrame, metric: str = PRIMARY_METRIC) -> float:
    predictions = predict_frame(model.pipeline, frame)
    return regression_metrics(frame[TARGET_COLUMN], predictions)[metric]


def relative_improvement(candidate: float, champion: float, metric: str) -> float:
    scale = abs(champion) if champion != 0 else 1.0
    gain = (champion - candidate) if LOWER_IS_BETTER[metric] else (candidate - champion)
    return gain / scale


def gate(
    candidate: LoadedModel,
    champion: LoadedModel | None,
    test_frame: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
    min_improvement: float = 0.0,
) -> GateResult:
    if min_improvement < 0:
        raise ValueError("min_improvement must not be negative")
    candidate_value = score(candidate, test_frame, metric)
    if champion is None:
        return GateResult(
            candidate_version=candidate.metadata.version,
            champion_version=None,
            metric=metric,
            candidate_value=candidate_value,
            champion_value=None,
            improvement=None,
            promoted=True,
            reason=f"no champion; candidate {metric}={candidate_value:.4f} promoted",
        )

    if candidate.metadata.training_data_hash != champion.metadata.training_data_hash:
        log.warning(
            "candidate and champion were trained on different data (%s vs %s); "
            "the shared test split may not be held out for both",
            candidate.metadata.training_data_hash[:8],
            champion.metadata.training_data_hash[:8],
        )

    champion_value = score(champion, test_frame, metric)
    improvement = relative_improvement(candidate_value, champion_value, metric)
    promoted = is_better(candidate_value, champion_value, metric) and improvement >= min_improvement
    verdict = "promoted" if promoted else "rejected"
    reason = (
        f"candidate {metric}={candidate_value:.4f} vs champion {champion_value:.4f} "
        f"({improvement:+.2%}, required >= {min_improvement:.2%}): {verdict}"
    )
    return GateResult(
        candidate_version=candidate.metadata.version,
        champion_version=champion.metadata.version,
        metric=metric,
        candidate_value=candidate_value,
        champion_value=champion_value,
        improvement=improvement,
        promoted=promoted,
        reason=reason,
    )


def record_decision(
    session: Session, result: GateResult, drift_report_id: int | None = None
) -> PromotionDecision:
    row = PromotionDecision(
        drift_report_id=drift_report_id,
        candidate_version=result.candidate_version,
        champion_version=result.champion_version,
        metric=result.metric,
        candidate_value=result.candidate_value,
        champion_value=result.champion_value,
        promoted=result.promoted,
        reason=result.reason[:255],
    )
    session.add(row)
    session.commit()
    return row


def promote(candidate: LoadedModel, model_dir: Path, session: Session | None = None) -> Path:
    target = set_current(candidate.source_dir, model_dir)
    if session is not None:
        register_model_version(session, candidate.metadata, source="promotion")
    log.info("Promoted %s to %s", candidate.metadata.version, target)
    return target
