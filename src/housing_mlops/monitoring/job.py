import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from housing_mlops.api.db import DriftReport, PredictionLog
from housing_mlops.data.schema import FEATURE_COLUMNS
from housing_mlops.monitoring.drift import DriftResult, DriftThresholds, compute_drift
from housing_mlops.training.metadata import ModelMetadata

log = logging.getLogger(__name__)

DEFAULT_WINDOW = timedelta(hours=24)
DEFAULT_MIN_REQUESTS = 100


def recent_requests(
    session: Session, model_version: str, since: datetime, until: datetime
) -> pd.DataFrame:
    columns = [getattr(PredictionLog, name) for name in FEATURE_COLUMNS]
    statement = (
        select(PredictionLog.created_at, *columns)
        .where(PredictionLog.model_version == model_version)
        .where(PredictionLog.created_at >= since)
        .where(PredictionLog.created_at <= until)
        .order_by(PredictionLog.created_at)
    )
    rows = session.execute(statement).all()
    return pd.DataFrame(rows, columns=["created_at", *FEATURE_COLUMNS])


def store_report(
    session: Session,
    metadata: ModelMetadata,
    result: DriftResult,
    window_start: datetime,
    window_end: datetime,
) -> DriftReport:
    report = DriftReport(
        model_version=metadata.version,
        window_start=window_start,
        window_end=window_end,
        recent_count=result.recent_count,
        n_drifted=result.n_drifted,
        drifted=result.drifted,
        details=result.to_dict(),
    )
    session.add(report)
    session.commit()
    return report


def run_drift_job(
    session: Session,
    metadata: ModelMetadata,
    *,
    window: timedelta = DEFAULT_WINDOW,
    min_requests: int = DEFAULT_MIN_REQUESTS,
    thresholds: DriftThresholds | None = None,
    now: datetime | None = None,
) -> DriftReport | None:
    window_end = now or datetime.now(UTC)
    window_start = window_end - window
    recent = recent_requests(session, metadata.version, window_start, window_end)
    if len(recent) < min_requests:
        log.info(
            "Only %d requests for %s since %s (need %d); skipping drift check",
            len(recent),
            metadata.version,
            window_start.isoformat(),
            min_requests,
        )
        return None

    result = compute_drift(metadata.reference_statistics, recent[FEATURE_COLUMNS], thresholds)
    report = store_report(session, metadata, result, window_start, window_end)
    log.info(
        "Drift report %s: %d requests, %d drifted features %s, drifted=%s",
        report.id,
        result.recent_count,
        result.n_drifted,
        result.drifted_features,
        result.drifted,
    )
    return report


def latest_report(session: Session, model_version: str | None = None) -> DriftReport | None:
    statement = select(DriftReport).order_by(DriftReport.created_at.desc(), DriftReport.id.desc())
    if model_version is not None:
        statement = statement.where(DriftReport.model_version == model_version)
    return session.scalars(statement.limit(1)).first()
