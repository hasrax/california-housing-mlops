import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from housing_mlops.api.db import DriftReport
from housing_mlops.data.split import DatasetSplit
from housing_mlops.monitoring.job import latest_report
from housing_mlops.retraining.promotion import GateResult, gate, promote, record_decision
from housing_mlops.training import GRADIENT_BOOSTING, LoadedModel, load_current, train_model

log = logging.getLogger(__name__)

CANDIDATE_PARAMS: dict[str, dict[str, Any]] = {
    GRADIENT_BOOSTING: {
        "learning_rate": 0.05,
        "max_iter": 600,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 10,
    },
}


@dataclass(frozen=True)
class RetrainingOutcome:
    triggered: bool
    reason: str
    drift_report_id: int | None
    champion_version: str | None
    gate: GateResult | None
    promoted_dir: Path | None

    @property
    def promoted(self) -> bool:
        return bool(self.gate and self.gate.promoted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "drift_report_id": self.drift_report_id,
            "champion_version": self.champion_version,
            "gate": self.gate.to_dict() if self.gate else None,
            "promoted": self.promoted,
            "promoted_dir": str(self.promoted_dir) if self.promoted_dir else None,
        }


def load_champion(model_dir: Path) -> LoadedModel | None:
    try:
        return load_current(model_dir)
    except FileNotFoundError:
        return None


def decide_trigger(
    champion: LoadedModel | None, report: DriftReport | None, force: bool
) -> tuple[bool, str]:
    if force:
        return True, "forced retraining"
    if champion is None:
        return True, "no champion model; training the first one"
    if report is None:
        return False, f"no drift report for champion {champion.metadata.version}"
    if not report.drifted:
        return False, f"drift report {report.id} is not drifted ({report.n_drifted} features)"
    return True, f"drift report {report.id} flagged {report.n_drifted} features"


def retrain_if_drifted(
    session: Session,
    split: DatasetSplit,
    *,
    model_dir: Path,
    tracking_uri: str,
    experiment_name: str,
    seed: int,
    model_name: str = GRADIENT_BOOSTING,
    params: dict[str, Any] | None = None,
    force: bool = False,
    min_improvement: float = 0.0,
    report: DriftReport | None = None,
) -> RetrainingOutcome:
    champion = load_champion(model_dir)
    if report is None and champion is not None:
        report = latest_report(session, champion.metadata.version)
    triggered, reason = decide_trigger(champion, report, force)
    champion_version = champion.metadata.version if champion else None
    report_id = report.id if report else None
    log.info("Retraining %s: %s", "triggered" if triggered else "skipped", reason)
    if not triggered:
        return RetrainingOutcome(False, reason, report_id, champion_version, None, None)

    candidate_params = {**CANDIDATE_PARAMS.get(model_name, {}), **(params or {})}
    result = train_model(
        split,
        model_name=model_name,
        seed=seed,
        model_dir=model_dir,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        params=candidate_params,
    )
    candidate = LoadedModel(result.pipeline, result.metadata, result.artifact_dir)
    decision = gate(candidate, champion, split.test, min_improvement=min_improvement)
    record_decision(session, decision, report_id)
    log.info("Gate: %s", decision.reason)

    promoted_dir = promote(candidate, model_dir, session) if decision.promoted else None
    return RetrainingOutcome(True, reason, report_id, champion_version, decision, promoted_dir)
