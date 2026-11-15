import logging
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy.orm import Session

from housing_mlops.api.db import PredictionLog
from housing_mlops.api.schemas import HousingFeatures
from housing_mlops.training import LoadedModel, load_current, predict_frame

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    request_id: str
    value: float
    latency_ms: float


class ModelService:
    def __init__(self, model: LoadedModel | None = None):
        self.model = model

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "ModelService":
        try:
            model = load_current(model_dir)
            log.info(
                "Loaded %s version %s from %s",
                model.metadata.model_name,
                model.metadata.version,
                model.source_dir,
            )
        except FileNotFoundError:
            log.warning("No current model found under %s; /predict will return 503", model_dir)
            model = None
        return cls(model)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def require(self) -> LoadedModel:
        if self.model is None:
            raise RuntimeError("no model loaded")
        return self.model

    def predict(self, features: HousingFeatures) -> Prediction:
        model = self.require()
        started = time.perf_counter()
        frame = pd.DataFrame([features.model_dump()])
        value = float(predict_frame(model.pipeline, frame)[0])
        latency_ms = (time.perf_counter() - started) * 1000.0
        return Prediction(request_id=str(uuid4()), value=value, latency_ms=latency_ms)


def log_prediction(
    session: Session, model: LoadedModel, features: HousingFeatures, prediction: Prediction
) -> PredictionLog:
    row = PredictionLog(
        request_id=prediction.request_id,
        model_name=model.metadata.model_name,
        model_version=model.metadata.version,
        prediction=prediction.value,
        latency_ms=prediction.latency_ms,
        **features.model_dump(),
    )
    session.add(row)
    session.commit()
    return row
