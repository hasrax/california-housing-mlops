import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from housing_mlops.data.schema import FEATURE_COLUMNS


def predict_frame(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing feature columns: {missing}")
    return np.asarray(pipeline.predict(frame[FEATURE_COLUMNS]), dtype=float)
