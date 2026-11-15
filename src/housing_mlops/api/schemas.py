from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from housing_mlops.data.schema import FEATURE_BOUNDS


def _bounded(name: str, description: str) -> Any:
    lower, upper = FEATURE_BOUNDS[name]
    return Field(gt=lower, le=upper, description=description)


class HousingFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MedInc: float = _bounded("MedInc", "Median income of the block group, in tens of thousands")
    HouseAge: float = _bounded("HouseAge", "Median house age in the block group, years")
    AveRooms: float = _bounded("AveRooms", "Average rooms per household")
    AveBedrms: float = _bounded("AveBedrms", "Average bedrooms per household")
    Population: float = _bounded("Population", "Block group population")
    AveOccup: float = _bounded("AveOccup", "Average household members")
    Latitude: float = _bounded("Latitude", "Block group latitude")
    Longitude: float = _bounded("Longitude", "Block group longitude")

    @model_validator(mode="after")
    def bedrooms_within_rooms(self) -> "HousingFeatures":
        if self.AveBedrms > self.AveRooms:
            raise ValueError("AveBedrms must not exceed AveRooms")
        return self


class PredictionResponse(BaseModel):
    request_id: str
    prediction: float = Field(description="Median house value in units of $100,000")
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    database: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    created_at: str
    seed: int
    params: dict[str, Any]
    feature_columns: list[str]
    target_column: str
    training_rows: int
    training_data_hash: str
    metrics: dict[str, dict[str, float]]
    mlflow_run_id: str | None
