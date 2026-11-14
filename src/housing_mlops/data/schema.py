import pandas as pd
from pandera.errors import SchemaErrors
from pandera.pandas import Check, Column, DataFrameSchema

FEATURE_COLUMNS: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COLUMN = "MedHouseVal"
ALL_COLUMNS: list[str] = [*FEATURE_COLUMNS, TARGET_COLUMN]

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "MedInc": (0.0, 20.0),
    "HouseAge": (0.0, 100.0),
    "AveRooms": (0.0, 200.0),
    "AveBedrms": (0.0, 50.0),
    "Population": (0.0, 50_000.0),
    "AveOccup": (0.0, 2_000.0),
    "Latitude": (32.0, 42.5),
    "Longitude": (-125.0, -114.0),
}
TARGET_BOUNDS: tuple[float, float] = (0.0, 5.5)


def _bounded_column(lower: float, upper: float) -> Column:
    return Column(
        float,
        checks=[Check.gt(lower), Check.le(upper)],
        nullable=False,
        coerce=True,
    )


def _bedrooms_within_rooms(frame: pd.DataFrame) -> pd.Series:
    return frame["AveBedrms"] <= frame["AveRooms"]


_feature_columns = {name: _bounded_column(*FEATURE_BOUNDS[name]) for name in FEATURE_COLUMNS}

FEATURES_SCHEMA = DataFrameSchema(
    columns=_feature_columns,
    checks=[Check(_bedrooms_within_rooms, error="AveBedrms must not exceed AveRooms")],
    strict=True,
    unique_column_names=True,
    name="california_housing_features",
)

HOUSING_SCHEMA = DataFrameSchema(
    columns={**_feature_columns, TARGET_COLUMN: _bounded_column(*TARGET_BOUNDS)},
    checks=[Check(_bedrooms_within_rooms, error="AveBedrms must not exceed AveRooms")],
    strict=True,
    unique_column_names=True,
    name="california_housing",
)


def validate_housing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return HOUSING_SCHEMA.validate(frame, lazy=True)


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return FEATURES_SCHEMA.validate(frame, lazy=True)


__all__ = [
    "ALL_COLUMNS",
    "FEATURES_SCHEMA",
    "FEATURE_BOUNDS",
    "FEATURE_COLUMNS",
    "HOUSING_SCHEMA",
    "TARGET_BOUNDS",
    "TARGET_COLUMN",
    "SchemaErrors",
    "validate_feature_frame",
    "validate_housing_frame",
]
