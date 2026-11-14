import numpy as np
import pytest

from housing_mlops.data.schema import (
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    TARGET_BOUNDS,
    TARGET_COLUMN,
    SchemaErrors,
    validate_feature_frame,
    validate_housing_frame,
)


def failed_columns(error: SchemaErrors) -> set[str]:
    return set(error.failure_cases["column"].dropna())


def column_level_failures(error: SchemaErrors) -> dict[str, set[str]]:
    cases = error.failure_cases
    column_cases = cases[cases["schema_context"] == "Column"]
    return {column: set(group["check"]) for column, group in column_cases.groupby("column")}


def test_valid_frame_passes_unchanged(housing_frame):
    validated = validate_housing_frame(housing_frame)
    assert validated.shape == housing_frame.shape
    assert list(validated.columns) == list(housing_frame.columns)


def test_feature_schema_accepts_frame_without_target(housing_frame):
    features = housing_frame.drop(columns=[TARGET_COLUMN])
    assert validate_feature_frame(features).shape == features.shape


def test_feature_schema_rejects_target_column(housing_frame):
    with pytest.raises(SchemaErrors):
        validate_feature_frame(housing_frame)


@pytest.mark.parametrize("column", FEATURE_COLUMNS)
def test_missing_column_is_rejected(housing_frame, column):
    with pytest.raises(SchemaErrors):
        validate_housing_frame(housing_frame.drop(columns=[column]))


def test_extra_column_is_rejected(housing_frame):
    frame = housing_frame.assign(Extra=1.0)
    with pytest.raises(SchemaErrors):
        validate_housing_frame(frame)


def test_null_value_is_rejected(housing_frame):
    frame = housing_frame.copy()
    frame.loc[3, "MedInc"] = np.nan
    with pytest.raises(SchemaErrors) as info:
        validate_housing_frame(frame)
    assert "MedInc" in failed_columns(info.value)


@pytest.mark.parametrize(("column", "bounds"), FEATURE_BOUNDS.items())
def test_value_at_or_below_lower_bound_is_rejected(housing_frame, column, bounds):
    frame = housing_frame.copy()
    frame.loc[0, column] = bounds[0]
    with pytest.raises(SchemaErrors) as info:
        validate_housing_frame(frame)
    failures = column_level_failures(info.value)
    assert set(failures) == {column}
    assert failures[column] == {f"greater_than({bounds[0]})"}


@pytest.mark.parametrize(("column", "bounds"), FEATURE_BOUNDS.items())
def test_value_above_upper_bound_is_rejected(housing_frame, column, bounds):
    frame = housing_frame.copy()
    frame.loc[0, column] = bounds[1] + 1.0
    with pytest.raises(SchemaErrors) as info:
        validate_housing_frame(frame)
    failures = column_level_failures(info.value)
    assert set(failures) == {column}
    assert failures[column] == {f"less_than_or_equal_to({bounds[1]})"}


def test_value_at_upper_bound_is_accepted(housing_frame):
    frame = housing_frame.copy()
    frame.loc[0, "HouseAge"] = FEATURE_BOUNDS["HouseAge"][1]
    validate_housing_frame(frame)


def test_target_out_of_range_is_rejected(housing_frame):
    frame = housing_frame.copy()
    frame.loc[0, TARGET_COLUMN] = TARGET_BOUNDS[1] + 0.1
    with pytest.raises(SchemaErrors) as info:
        validate_housing_frame(frame)
    assert failed_columns(info.value) == {TARGET_COLUMN}


def test_more_bedrooms_than_rooms_is_rejected(housing_frame):
    frame = housing_frame.copy()
    frame.loc[0, "AveBedrms"] = frame.loc[0, "AveRooms"] + 1.0
    with pytest.raises(SchemaErrors) as info:
        validate_housing_frame(frame)
    assert "AveBedrms must not exceed AveRooms" in str(info.value)


def test_integer_column_is_coerced_to_float(housing_frame):
    frame = housing_frame.copy()
    frame["HouseAge"] = frame["HouseAge"].astype("int64")
    validated = validate_housing_frame(frame)
    assert validated["HouseAge"].dtype == np.float64


def test_non_numeric_column_is_rejected(housing_frame):
    frame = housing_frame.copy()
    frame["MedInc"] = "rich"
    with pytest.raises(SchemaErrors):
        validate_housing_frame(frame)


def test_lazy_validation_reports_every_failing_column(housing_frame):
    frame = housing_frame.copy()
    frame.loc[0, "MedInc"] = -1.0
    frame.loc[1, "Latitude"] = 0.0
    with pytest.raises(SchemaErrors) as info:
        validate_housing_frame(frame)
    assert {"MedInc", "Latitude"} <= failed_columns(info.value)
