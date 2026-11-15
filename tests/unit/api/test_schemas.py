import pytest
from pydantic import ValidationError

from housing_mlops.api.schemas import HousingFeatures
from housing_mlops.data.schema import FEATURE_BOUNDS, FEATURE_COLUMNS
from tests.unit.api.conftest import VALID_FEATURES


def test_valid_payload_roundtrips():
    features = HousingFeatures(**VALID_FEATURES)
    assert features.model_dump() == VALID_FEATURES


def test_schema_fields_match_feature_columns():
    assert list(HousingFeatures.model_fields) == FEATURE_COLUMNS


@pytest.mark.parametrize("column", FEATURE_COLUMNS)
def test_missing_field_is_rejected(column):
    payload = {k: v for k, v in VALID_FEATURES.items() if k != column}
    with pytest.raises(ValidationError) as info:
        HousingFeatures(**payload)
    assert info.value.errors()[0]["loc"] == (column,)


def test_extra_field_is_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HousingFeatures(**VALID_FEATURES, Zip=94110)


@pytest.mark.parametrize(("column", "bounds"), FEATURE_BOUNDS.items())
def test_lower_bound_is_exclusive(column, bounds):
    with pytest.raises(ValidationError) as info:
        HousingFeatures(**{**VALID_FEATURES, column: bounds[0]})
    assert info.value.errors()[0]["type"] == "greater_than"


@pytest.mark.parametrize(("column", "bounds"), FEATURE_BOUNDS.items())
def test_upper_bound_is_inclusive(column, bounds):
    payload = {**VALID_FEATURES, column: bounds[1]}
    if column == "AveBedrms":
        payload["AveRooms"] = bounds[1]
    HousingFeatures(**payload)
    with pytest.raises(ValidationError) as info:
        HousingFeatures(**{**payload, column: bounds[1] + 0.001})
    assert info.value.errors()[0]["type"] == "less_than_equal"


def test_more_bedrooms_than_rooms_is_rejected():
    with pytest.raises(ValidationError, match="AveBedrms must not exceed AveRooms"):
        HousingFeatures(**{**VALID_FEATURES, "AveBedrms": 7.5})


def test_non_numeric_value_is_rejected():
    with pytest.raises(ValidationError, match="float_parsing"):
        HousingFeatures(**{**VALID_FEATURES, "MedInc": "rich"})


def test_numeric_strings_are_coerced():
    assert HousingFeatures(**{**VALID_FEATURES, "HouseAge": "41"}).HouseAge == 41.0
