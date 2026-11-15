import pytest

from housing_mlops.data.schema import FEATURE_COLUMNS
from housing_mlops.monitoring.drift import DriftThresholds, compute_drift
from housing_mlops.training.reference import compute_reference_statistics
from tests.conftest import make_housing_frame


@pytest.fixture(scope="module")
def reference():
    return compute_reference_statistics(make_housing_frame(n_rows=5000, seed=1))


@pytest.fixture
def recent():
    return make_housing_frame(n_rows=1000, seed=2)


def test_same_distribution_reports_no_drift(reference, recent):
    result = compute_drift(reference, recent)
    assert [f.feature for f in result.features] == FEATURE_COLUMNS
    assert result.recent_count == 1000
    assert result.n_drifted == 0
    assert result.drifted is False
    assert all(f.psi < 0.1 for f in result.features)
    assert all(f.recent_count == 1000 for f in result.features)


def test_single_shifted_feature_is_flagged_but_below_report_threshold(reference, recent):
    shifted = recent.assign(MedInc=recent["MedInc"] * 1.3)
    result = compute_drift(reference, shifted)
    assert result.drifted_features == ["MedInc"]
    assert result.drifted is False
    medinc = next(f for f in result.features if f.feature == "MedInc")
    assert medinc.psi > DriftThresholds().psi or medinc.ks_statistic > 0.1
    assert medinc.recent_mean > medinc.reference_mean


def test_two_shifted_features_make_the_report_drifted(reference, recent):
    shifted = recent.assign(MedInc=recent["MedInc"] * 1.3, HouseAge=recent["HouseAge"] * 0.5)
    result = compute_drift(reference, shifted)
    assert set(result.drifted_features) == {"MedInc", "HouseAge"}
    assert result.drifted is True
    assert result.n_drifted == 2


def test_min_drifted_features_threshold_is_configurable(reference, recent):
    shifted = recent.assign(MedInc=recent["MedInc"] * 1.3)
    result = compute_drift(reference, shifted, DriftThresholds(min_drifted_features=1))
    assert result.drifted is True


def test_to_dict_is_json_friendly(reference, recent):
    payload = compute_drift(reference, recent).to_dict()
    assert set(payload) == {
        "recent_count",
        "n_drifted",
        "drifted",
        "drifted_features",
        "thresholds",
        "features",
    }
    assert len(payload["features"]) == len(FEATURE_COLUMNS)
    assert set(payload["features"][0]) == {
        "feature",
        "psi",
        "ks_statistic",
        "ks_p_value",
        "reference_mean",
        "recent_mean",
        "recent_count",
        "drifted",
    }
    assert payload["thresholds"]["psi"] == DriftThresholds.psi


def test_missing_or_empty_recent_frame_raises(reference, recent):
    with pytest.raises(ValueError, match="missing feature columns"):
        compute_drift(reference, recent.drop(columns=["Latitude"]))
    with pytest.raises(ValueError, match="empty"):
        compute_drift(reference, recent.iloc[0:0])


@pytest.mark.parametrize(
    ("psi", "ks", "p", "expected"),
    [
        (0.0, 0.0, 1.0, False),
        (0.25, 0.0, 1.0, True),
        (0.0, 0.2, 0.001, True),
        (0.0, 0.2, 0.5, False),
        (0.0, 0.05, 0.0001, False),
    ],
)
def test_feature_drift_rule(psi, ks, p, expected):
    assert DriftThresholds().feature_drifted(psi, ks, p) is expected
