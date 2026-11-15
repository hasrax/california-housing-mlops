import numpy as np
import pytest

from housing_mlops.api.schemas import HousingFeatures
from housing_mlops.data.schema import FEATURE_BOUNDS, FEATURE_COLUMNS
from housing_mlops.monitoring.drift import compute_drift
from housing_mlops.monitoring.simulator import (
    DriftScenario,
    TrafficPlan,
    clip_to_bounds,
    run_traffic,
    to_payloads,
)
from housing_mlops.training.reference import compute_reference_statistics
from tests.conftest import make_housing_frame


@pytest.fixture
def source():
    return make_housing_frame(n_rows=2000, seed=11)


def test_zero_intensity_leaves_data_unchanged(source):
    out = DriftScenario().apply(source, 0.0)
    np.testing.assert_allclose(out[FEATURE_COLUMNS].to_numpy(), source[FEATURE_COLUMNS].to_numpy())


def test_full_intensity_moves_the_configured_features(source):
    out = DriftScenario().apply(source, 1.0)
    assert (out["MedInc"] >= source["MedInc"]).all()
    assert out["MedInc"].mean() > 1.5 * source["MedInc"].mean()
    assert out["HouseAge"].mean() < 0.5 * source["HouseAge"].mean()
    shift = out["Latitude"].mean() - source["Latitude"].mean()
    assert -2.0 <= shift < -1.0
    assert out["Latitude"].min() > FEATURE_BOUNDS["Latitude"][0]
    np.testing.assert_allclose(out["AveRooms"], source["AveRooms"])


def test_drifted_payloads_always_pass_request_validation(source):
    for intensity in (0.0, 0.3, 0.7, 1.0, 5.0):
        for payload in to_payloads(DriftScenario().apply(source.head(200), intensity)):
            HousingFeatures(**payload)


def test_clip_to_bounds_keeps_values_valid():
    frame = make_housing_frame(50, seed=1)
    frame.loc[0, "MedInc"] = 999.0
    frame.loc[1, "HouseAge"] = -5.0
    frame.loc[2, "AveBedrms"] = frame.loc[2, "AveRooms"] + 3.0
    clipped = clip_to_bounds(frame)
    assert clipped.loc[0, "MedInc"] == FEATURE_BOUNDS["MedInc"][1]
    assert clipped.loc[1, "HouseAge"] > FEATURE_BOUNDS["HouseAge"][0]
    assert clipped.loc[2, "AveBedrms"] <= clipped.loc[2, "AveRooms"]
    for payload in to_payloads(clipped):
        HousingFeatures(**payload)


def test_plan_yields_normal_batch_then_increasing_drift(source):
    plan = TrafficPlan(source=source, normal_requests=50, drifted_requests=20, seed=3)
    batches = list(plan.batches())
    assert batches[0][0] == "normal"
    assert batches[0][1] == 0.0
    assert len(batches[0][2]) == 50
    drifted = batches[1:]
    assert len(drifted) == 20
    intensities = [intensity for _, intensity, _ in drifted]
    assert intensities == sorted(intensities)
    assert intensities[0] == 0.0 and intensities[-1] == 1.0
    assert all(len(payloads) == 1 for _, _, payloads in drifted)


@pytest.mark.parametrize(("normal", "drifted"), [(0, 5), (5, 0), (0, 0), (0, 1)])
def test_plan_handles_zero_or_single_counts(source, normal, drifted):
    batches = list(
        TrafficPlan(source=source, normal_requests=normal, drifted_requests=drifted).batches()
    )
    phases = [phase for phase, _, _ in batches]
    assert phases.count("normal") == (1 if normal else 0)
    assert phases.count("drifted") == drifted
    assert sum(len(p) for _, _, p in batches) == normal + drifted


def test_plan_rejects_negative_counts(source):
    with pytest.raises(ValueError):
        list(TrafficPlan(source=source, normal_requests=-1, drifted_requests=0).batches())


def test_plan_is_deterministic_for_a_seed(source):
    first = list(TrafficPlan(source, 10, 5, seed=9).batches())
    second = list(TrafficPlan(source, 10, 5, seed=9).batches())
    assert first == second
    assert first != list(TrafficPlan(source, 10, 5, seed=10).batches())


def test_run_traffic_records_each_request_and_failures(source):
    plan = TrafficPlan(source=source, normal_requests=8, drifted_requests=4, seed=0)
    sent = []

    def fake_send(payload):
        sent.append(payload)
        return 200 if len(sent) % 3 else 422

    summary = run_traffic(plan, fake_send)
    assert summary.sent == len(sent) == 12
    assert summary.by_phase == {"normal": 8, "drifted": 4}
    assert summary.failed == 4
    assert summary.succeeded == 8


def test_drifted_traffic_is_detected_and_normal_traffic_is_not(source):
    reference = compute_reference_statistics(make_housing_frame(n_rows=5000, seed=12))
    plan = TrafficPlan(source=source, normal_requests=400, drifted_requests=400, seed=5)
    phases = {"normal": [], "drifted": []}
    for phase, _, payloads in plan.batches():
        phases[phase].extend(payloads)

    import pandas as pd

    normal = compute_drift(reference, pd.DataFrame(phases["normal"]))
    drifted = compute_drift(reference, pd.DataFrame(phases["drifted"]))
    assert normal.drifted is False
    assert drifted.drifted is True
    assert {"MedInc", "HouseAge"} <= set(drifted.drifted_features)
