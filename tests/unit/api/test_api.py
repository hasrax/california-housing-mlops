import uuid

import pandas as pd
import pytest
from sqlalchemy import select

from housing_mlops import __version__
from housing_mlops.api.db import PredictionLog
from housing_mlops.training import load_current, predict_frame
from tests.unit.api.conftest import VALID_FEATURES
from tests.unit.training.test_metadata import sample_metadata


def logged_rows(client) -> list[PredictionLog]:
    with client.app.state.session_factory() as session:
        return list(session.scalars(select(PredictionLog)).all())


def test_health_reports_ok_with_model_and_database(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": sample_metadata().version,
        "database": "ok",
    }


def test_openapi_exposes_version_and_routes(client):
    spec = client.get("/openapi.json").json()
    assert spec["info"]["version"] == __version__
    assert {"/health", "/predict", "/model-info"} <= set(spec["paths"])


def test_predict_returns_prediction_and_logs_it(client, api_settings):
    response = client.post("/predict", json=VALID_FEATURES)
    assert response.status_code == 200, response.text
    body = response.json()
    uuid.UUID(body["request_id"])
    assert body["model_name"] == "baseline"
    assert body["model_version"] == sample_metadata().version

    model = load_current(api_settings.model_dir)
    expected = predict_frame(model.pipeline, pd.DataFrame([VALID_FEATURES]))[0]
    assert body["prediction"] == pytest.approx(expected)

    rows = logged_rows(client)
    assert len(rows) == 1
    row = rows[0]
    assert row.request_id == body["request_id"]
    assert row.prediction == pytest.approx(expected)
    assert row.model_version == body["model_version"]
    assert row.latency_ms >= 0
    assert row.created_at is not None
    assert {k: getattr(row, k) for k in VALID_FEATURES} == VALID_FEATURES


def test_each_prediction_gets_a_unique_request_id(client):
    ids = {client.post("/predict", json=VALID_FEATURES).json()["request_id"] for _ in range(5)}
    assert len(ids) == 5
    assert len(logged_rows(client)) == 5


@pytest.mark.parametrize(
    ("payload", "expected_loc", "expected_type"),
    [
        ({**VALID_FEATURES, "MedInc": 0.0}, ["body", "MedInc"], "greater_than"),
        ({**VALID_FEATURES, "Latitude": 50.0}, ["body", "Latitude"], "less_than_equal"),
        ({**VALID_FEATURES, "MedInc": "lots"}, ["body", "MedInc"], "float_parsing"),
        ({**VALID_FEATURES, "Extra": 1}, ["body", "Extra"], "extra_forbidden"),
        (
            {k: v for k, v in VALID_FEATURES.items() if k != "AveOccup"},
            ["body", "AveOccup"],
            "missing",
        ),
        ({**VALID_FEATURES, "AveBedrms": 9.0}, ["body"], "value_error"),
    ],
)
def test_invalid_payloads_are_rejected_and_not_logged(client, payload, expected_loc, expected_type):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == expected_loc
    assert error["type"] == expected_type
    assert logged_rows(client) == []


def test_model_info_matches_metadata(client, api_settings):
    metadata = load_current(api_settings.model_dir).metadata
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == metadata.model_name
    assert body["version"] == metadata.version
    assert body["training_data_hash"] == metadata.training_data_hash
    assert body["metrics"] == metadata.metrics
    assert body["feature_columns"] == metadata.feature_columns
    assert "reference_statistics" not in body


def test_without_a_model_health_is_degraded_and_predict_is_503(client_without_model):
    health = client_without_model.get("/health").json()
    assert health["status"] == "degraded"
    assert health["model_loaded"] is False
    assert health["model_version"] is None
    assert health["database"] == "ok"
    assert client_without_model.post("/predict", json=VALID_FEATURES).status_code == 503
    assert client_without_model.get("/model-info").status_code == 503
