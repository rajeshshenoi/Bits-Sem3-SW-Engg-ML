"""Integration tests: HTTP API -> services -> model, end-to-end."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from brakeguard.api import create_app
from brakeguard.data import SyntheticTelemetrySource
from brakeguard.schemas import RAW_FEATURES
from tests.conftest import HIGH_RISK, LOW_RISK


# -- operations ------------------------------------------------------------------------
def test_health_and_readiness(client):
    assert client.get("/health").json()["status"] == "alive"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["model_loaded"] is True


def test_readiness_fails_without_model(no_model_client):
    assert no_model_client.get("/health").status_code == 200
    response = no_model_client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"


def test_prediction_without_model_returns_503(no_model_client):
    response = no_model_client.post("/api/v1/predictions", json={"telemetry": HIGH_RISK})
    assert response.status_code == 503


def test_request_id_is_propagated(client):
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert "X-Process-Time-ms" in response.headers


def test_dashboard_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "BrakeGuard AI" in response.text


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/v1/predictions", "/api/v1/predictions/batch", "/api/v1/telemetry", "/ready"):
        assert path in paths


# -- predictions ----------------------------------------------------------------------------
def test_predict_from_body(client):
    response = client.post("/api/v1/predictions", json={"vehicle_id": "BG-77777", "telemetry": HIGH_RISK})
    assert response.status_code == 200
    body = response.json()
    assert body["risk"] == "HIGH"
    assert 0 <= body["fault_probability"] <= 1
    assert len(body["top_factors"]) == 3
    assert body["model_version"] == "brakeguard-rf-v2.0"


def test_predict_from_feature_store_is_audited(client):
    response = client.post("/api/v1/vehicles/BG-10245/predictions")
    assert response.status_code == 200
    audit = client.get("/api/v1/audit").json()["predictions"]
    assert audit[0]["prediction_id"] == response.json()["prediction_id"]
    assert audit[0]["vehicle_id"] == "BG-10245"


def test_batch_prediction_preserves_order(client):
    items = [
        {"vehicle_id": "BG-00001", "telemetry": HIGH_RISK},
        {"vehicle_id": "BG-00002", "telemetry": LOW_RISK},
    ]
    response = client.post("/api/v1/predictions/batch", json={"items": items})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["vehicle_id"] for p in body["predictions"]] == ["BG-00001", "BG-00002"]
    assert body["predictions"][0]["fault_probability"] > body["predictions"][1]["fault_probability"]


def test_batch_size_limit(client):
    items = [{"telemetry": LOW_RISK}] * 501
    assert client.post("/api/v1/predictions/batch", json={"items": items}).status_code == 422
    assert client.post("/api/v1/predictions/batch", json={"items": []}).status_code == 422


# -- input validation (security) ------------------------------------------------------------
def test_out_of_range_input_rejected(client):
    response = client.post("/api/v1/predictions", json={"telemetry": {**HIGH_RISK, "pad_wear_pct": 140}})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["loc"][-1] == "pad_wear_pct"


def test_missing_field_rejected(client):
    telemetry = {k: v for k, v in HIGH_RISK.items() if k != "vibration_rms"}
    assert client.post("/api/v1/predictions", json={"telemetry": telemetry}).status_code == 422


def test_unexpected_field_rejected(client):
    response = client.post("/api/v1/predictions", json={"telemetry": {**HIGH_RISK, "__class__": "x"}})
    assert response.status_code == 422


def test_wrong_type_rejected(client):
    response = client.post("/api/v1/predictions", json={"telemetry": {**HIGH_RISK, "brake_pressure": "high"}})
    assert response.status_code == 422


def test_malformed_vehicle_id_rejected(client):
    assert client.get("/api/v1/vehicles/<script>/features").status_code in (404, 422)
    assert client.get("/api/v1/vehicles/bg-lower/features").status_code == 422


def test_unknown_vehicle_returns_404(client):
    response = client.get("/api/v1/vehicles/BG-99999/features")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "vehicle_not_found"


# -- authentication -----------------------------------------------------------------------------
def test_api_key_required_when_configured(secured_client):
    assert secured_client.post("/api/v1/predictions", json={"telemetry": HIGH_RISK}).status_code == 401
    wrong = {"X-API-Key": "guess"}
    assert (
        secured_client.post("/api/v1/predictions", json={"telemetry": HIGH_RISK}, headers=wrong).status_code
        == 401
    )
    ok = {"X-API-Key": "s3cret-key"}
    assert (
        secured_client.post("/api/v1/predictions", json={"telemetry": HIGH_RISK}, headers=ok).status_code
        == 200
    )


def test_health_is_public_even_with_api_key(secured_client):
    assert secured_client.get("/health").status_code == 200


# -- event-driven flow ---------------------------------------------------------------------------
def test_telemetry_event_updates_feature_store(client):
    payload = {"vehicle_id": "BG-55555", **HIGH_RISK}
    response = client.post("/api/v1/telemetry", json=payload)
    assert response.status_code == 202
    assert set(response.json()["delivered_to"]) == {"FeatureStoreConsumer", "DriftMonitorConsumer"}
    features = client.get("/api/v1/vehicles/BG-55555/features").json()["features"]
    assert features["pad_wear_pct"] == HIGH_RISK["pad_wear_pct"]
    assert client.get("/api/v1/events").json()["events"][0]["vehicle_id"] == "BG-55555"


def test_monitoring_waits_for_minimum_drift_window(client):
    drift = client.get("/api/v1/monitoring").json()["drift"]
    assert drift == {"status": "insufficient_data", "samples": 0, "required_samples": 200}


def _publish_fleet(client, frame):
    for i, row in frame.iterrows():
        payload = {"vehicle_id": f"BG-{i:05d}", **{k: round(float(row[k]), 3) for k in RAW_FEATURES}}
        assert client.post("/api/v1/telemetry", json=payload).status_code == 202


def test_monitoring_detects_shifted_live_fleet(settings, trained_bundle):
    """Heat-wave scenario: brakes run hotter and pads are more worn than in training."""
    app = create_app(settings=replace(settings, drift_min_samples=200), bundle=trained_bundle)
    with TestClient(app) as client:
        _publish_fleet(client, SyntheticTelemetrySource(250, seed=5, shift=0.8).load())
        drift = client.get("/api/v1/monitoring").json()["drift"]
    assert {"brake_temperature", "pad_wear_pct"} <= set(drift["drifted_features"])
    assert drift["status"]["odometer_km"] == "stable"


def test_monitoring_stays_quiet_for_in_distribution_fleet(settings, trained_bundle):
    app = create_app(settings=replace(settings, drift_min_samples=200), bundle=trained_bundle)
    with TestClient(app) as client:
        _publish_fleet(client, SyntheticTelemetrySource(250, seed=6).load())
        monitoring = client.get("/api/v1/monitoring").json()
    assert monitoring["drift"]["drifted_features"] == []
    assert monitoring["model_quality"]["recall"] >= 0.75
