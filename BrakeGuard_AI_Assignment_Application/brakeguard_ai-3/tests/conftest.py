"""Shared pytest fixtures."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from brakeguard.api import create_app
from brakeguard.config import get_settings
from brakeguard.data import SyntheticTelemetrySource
from brakeguard.pipeline import TrainingPipeline

HIGH_RISK = {
    "brake_temperature": 230.0,
    "brake_pressure": 95.0,
    "pad_wear_pct": 92.0,
    "vibration_rms": 5.5,
    "fluid_age_days": 700.0,
    "odometer_km": 150000.0,
}
LOW_RISK = {
    "brake_temperature": 100.0,
    "brake_pressure": 65.0,
    "pad_wear_pct": 10.0,
    "vibration_rms": 0.4,
    "fluid_age_days": 30.0,
    "odometer_km": 15000.0,
}


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    """Settings pointing the model registry to a temporary directory."""
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    return replace(get_settings(), model_path=model_path, api_key="")


@pytest.fixture(scope="session")
def trained_bundle(settings):
    """Production model trained once per test session (~1 s)."""
    return TrainingPipeline(settings=settings).run(register=True)


@pytest.fixture()
def telemetry_frame():
    """Small, valid, labelled telemetry frame."""
    return SyntheticTelemetrySource(n_samples=400, seed=7).load()


@pytest.fixture()
def client(settings, trained_bundle):
    """Test client of an isolated app instance (fresh state per test)."""
    app = create_app(settings=settings, bundle=trained_bundle)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def secured_client(settings, trained_bundle):
    """App instance that requires an API key."""
    app = create_app(settings=replace(settings, api_key="s3cret-key"), bundle=trained_bundle)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def no_model_client(settings):
    """App instance started without a model (degraded mode)."""
    app = create_app(settings=settings, bundle=None)
    with TestClient(app) as test_client:
        yield test_client
