"""Unit tests: each class tested in isolation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from brakeguard.config import RiskThresholds
from brakeguard.evaluation import evaluate_classifier, expected_calibration_error
from brakeguard.exceptions import (
    FeatureEngineeringError,
    ModelNotReadyError,
    ModelRegistryError,
    VehicleNotFoundError,
)
from brakeguard.features import ENGINEERED_FEATURES, FeatureEngineer, FeatureStore
from brakeguard.models import BrakeFaultPredictor, ModelRegistry
from brakeguard.monitoring import AuditLog, EventBus
from brakeguard.schemas import RAW_FEATURES
from tests.conftest import HIGH_RISK


# -- configuration -------------------------------------------------------------
@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "LOW"), (0.39, "LOW"), (0.40, "MEDIUM"), (0.69, "MEDIUM"), (0.70, "HIGH"), (1.0, "HIGH")],
)
def test_risk_threshold_boundaries(probability, expected):
    assert RiskThresholds().classify(probability) == expected


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        RiskThresholds(high=0.3, medium=0.5)


# -- feature engineering ---------------------------------------------------------
def test_feature_engineer_adds_derived_features():
    frame = pd.DataFrame([HIGH_RISK])
    out = FeatureEngineer().fit(frame).transform(frame)
    assert list(out.columns) == list(ENGINEERED_FEATURES)
    assert out.loc[0, "pressure_deviation"] == pytest.approx(30.0)
    assert out.loc[0, "thermal_wear_index"] == pytest.approx(230.0 * 92.0 / 100)
    assert out.loc[0, "fluid_age_years"] == pytest.approx(700 / 365)


def test_feature_engineer_imputes_missing_with_training_median(caplog):
    train = pd.DataFrame([HIGH_RISK, {**HIGH_RISK, "pad_wear_pct": 50.0}])
    engineer = FeatureEngineer().fit(train)
    sample = pd.DataFrame([{**HIGH_RISK, "pad_wear_pct": np.nan}])
    with caplog.at_level(logging.WARNING, logger="brakeguard"):
        out = engineer.transform(sample)
    assert out.loc[0, "pad_wear_pct"] == pytest.approx(71.0)
    assert "Imputing 1 missing value" in caplog.text


def test_feature_engineer_rejects_missing_columns(caplog):
    frame = pd.DataFrame([{k: v for k, v in HIGH_RISK.items() if k != "vibration_rms"}])
    with caplog.at_level(logging.ERROR, logger="brakeguard"), pytest.raises(FeatureEngineeringError):
        FeatureEngineer().fit(frame)
    assert "missing columns" in caplog.text


def test_feature_engineer_rejects_non_numeric():
    frame = pd.DataFrame([{**HIGH_RISK, "brake_pressure": "high"}])
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineer().fit(frame)


def test_feature_engineer_requires_fit():
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineer().transform(pd.DataFrame([HIGH_RISK]))


# -- feature store -----------------------------------------------------------------
def test_feature_store_upsert_and_get():
    store = FeatureStore(seed={})
    store.upsert("BG-1", {**HIGH_RISK, "vehicle_id": "BG-1", "event_type": "x"})
    assert store.get("BG-1") == HIGH_RISK
    assert "BG-1" in store and len(store) == 1


def test_feature_store_returns_copies():
    store = FeatureStore()
    store.get("BG-10245")["pad_wear_pct"] = -1
    assert store.get("BG-10245")["pad_wear_pct"] == 78.0


def test_feature_store_unknown_vehicle():
    with pytest.raises(VehicleNotFoundError):
        FeatureStore().get("NOPE-1")


# -- event bus / audit ---------------------------------------------------------------
def test_event_bus_isolates_failing_subscriber():
    bus, received = EventBus(), []

    def broken(_event):
        raise RuntimeError("boom")

    bus.subscribe("E", broken, "broken")
    bus.subscribe("E", received.append, "good")
    event = bus.publish({"event_type": "E", "x": 1})
    assert received and received[0]["x"] == 1
    assert event["delivered_to"] == ["good"]
    assert event["failed_subscribers"] == ["broken"]


def test_event_bus_requires_event_type():
    with pytest.raises(ValueError):
        EventBus().publish({"x": 1})


def test_audit_log_is_bounded_and_newest_first():
    log = AuditLog(max_entries=3)
    for i in range(5):
        log.record(
            vehicle_id=f"V{i}",
            fault_probability=0.1,
            risk="LOW",
            model_version="m",
            latency_ms=1,
            explanation="",
        )
    recent = log.recent()
    assert [r["vehicle_id"] for r in recent] == ["V4", "V3", "V2"]
    assert log.total == 5


# -- metrics -----------------------------------------------------------------------------
def test_ece_is_zero_for_perfect_calibration():
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error([0, 0, 1, 1], y_prob) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    # model says 90% but only 50% are positive -> ECE = 0.4
    assert expected_calibration_error([0, 1, 0, 1], [0.9, 0.9, 0.9, 0.9]) == pytest.approx(0.4)


def test_evaluate_classifier_perfect_predictions():
    report = evaluate_classifier([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert report.recall == report.precision == report.f1 == report.accuracy == 1.0
    assert report.roc_auc == 1.0


# -- registry / predictor ------------------------------------------------------------------
def test_registry_roundtrip(tmp_path, trained_bundle):
    registry = ModelRegistry(tmp_path / "m.joblib")
    registry.save(trained_bundle)
    loaded = registry.load()
    assert loaded.version == trained_bundle.version
    assert loaded.metrics == trained_bundle.metrics


def test_registry_missing_and_corrupt_artefacts(tmp_path, caplog):
    with pytest.raises(ModelRegistryError):
        ModelRegistry(tmp_path / "absent.joblib").load()
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a pickle")
    with caplog.at_level(logging.ERROR, logger="brakeguard"), pytest.raises(ModelRegistryError):
        ModelRegistry(corrupt).load()
    assert "corrupt" in caplog.text


def test_predictor_without_model_raises():
    predictor = BrakeFaultPredictor(None)
    assert not predictor.is_ready
    with pytest.raises(ModelNotReadyError):
        predictor.predict(HIGH_RISK)


def test_predictor_rejects_incomplete_record(trained_bundle):
    predictor = BrakeFaultPredictor(trained_bundle)
    with pytest.raises(FeatureEngineeringError):
        predictor.predict({"brake_temperature": 100})


def test_schema_and_engineered_features_consistent():
    assert set(RAW_FEATURES).issubset(ENGINEERED_FEATURES)
