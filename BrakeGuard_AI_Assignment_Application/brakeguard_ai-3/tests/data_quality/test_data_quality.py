"""Data-validation tests: schema, completeness, validity, uniqueness, drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brakeguard.data import CSVTelemetrySource, DataValidator, SyntheticTelemetrySource
from brakeguard.exceptions import DataIngestionError, DataValidationError
from brakeguard.monitoring import DriftDetector, population_stability_index
from brakeguard.schemas import RAW_FEATURES, TARGET_COLUMN


def test_synthetic_data_passes_all_quality_gates(telemetry_frame):
    report = DataValidator().validate(telemetry_frame)
    assert report.passed, report.issues
    assert report.schema_valid
    assert report.overall_missing_ratio == 0.0
    assert report.overall_out_of_range_ratio == 0.0
    assert report.duplicate_ratio == 0.0


def test_missing_column_fails_schema(telemetry_frame):
    report = DataValidator().validate(telemetry_frame.drop(columns=["pad_wear_pct"]))
    assert not report.schema_valid
    assert report.missing_columns == ["pad_wear_pct"]


def test_missing_target_only_required_for_training(telemetry_frame):
    frame = telemetry_frame.drop(columns=[TARGET_COLUMN])
    assert not DataValidator().validate(frame, require_target=True).passed
    assert DataValidator().validate(frame, require_target=False).passed


def test_missing_values_above_threshold_fail(telemetry_frame):
    frame = telemetry_frame.copy()
    frame.loc[frame.index[:40], "brake_temperature"] = np.nan  # 10 %
    report = DataValidator(max_missing_ratio=0.05).validate(frame)
    assert report.missing_ratio["brake_temperature"] == pytest.approx(0.10)
    assert not report.passed


def test_missing_values_below_threshold_warn_only(telemetry_frame):
    frame = telemetry_frame.copy()
    frame.loc[frame.index[:4], "brake_temperature"] = np.nan  # 1 %
    assert DataValidator(max_missing_ratio=0.05).validate(frame).passed


def test_physically_impossible_values_fail(telemetry_frame):
    frame = telemetry_frame.copy()
    frame.loc[frame.index[:20], "pad_wear_pct"] = 150.0
    report = DataValidator().validate(frame)
    assert report.out_of_range_ratio["pad_wear_pct"] == pytest.approx(0.05)
    assert any("pad_wear_pct" in issue for issue in report.issues)


def test_non_numeric_column_fails(telemetry_frame):
    frame = telemetry_frame.copy()
    frame["vibration_rms"] = "n/a"
    assert "vibration_rms" in DataValidator().validate(frame).non_numeric_columns


def test_invalid_labels_fail(telemetry_frame):
    frame = telemetry_frame.copy()
    frame.loc[frame.index[0], TARGET_COLUMN] = 2
    report = DataValidator().validate(frame)
    assert not report.label_values_valid


def test_duplicates_are_measured(telemetry_frame):
    frame = telemetry_frame.copy()
    frame = pd.concat([frame, frame.iloc[:40]], ignore_index=True)
    assert DataValidator().validate(frame).duplicate_ratio == pytest.approx(40 / 440)


def test_validate_or_raise(telemetry_frame):
    with pytest.raises(DataValidationError) as info:
        DataValidator().validate_or_raise(telemetry_frame.drop(columns=["odometer_km"]))
    assert info.value.issues


def test_empty_frame_fails(telemetry_frame):
    assert not DataValidator().validate(telemetry_frame.iloc[0:0]).passed


# -- ingestion ----------------------------------------------------------------------
def test_csv_roundtrip(tmp_path, telemetry_frame):
    path = tmp_path / "fleet.csv"
    telemetry_frame.to_csv(path, index=False)
    loaded = CSVTelemetrySource(path).load()
    assert list(loaded.columns) == list(telemetry_frame.columns)
    assert len(loaded) == len(telemetry_frame)


def test_csv_missing_file(tmp_path):
    with pytest.raises(DataIngestionError):
        CSVTelemetrySource(tmp_path / "missing.csv").load()


def test_csv_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataIngestionError):
        CSVTelemetrySource(path).load()


def test_synthetic_source_rejects_bad_size():
    with pytest.raises(DataIngestionError):
        SyntheticTelemetrySource(n_samples=0)


# -- drift ----------------------------------------------------------------------------
def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(size=5000), rng.normal(1.0, size=5000)) > 0.25


def test_drift_detector_stable_on_fresh_sample():
    reference = SyntheticTelemetrySource(2000, seed=1).load()
    current = SyntheticTelemetrySource(500, seed=2).load()
    report = DriftDetector(reference).evaluate(current)
    assert report.drifted_features == []
    assert set(report.psi) == set(RAW_FEATURES)


def test_drift_detector_flags_shifted_fleet():
    reference = SyntheticTelemetrySource(2000, seed=1).load()
    current = SyntheticTelemetrySource(500, seed=2, shift=1.0).load()
    report = DriftDetector(reference).evaluate(current)
    assert "brake_temperature" in report.drifted_features
    assert report.ks_pvalue["brake_temperature"] < 0.01
    assert report.status["odometer_km"] == "stable"
