"""Integration tests: training pipeline, registry and CLI working together."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from brakeguard.data import CSVTelemetrySource, SyntheticTelemetrySource
from brakeguard.exceptions import DataValidationError
from brakeguard.models import ModelRegistry
from brakeguard.pipeline import TrainingPipeline, load_or_train
from scripts import train as train_cli


def test_pipeline_from_csv_registers_loadable_model(tmp_path, settings):
    csv = tmp_path / "fleet.csv"
    SyntheticTelemetrySource(3000, seed=21).load().to_csv(csv, index=False)
    # This test checks CSV -> train -> register -> load plumbing, not the recall gate
    # (hold-out recall varies 0.70-0.82 across synthetic seeds), so the gate is relaxed.
    local = replace(settings, model_path=tmp_path / "model.joblib", min_recall=0.65)
    bundle = TrainingPipeline(settings=local, source=CSVTelemetrySource(csv)).run()
    loaded = ModelRegistry(local.model_path).load()
    assert loaded.version == bundle.version
    assert loaded.data_quality["passed"] is True
    probabilities = loaded.pipeline.predict_proba(loaded.reference_frame.head(5))[:, 1]
    assert np.allclose(probabilities, bundle.pipeline.predict_proba(bundle.reference_frame.head(5))[:, 1])


def test_pipeline_refuses_bad_data(tmp_path, settings):
    class BadSource(SyntheticTelemetrySource):
        def load(self):
            frame = super().load()
            frame.loc[: len(frame) // 5, "brake_temperature"] = np.nan
            return frame

    local = replace(settings, model_path=tmp_path / "model.joblib")
    with pytest.raises(DataValidationError):
        TrainingPipeline(settings=local, source=BadSource(500)).run()
    assert not local.model_path.exists()


def test_load_or_train_recovers_from_missing_model(tmp_path, settings, caplog):
    local = replace(settings, model_path=tmp_path / "absent" / "model.joblib")
    bundle = load_or_train(local)
    assert bundle.version == local.model_version
    assert local.model_path.exists()
    assert "training a new one" in caplog.text


def test_cli_writes_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAKEGUARD_MODEL_PATH", str(tmp_path / "cli.joblib"))
    out = tmp_path / "metrics.json"
    assert train_cli.main(["--metrics-out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["model_quality"]["recall"] > 0.75
    assert payload["data_quality"]["passed"] is True


def test_cli_returns_error_code_on_failure(tmp_path):
    assert train_cli.main(["--csv", str(tmp_path / "missing.csv"), "--no-register"]) == 1
