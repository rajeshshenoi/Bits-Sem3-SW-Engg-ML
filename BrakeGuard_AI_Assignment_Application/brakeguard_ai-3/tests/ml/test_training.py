"""ML tests - model training behaviour."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from brakeguard.data import SyntheticTelemetrySource
from brakeguard.exceptions import ModelTrainingError
from brakeguard.models import ModelTrainer
from brakeguard.pipeline import TrainingPipeline
from brakeguard.schemas import TARGET_COLUMN


def test_model_can_overfit_a_small_batch():
    """A model with enough capacity must memorise 32 samples.

    If it cannot, the pipeline is broken (e.g. labels misaligned with
    features or features silently dropped) - the classic "overfit one
    batch" sanity check.
    """
    batch = SyntheticTelemetrySource(n_samples=32, seed=3).load()
    trainer = ModelTrainer()
    X, y = trainer.split_xy(batch)
    pipeline = trainer.build_pipeline(n_estimators=50, max_depth=None, min_samples_leaf=1, class_weight=None)
    pipeline.fit(X, y)
    assert (pipeline.predict(X) == y).mean() == 1.0


def test_training_loss_decreases_as_trees_are_added():
    frame = SyntheticTelemetrySource(n_samples=600, seed=5).load()
    trainer = ModelTrainer()
    X, y = trainer.split_xy(frame)
    losses = trainer.loss_curve(X, y, steps=(1, 5, 10, 25, 50, 100))
    assert losses[-1] < losses[0] * 0.5  # big drop from a single tree to the ensemble
    # once converged the curve may plateau, but it must never climb noticeably
    assert all(later <= earlier + 0.01 for earlier, later in zip(losses, losses[1:]))


def test_model_beats_trivial_baseline(trained_bundle):
    positive_rate = 0.515  # PR-AUC of a random classifier equals the positive rate
    assert trained_bundle.metrics["pr_auc"] > positive_rate + 0.2
    assert trained_bundle.metrics["roc_auc"] > 0.75


def test_model_meets_quality_targets(trained_bundle):
    metrics = trained_bundle.metrics
    assert metrics["recall"] >= 0.75
    assert metrics["expected_calibration_error"] <= 0.10
    assert metrics["brier_score"] <= 0.20


def test_training_is_reproducible():
    frame = SyntheticTelemetrySource(n_samples=500, seed=11).load()
    first = ModelTrainer().train(frame)
    second = ModelTrainer().train(frame)
    assert first.test_metrics == second.test_metrics


def test_no_severe_overfitting(trained_bundle, settings):
    result = ModelTrainer(settings.hyper_params).train(SyntheticTelemetrySource(3000, seed=42).load())
    assert result.train_metrics.roc_auc - result.test_metrics.roc_auc < 0.15


def test_single_class_target_is_rejected():
    frame = SyntheticTelemetrySource(n_samples=100, seed=1).load()
    frame[TARGET_COLUMN] = 0
    with pytest.raises(ModelTrainingError):
        ModelTrainer().train(frame)


def test_quality_gate_rejects_model_trained_on_noise(settings, caplog):
    class ShuffledSource(SyntheticTelemetrySource):
        def load(self):
            frame = super().load()
            frame[TARGET_COLUMN] = np.random.default_rng(0).permutation(frame[TARGET_COLUMN].to_numpy())
            return frame

    pipeline = TrainingPipeline(settings=replace(settings), source=ShuffledSource(1000, seed=1))
    with pytest.raises(ModelTrainingError, match="quality gate"):
        pipeline.run(register=False)
    assert "rejected by quality gate" in caplog.text
