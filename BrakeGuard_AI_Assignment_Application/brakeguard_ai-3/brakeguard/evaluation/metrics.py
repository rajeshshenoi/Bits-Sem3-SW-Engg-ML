"""Model-quality metrics: discrimination *and* calibration.

For a safety-related maintenance alert we care about

* **Recall** - share of real brake faults we catch (missed faults are the
  costly error), together with **precision** / **F1** to keep false alarms
  manageable, and **PR-AUC** as a threshold-free summary on an imbalanced
  target.
* **Calibration** - the fault *probability* is shown to fleet managers, so it
  must mean what it says. Measured with the **Brier score** and the
  **Expected Calibration Error (ECE)**.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ModelQualityReport:
    """Model-quality metrics on a held-out set."""

    n_samples: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    brier_score: float
    expected_calibration_error: float
    log_loss: float

    def to_dict(self) -> dict:
        """Serialisable representation."""
        return asdict(self)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted mean |accuracy - confidence| across equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot compute ECE on an empty array")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def evaluate_classifier(y_true, y_prob, threshold: float = 0.5) -> ModelQualityReport:
    """Compute the full model-quality report for binary probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    def r(value: float) -> float:
        return round(float(value), 4)

    return ModelQualityReport(
        n_samples=int(y_true.size),
        threshold=threshold,
        accuracy=r(accuracy_score(y_true, y_pred)),
        precision=r(precision_score(y_true, y_pred, zero_division=0)),
        recall=r(recall_score(y_true, y_pred, zero_division=0)),
        f1=r(f1_score(y_true, y_pred, zero_division=0)),
        pr_auc=r(average_precision_score(y_true, y_prob)),
        roc_auc=r(roc_auc_score(y_true, y_prob)),
        brier_score=r(brier_score_loss(y_true, y_prob)),
        expected_calibration_error=r(expected_calibration_error(y_true, y_prob)),
        log_loss=r(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
    )
