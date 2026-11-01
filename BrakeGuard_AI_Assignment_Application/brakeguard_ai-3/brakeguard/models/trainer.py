"""Model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..config import ModelHyperParams
from ..evaluation import ModelQualityReport, evaluate_classifier
from ..exceptions import ModelTrainingError
from ..features import FeatureEngineer
from ..logging_config import get_logger
from ..schemas import RAW_FEATURES, TARGET_COLUMN

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Output of :meth:`ModelTrainer.train`."""

    pipeline: Pipeline
    test_metrics: ModelQualityReport
    train_metrics: ModelQualityReport
    X_test: pd.DataFrame  # pylint: disable=invalid-name
    y_test: np.ndarray
    reference_frame: pd.DataFrame


class ModelTrainer:
    """Builds and fits the ``FeatureEngineer -> RandomForest`` pipeline."""

    def __init__(self, hyper_params: ModelHyperParams | None = None, test_size: float = 0.2) -> None:
        self.hyper_params = hyper_params or ModelHyperParams()
        self.test_size = test_size

    def build_pipeline(self, **overrides) -> Pipeline:
        """Return an *unfitted* pipeline; ``overrides`` tweak RF parameters."""
        params = {
            "n_estimators": self.hyper_params.n_estimators,
            "max_depth": self.hyper_params.max_depth,
            "min_samples_leaf": self.hyper_params.min_samples_leaf,
            "class_weight": self.hyper_params.class_weight,
            "random_state": self.hyper_params.random_state,
            "n_jobs": 1,
        }
        params.update(overrides)
        return Pipeline(
            [
                ("features", FeatureEngineer()),
                ("model", RandomForestClassifier(**params)),
            ]
        )

    @staticmethod
    def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Separate feature columns from the label column."""
        if TARGET_COLUMN not in frame.columns:
            raise ModelTrainingError(f"Training data has no '{TARGET_COLUMN}' column")
        return frame[list(RAW_FEATURES)], frame[TARGET_COLUMN].to_numpy(dtype=int)

    def train(self, frame: pd.DataFrame) -> TrainingResult:
        """Split, fit and evaluate. Raises :class:`ModelTrainingError` on failure."""
        X, y = self.split_xy(frame)  # pylint: disable=invalid-name
        if len(np.unique(y)) < 2:
            logger.error("Training aborted: target has a single class")
            raise ModelTrainingError("Training data must contain both classes")
        X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
            X,
            y,
            test_size=self.test_size,
            random_state=self.hyper_params.random_state,
            stratify=y,
        )
        pipeline = self.build_pipeline()
        logger.info(
            "Training RandomForest on %d rows (%d features, positive rate %.3f)",
            len(X_train),
            X_train.shape[1],
            y_train.mean(),
        )
        try:
            pipeline.fit(X_train, y_train)
        except ValueError as exc:
            logger.error("Model fit failed: %s", exc)
            raise ModelTrainingError(f"Model fit failed: {exc}") from exc

        train_metrics = evaluate_classifier(y_train, pipeline.predict_proba(X_train)[:, 1])
        test_metrics = evaluate_classifier(y_test, pipeline.predict_proba(X_test)[:, 1])
        logger.info(
            "Hold-out metrics: recall=%.3f precision=%.3f f1=%.3f pr_auc=%.3f brier=%.3f ece=%.3f",
            test_metrics.recall,
            test_metrics.precision,
            test_metrics.f1,
            test_metrics.pr_auc,
            test_metrics.brier_score,
            test_metrics.expected_calibration_error,
        )
        return TrainingResult(
            pipeline=pipeline,
            test_metrics=test_metrics,
            train_metrics=train_metrics,
            X_test=X_test,
            y_test=y_test,
            reference_frame=X_train.reset_index(drop=True),
        )

    def loss_curve(self, X: pd.DataFrame, y, steps=(1, 5, 10, 25, 50, 100)) -> list[float]:  # noqa: N803
        """Training log-loss as trees are added (``warm_start``).

        Random forests have no gradient-descent loss, but the ensemble's
        training log-loss should fall as estimators are added - the
        equivalent of "loss decreases over epochs" for this model family.
        """
        # class_weight=None: "balanced" weights are not supported with warm_start.
        pipeline = self.build_pipeline(n_estimators=steps[0], warm_start=True, class_weight=None)
        features = pipeline.named_steps["features"].fit(X)
        X_t = features.transform(X)  # pylint: disable=invalid-name
        model = pipeline.named_steps["model"]
        losses = []
        for n in steps:
            model.set_params(n_estimators=n)
            model.fit(X_t, y)
            proba = np.clip(model.predict_proba(X_t)[:, 1], 1e-6, 1 - 1e-6)
            losses.append(float(log_loss(y, proba, labels=[0, 1])))
        logger.debug("Loss curve %s -> %s", list(steps), losses)
        return losses
