"""Inference service."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RiskThresholds
from ..exceptions import FeatureEngineeringError, ModelNotReadyError
from ..logging_config import get_logger
from ..schemas import RAW_FEATURES
from .registry import ModelBundle

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Result of scoring one vehicle."""

    fault_probability: float
    risk: str
    top_factors: list[dict] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        """Human-readable summary of the main drivers."""
        if not self.top_factors:
            return "No explanation available"
        parts = [f"{f['feature']} ({f['direction']} than fleet average)" for f in self.top_factors]
        return "Main drivers: " + ", ".join(parts)


class BrakeFaultPredictor:
    """Scores telemetry with a registered :class:`ModelBundle`.

    Explanations use a light-weight local attribution: global feature
    importance x |z-score| of the vehicle's value relative to the training
    distribution. Unlike the Assignment I heuristic (importance x raw value),
    this is unit-independent, so ``odometer_km`` no longer dominates simply
    because it is measured in large numbers.
    """

    def __init__(self, bundle: ModelBundle | None, thresholds: RiskThresholds | None = None) -> None:
        self.bundle = bundle
        self.thresholds = thresholds or RiskThresholds()
        if bundle is not None:
            features = bundle.pipeline.named_steps["features"]
            engineered = features.transform(bundle.reference_frame)
            self._mean = engineered.mean()
            self._std = engineered.std().replace(0, 1.0)
            self._importances = pd.Series(
                bundle.pipeline.named_steps["model"].feature_importances_,
                index=engineered.columns,
            )

    @property
    def is_ready(self) -> bool:
        """Whether a model is loaded."""
        return self.bundle is not None

    @property
    def version(self) -> str:
        """Model version string or ``"unavailable"``."""
        return self.bundle.version if self.bundle else "unavailable"

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        """Fault probabilities for a list of raw-telemetry dicts."""
        self._ensure_ready()
        frame = self._to_frame(records)
        return self.bundle.pipeline.predict_proba(frame)[:, 1]

    def predict(self, features: dict, top_k: int = 3) -> Prediction:
        """Score a single vehicle."""
        return self.predict_batch([features], top_k=top_k)[0]

    def predict_batch(self, records: list[dict], top_k: int = 3) -> list[Prediction]:
        """Score many vehicles in one vectorised call."""
        self._ensure_ready()
        frame = self._to_frame(records)
        probabilities = self.bundle.pipeline.predict_proba(frame)[:, 1]
        engineered = self.bundle.pipeline.named_steps["features"].transform(frame)
        z = (engineered - self._mean) / self._std
        scores = z.abs() * self._importances
        results = []
        for i, probability in enumerate(probabilities):
            top = scores.iloc[i].sort_values(ascending=False).head(top_k)
            factors = [
                {
                    "feature": name,
                    "contribution": round(float(value), 4),
                    "direction": "higher" if z.iloc[i][name] >= 0 else "lower",
                }
                for name, value in top.items()
            ]
            results.append(
                Prediction(
                    fault_probability=float(probability),
                    risk=self.thresholds.classify(float(probability)),
                    top_factors=factors,
                )
            )
        logger.info("Scored %d record(s) with %s", len(results), self.version)
        return results

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            logger.error("Inference requested but no model is loaded")
            raise ModelNotReadyError("Model is not loaded")

    @staticmethod
    def _to_frame(records: list[dict]) -> pd.DataFrame:
        if not records:
            raise FeatureEngineeringError("At least one record is required")
        missing = sorted({c for r in records for c in RAW_FEATURES if c not in r})
        if missing:
            logger.error("Inference input missing features %s", missing)
            raise FeatureEngineeringError(f"Missing required features: {missing}")
        return pd.DataFrame([{c: r[c] for c in RAW_FEATURES} for r in records])
