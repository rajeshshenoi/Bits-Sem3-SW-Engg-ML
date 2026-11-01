"""Model registry: versioned persistence of the fitted pipeline + metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from ..exceptions import ModelRegistryError
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """Everything needed to serve and monitor one model version."""

    pipeline: Pipeline
    version: str
    metrics: dict
    feature_names: list[str]
    reference_frame: pd.DataFrame
    data_quality: dict = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sklearn_version: str = sklearn.__version__


class ModelRegistry:
    """File-system model registry (one artefact per path)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        """Whether an artefact is present."""
        return self.path.exists()

    def save(self, bundle: ModelBundle) -> Path:
        """Persist a bundle with joblib."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, self.path)
        except OSError as exc:
            logger.error("Could not save model to %s: %s", self.path, exc)
            raise ModelRegistryError(f"Could not save model: {exc}") from exc
        logger.info("Registered model %s at %s", bundle.version, self.path)
        return self.path

    def load(self) -> ModelBundle:
        """Load a bundle; raises :class:`ModelRegistryError` on any problem."""
        if not self.exists():
            raise ModelRegistryError(f"No model artefact at {self.path}")
        try:
            bundle = joblib.load(self.path)
        except Exception as exc:  # joblib can raise many unpickling errors
            logger.error("Model artefact %s is corrupt: %s", self.path, exc)
            raise ModelRegistryError(f"Corrupt model artefact: {exc}") from exc
        if not isinstance(bundle, ModelBundle):
            logger.error("Artefact %s has unexpected type %s", self.path, type(bundle).__name__)
            raise ModelRegistryError("Artefact is not a ModelBundle")
        if bundle.sklearn_version != sklearn.__version__:
            logger.warning(
                "Model trained with scikit-learn %s, running %s - consider retraining",
                bundle.sklearn_version,
                sklearn.__version__,
            )
        logger.info("Loaded model %s (trained %s)", bundle.version, bundle.trained_at)
        return bundle
