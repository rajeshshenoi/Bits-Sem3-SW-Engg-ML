"""Feature engineering as a scikit-learn compatible transformer.

Packaging feature engineering as a transformer (instead of ad-hoc notebook
cells) means the *exact same* transformation is fitted during training,
serialised together with the model, and replayed at inference time - this
eliminates training/serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..exceptions import FeatureEngineeringError
from ..logging_config import get_logger
from ..schemas import RAW_FEATURES

logger = get_logger(__name__)

NOMINAL_PRESSURE_BAR = 65.0

DERIVED_FEATURES: tuple[str, ...] = (
    "pressure_deviation",
    "thermal_wear_index",
    "fluid_age_years",
)
ENGINEERED_FEATURES: tuple[str, ...] = RAW_FEATURES + DERIVED_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Imputes missing values and derives domain features.

    * ``pressure_deviation``  - |pressure - nominal|: both low and high line
      pressure indicate hydraulic problems.
    * ``thermal_wear_index``  - temperature x pad-wear interaction: worn pads
      running hot are the dominant failure mode.
    * ``fluid_age_years``     - fluid age in years (human-readable scale).
    """

    def __init__(self, nominal_pressure: float = NOMINAL_PRESSURE_BAR) -> None:
        self.nominal_pressure = nominal_pressure

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803  pylint: disable=unused-argument
        """Learn per-column medians used for imputation (``y`` is ignored, sklearn API)."""
        frame = self._check_input(X)
        # sklearn convention: learned attributes end with "_" and are set in fit()
        # pylint: disable=attribute-defined-outside-init
        self.medians_ = frame.median(numeric_only=True).to_dict()
        self.feature_names_out_ = np.array(ENGINEERED_FEATURES, dtype=object)
        logger.debug("FeatureEngineer fitted on %d rows", len(frame))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return a frame with raw + derived features, imputed."""
        if not hasattr(self, "medians_"):
            raise FeatureEngineeringError("FeatureEngineer must be fitted before transform")
        frame = self._check_input(X).copy()
        n_missing = int(frame.isna().sum().sum())
        if n_missing:
            logger.warning("Imputing %d missing value(s) with training medians", n_missing)
            frame = frame.fillna(value=self.medians_)
        frame["pressure_deviation"] = (frame["brake_pressure"] - self.nominal_pressure).abs()
        frame["thermal_wear_index"] = frame["brake_temperature"] * frame["pad_wear_pct"] / 100.0
        frame["fluid_age_years"] = frame["fluid_age_days"] / 365.0
        return frame[list(ENGINEERED_FEATURES)]

    def get_feature_names_out(self, input_features=None):  # pylint: disable=unused-argument
        """Names of the produced columns (sklearn API)."""
        return np.array(ENGINEERED_FEATURES, dtype=object)

    @staticmethod
    def _check_input(X) -> pd.DataFrame:  # noqa: N803
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError(f"Expected a pandas DataFrame, got {type(X).__name__}")
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            logger.error("Feature engineering input is missing columns %s", missing)
            raise FeatureEngineeringError(f"Missing required columns: {missing}")
        try:
            return X[list(RAW_FEATURES)].astype(float)
        except (TypeError, ValueError) as exc:
            logger.error("Non-numeric telemetry received: %s", exc)
            raise FeatureEngineeringError(f"Non-numeric telemetry: {exc}") from exc
