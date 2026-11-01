"""Central, immutable application configuration.

Research code typically hard-codes paths, thresholds and seeds inline. In the
production package every tunable lives here and can be overridden through
environment variables (12-factor style) without touching the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass(frozen=True)
class RiskThresholds:
    """Probability cut-offs used to map a fault probability to a risk band."""

    high: float = 0.70
    medium: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.medium < self.high < 1.0:
            raise ValueError(
                f"Invalid thresholds: require 0 < medium ({self.medium}) < high ({self.high}) < 1"
            )

    def classify(self, probability: float) -> str:
        """Return ``HIGH``, ``MEDIUM`` or ``LOW`` for a probability."""
        if probability >= self.high:
            return "HIGH"
        if probability >= self.medium:
            return "MEDIUM"
        return "LOW"


@dataclass(frozen=True)
class ModelHyperParams:
    """Random-forest hyper-parameters (kept identical to Assignment I)."""

    n_estimators: int = 160
    max_depth: int | None = 8
    min_samples_leaf: int = 4
    class_weight: str | None = "balanced"
    random_state: int = 42


@dataclass(frozen=True)
class Settings:
    """Application settings, resolved once at start-up."""

    app_name: str = "BrakeGuard AI"
    model_version: str = os.getenv("BRAKEGUARD_MODEL_VERSION", "brakeguard-rf-v2.0")
    model_path: Path = Path(
        os.getenv(
            "BRAKEGUARD_MODEL_PATH",
            str(PROJECT_ROOT / "artifacts" / "brakeguard_model.joblib"),
        )
    )
    training_samples: int = _env_int("BRAKEGUARD_TRAINING_SAMPLES", 3000)
    test_size: float = _env_float("BRAKEGUARD_TEST_SIZE", 0.2)
    random_seed: int = _env_int("BRAKEGUARD_SEED", 42)
    thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    hyper_params: ModelHyperParams = field(default_factory=ModelHyperParams)
    # Data-quality gates used by the validator and the drift monitor.
    max_missing_ratio: float = _env_float("BRAKEGUARD_MAX_MISSING_RATIO", 0.05)
    psi_warning: float = _env_float("BRAKEGUARD_PSI_WARNING", 0.10)
    psi_alert: float = _env_float("BRAKEGUARD_PSI_ALERT", 0.25)
    # PSI over 10 bins is noise-dominated on small windows; require enough live samples.
    drift_min_samples: int = _env_int("BRAKEGUARD_DRIFT_MIN_SAMPLES", 200)
    # Quality gates the trained model must pass before it is registered.
    min_recall: float = _env_float("BRAKEGUARD_MIN_RECALL", 0.75)
    min_pr_auc: float = _env_float("BRAKEGUARD_MIN_PR_AUC", 0.70)
    # Security: optional API key. Empty string disables auth (local demo).
    api_key: str = os.getenv("BRAKEGUARD_API_KEY", "")
    max_batch_size: int = _env_int("BRAKEGUARD_MAX_BATCH_SIZE", 500)
    log_level: str = os.getenv("BRAKEGUARD_LOG_LEVEL", "INFO")


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance (reads the environment)."""
    return Settings()
