"""Domain-specific exception hierarchy.

Raising typed exceptions (instead of returning ``None`` or bare ``Exception``)
lets the API layer translate each failure into the right HTTP status code.
"""


class BrakeGuardError(Exception):
    """Base class for all BrakeGuard errors."""


class DataIngestionError(BrakeGuardError):
    """Raised when telemetry cannot be loaded or parsed."""


class DataValidationError(BrakeGuardError):
    """Raised when data violates the telemetry schema or quality gates."""

    def __init__(self, message: str, issues: list[str] | None = None) -> None:
        super().__init__(message)
        self.issues = issues or []


class FeatureEngineeringError(BrakeGuardError):
    """Raised when the feature pipeline receives unusable input."""


class ModelNotReadyError(BrakeGuardError):
    """Raised when inference is requested before a model is loaded."""


class ModelTrainingError(BrakeGuardError):
    """Raised when training fails or the model misses its quality gates."""


class ModelRegistryError(BrakeGuardError):
    """Raised when a model artefact cannot be saved or loaded."""


class VehicleNotFoundError(BrakeGuardError):
    """Raised when a vehicle has no features in the feature store."""
