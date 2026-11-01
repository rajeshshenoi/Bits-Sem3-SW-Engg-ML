"""Pydantic request/response schemas (the public API contract).

Ranges are taken from :data:`brakeguard.schemas.TELEMETRY_SCHEMA` so the
API rejects physically implausible input with ``422`` before it reaches the
model (input validation is also our first security control).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import SCHEMA_BY_NAME

VEHICLE_ID_PATTERN = r"^[A-Z0-9][A-Z0-9-]{2,31}$"


def _feature(name: str, example: float) -> Field:
    spec = SCHEMA_BY_NAME[name]
    return Field(
        ...,
        ge=spec.min_value,
        le=spec.max_value,
        description=f"{spec.description} [{spec.unit}]",
        examples=[example],
    )


class TelemetryFeatures(BaseModel):
    """Raw brake telemetry for one vehicle."""

    model_config = ConfigDict(extra="forbid")

    brake_temperature: float = _feature("brake_temperature", 186.4)
    brake_pressure: float = _feature("brake_pressure", 72.1)
    pad_wear_pct: float = _feature("pad_wear_pct", 78.0)
    vibration_rms: float = _feature("vibration_rms", 3.2)
    fluid_age_days: float = _feature("fluid_age_days", 410)
    odometer_km: float = _feature("odometer_km", 84250)


class TelemetryEvent(TelemetryFeatures):
    """Telemetry message published by a vehicle gateway."""

    vehicle_id: str = Field(..., pattern=VEHICLE_ID_PATTERN, examples=["BG-10245"])


class PredictionRequest(BaseModel):
    """Score ad-hoc telemetry (vehicle id optional, used for auditing)."""

    model_config = ConfigDict(extra="forbid")

    vehicle_id: str | None = Field(None, pattern=VEHICLE_ID_PATTERN, examples=["BG-10245"])
    telemetry: TelemetryFeatures


class BatchPredictionRequest(BaseModel):
    """Score many vehicles at once."""

    model_config = ConfigDict(extra="forbid")

    items: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class Factor(BaseModel):
    """One explanation factor."""

    feature: str
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    """Prediction returned to the client."""

    prediction_id: str
    vehicle_id: str | None
    fault_probability: float = Field(..., ge=0, le=1)
    risk: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")
    top_factors: list[Factor]
    explanation: str
    model_version: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction results (same order as the request)."""

    count: int
    model_version: str
    predictions: list[PredictionResponse]


class VehicleFeaturesResponse(BaseModel):
    """Feature-store entry of a vehicle."""

    vehicle_id: str
    features: TelemetryFeatures


class TelemetryAccepted(BaseModel):
    """Acknowledgement for an accepted telemetry event (HTTP 202)."""

    status: str = "accepted"
    event_type: str
    vehicle_id: str
    delivered_to: list[str]
    timestamp: str


class ModelInfo(BaseModel):
    """Metadata of the served model."""

    model_version: str
    trained_at: str
    sklearn_version: str
    features: list[str]
    metrics: dict
    risk_thresholds: dict


class HealthResponse(BaseModel):
    """Liveness / readiness response."""

    status: str
    model_loaded: bool
    model_version: str


class ErrorBody(BaseModel):
    """Error details."""

    code: str
    message: str
    request_id: str | None = None
    details: list | dict | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response."""

    error: ErrorBody
