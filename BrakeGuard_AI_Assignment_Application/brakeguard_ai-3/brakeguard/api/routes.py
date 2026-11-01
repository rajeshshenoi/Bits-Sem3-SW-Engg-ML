"""REST endpoints (versioned under ``/api/v1``)."""

from __future__ import annotations

from statistics import median
from time import perf_counter

from fastapi import APIRouter, Depends, Path, Query, Request, status

from ..exceptions import ModelNotReadyError
from ..logging_config import get_logger
from ..models import Prediction
from .container import TELEMETRY_EVENT, ServiceContainer
from .schemas import (
    VEHICLE_ID_PATTERN,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    TelemetryAccepted,
    TelemetryEvent,
    VehicleFeaturesResponse,
)
from .security import require_api_key

logger = get_logger(__name__)

ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Vehicle not found"}}

VEHICLE_ID_PATH = Path(..., pattern=VEHICLE_ID_PATTERN, examples=["BG-10245"])


def get_container(request: Request) -> ServiceContainer:
    """Dependency returning the app's service container."""
    return request.app.state.container


ops_router = APIRouter(tags=["operations"])
api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


# -- operations ------------------------------------------------------------
@ops_router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(c: ServiceContainer = Depends(get_container)) -> HealthResponse:
    """Always 200 while the process is alive."""
    return HealthResponse(
        status="alive", model_loaded=c.predictor.is_ready, model_version=c.predictor.version
    )


@ops_router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Readiness probe",
)
def ready(c: ServiceContainer = Depends(get_container)) -> HealthResponse:
    """200 only when a model is loaded (Kubernetes readiness probe)."""
    if not c.predictor.is_ready:
        raise ModelNotReadyError("Model is not loaded")
    return HealthResponse(status="ready", model_loaded=True, model_version=c.predictor.version)


# -- model -------------------------------------------------------------------
@api_router.get("/model", response_model=ModelInfo, responses=ERRORS, tags=["model"])
def model_info(c: ServiceContainer = Depends(get_container)) -> ModelInfo:
    """Metadata and hold-out metrics of the served model."""
    if c.bundle is None:
        raise ModelNotReadyError("Model is not loaded")
    return ModelInfo(
        model_version=c.bundle.version,
        trained_at=c.bundle.trained_at,
        sklearn_version=c.bundle.sklearn_version,
        features=c.bundle.feature_names,
        metrics=c.bundle.metrics,
        risk_thresholds={"high": c.settings.thresholds.high, "medium": c.settings.thresholds.medium},
    )


def _to_response(
    c: ServiceContainer, vehicle_id: str | None, pred: Prediction, latency_ms: float
) -> PredictionResponse:
    entry = c.audit_log.record(
        vehicle_id=vehicle_id or "ad-hoc",
        fault_probability=pred.fault_probability,
        risk=pred.risk,
        model_version=c.predictor.version,
        latency_ms=latency_ms,
        explanation=pred.explanation,
    )
    return PredictionResponse(
        prediction_id=entry["prediction_id"],
        vehicle_id=vehicle_id,
        fault_probability=round(pred.fault_probability, 4),
        risk=pred.risk,
        top_factors=pred.top_factors,
        explanation=pred.explanation,
        model_version=c.predictor.version,
        latency_ms=round(latency_ms, 3),
    )


@api_router.post("/predictions", response_model=PredictionResponse, responses=ERRORS, tags=["predictions"])
def predict(body: PredictionRequest, c: ServiceContainer = Depends(get_container)) -> PredictionResponse:
    """Score telemetry supplied in the request body."""
    start = perf_counter()
    pred = c.predictor.predict(body.telemetry.model_dump())
    return _to_response(c, body.vehicle_id, pred, (perf_counter() - start) * 1000)


@api_router.post(
    "/predictions/batch",
    response_model=BatchPredictionResponse,
    responses=ERRORS,
    tags=["predictions"],
)
def predict_batch(
    body: BatchPredictionRequest, c: ServiceContainer = Depends(get_container)
) -> BatchPredictionResponse:
    """Score up to 500 telemetry records in one vectorised call."""
    start = perf_counter()
    preds = c.predictor.predict_batch([item.telemetry.model_dump() for item in body.items])
    per_item = (perf_counter() - start) * 1000 / len(preds)
    responses = [_to_response(c, item.vehicle_id, p, per_item) for item, p in zip(body.items, preds)]
    return BatchPredictionResponse(
        count=len(responses), model_version=c.predictor.version, predictions=responses
    )


# -- vehicles / feature store ---------------------------------------------
@api_router.get("/vehicles", tags=["vehicles"], responses=ERRORS)
def list_vehicles(c: ServiceContainer = Depends(get_container)) -> dict:
    """Vehicles known to the feature store."""
    return {"vehicles": c.feature_store.vehicle_ids()}


@api_router.get(
    "/vehicles/{vehicle_id}/features",
    response_model=VehicleFeaturesResponse,
    responses={**ERRORS, **NOT_FOUND},
    tags=["vehicles"],
)
def vehicle_features(vehicle_id: str = VEHICLE_ID_PATH, c: ServiceContainer = Depends(get_container)):
    """Latest feature vector of a vehicle."""
    return VehicleFeaturesResponse(vehicle_id=vehicle_id, features=c.feature_store.get(vehicle_id))


@api_router.post(
    "/vehicles/{vehicle_id}/predictions",
    response_model=PredictionResponse,
    responses={**ERRORS, **NOT_FOUND},
    tags=["predictions"],
)
def predict_vehicle(
    vehicle_id: str = VEHICLE_ID_PATH, c: ServiceContainer = Depends(get_container)
) -> PredictionResponse:
    """Score a vehicle using its features from the feature store."""
    features = c.feature_store.get(vehicle_id)
    start = perf_counter()
    pred = c.predictor.predict(features)
    return _to_response(c, vehicle_id, pred, (perf_counter() - start) * 1000)


# -- event-driven telemetry ---------------------------------------------------
@api_router.post(
    "/telemetry",
    response_model=TelemetryAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERRORS,
    tags=["telemetry"],
)
def ingest_telemetry(body: TelemetryEvent, c: ServiceContainer = Depends(get_container)) -> TelemetryAccepted:
    """Publish a ``BrakeTelemetryReceived`` event (202 Accepted: processed asynchronously in production)."""
    event = c.event_bus.publish({"event_type": TELEMETRY_EVENT, **body.model_dump()})
    return TelemetryAccepted(
        event_type=TELEMETRY_EVENT,
        vehicle_id=body.vehicle_id,
        delivered_to=event["delivered_to"],
        timestamp=event["timestamp"],
    )


@api_router.get("/events", tags=["monitoring"], responses=ERRORS)
def events(limit: int = Query(20, ge=1, le=200), c: ServiceContainer = Depends(get_container)) -> dict:
    """Most recent events on the bus."""
    return {"events": c.event_bus.recent(limit)}


@api_router.get("/audit", tags=["monitoring"], responses=ERRORS)
def audit(limit: int = Query(20, ge=1, le=200), c: ServiceContainer = Depends(get_container)) -> dict:
    """Most recent prediction audit records."""
    return {"predictions": c.audit_log.recent(limit)}


@api_router.get("/monitoring", tags=["monitoring"], responses=ERRORS)
def monitoring(c: ServiceContainer = Depends(get_container)) -> dict:
    """Operational metrics, model-quality metrics and data-drift status."""
    latencies = sorted(c.audit_log.latencies())
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    drift = c.drift_report()
    return {
        "model_version": c.predictor.version,
        "predictions_served": c.audit_log.total,
        "latency_ms": {
            "p50": round(median(latencies), 3) if latencies else 0.0,
            "p95": round(p95, 3),
            "target_p95": 500,
        },
        "feature_store_vehicles": len(c.feature_store),
        "model_quality": c.bundle.metrics if c.bundle else {},
        "training_data_quality": c.bundle.data_quality if c.bundle else {},
        "drift": (
            drift.to_dict()
            if drift
            else {
                "status": "insufficient_data",
                "samples": len(c.telemetry_window),
                "required_samples": c.settings.drift_min_samples,
            }
        ),
    }
