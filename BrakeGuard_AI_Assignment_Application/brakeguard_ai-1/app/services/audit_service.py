"""Audit/monitoring service for prediction events."""
from datetime import datetime, timezone

PREDICTION_LOG: list[dict] = []


def record_prediction(vehicle_id, probability, risk, model_version, latency_ms, explanation):
    item = {
        "vehicle_id": vehicle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probability": round(probability, 4),
        "risk": risk,
        "model_version": model_version,
        "latency_ms": round(latency_ms, 2),
        "explanation": explanation,
    }
    PREDICTION_LOG.append(item)
    return item
