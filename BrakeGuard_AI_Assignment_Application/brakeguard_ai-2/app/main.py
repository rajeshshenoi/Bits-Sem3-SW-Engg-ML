from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from time import perf_counter
from .services.model_service import predict, METRICS, MODEL_VERSION
from .services.feature_service import get_features, update_features, FEATURE_STORE
from .services.event_bus import publish, subscribe, recent_events
from .services.audit_service import PREDICTION_LOG, record_prediction

app = FastAPI(title="BrakeGuard AI", version="1.0")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class Telemetry(BaseModel):
    vehicle_id: str
    brake_temperature: float = Field(..., ge=0)
    brake_pressure: float = Field(..., ge=0)
    pad_wear_pct: float = Field(..., ge=0, le=100)
    vibration_rms: float = Field(..., ge=0)
    fluid_age_days: float = Field(..., ge=0)
    odometer_km: float = Field(..., ge=0)


def handle_telemetry(event):
    update_features(event["vehicle_id"], event)

subscribe("BrakeTelemetryReceived", handle_telemetry)

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    # Starlette 1.x expects request/name/context as explicit arguments.
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "vehicles": sorted(FEATURE_STORE.keys()),
            "metrics": METRICS,
            "model_version": MODEL_VERSION,
            "predictions": list(reversed(PREDICTION_LOG[-6:])),
            "events": recent_events(),
        },
    )

@app.get("/api/v1/vehicles/{vehicle_id}/features")
def vehicle_features(vehicle_id: str):
    f = get_features(vehicle_id)
    if not f:
        raise HTTPException(404, "Vehicle not found in feature store")
    return {"vehicle_id": vehicle_id, "features": f}

@app.post("/api/v1/predictions")
def prediction(vehicle_id: str):
    features = get_features(vehicle_id)
    if not features:
        raise HTTPException(404, "Vehicle not found in feature store")
    start = perf_counter()
    probability, risk, contributions = predict(features)
    explanation = ", ".join([f"{k} is influential" for k, _ in contributions[:3]])
    latency = (perf_counter() - start) * 1000
    audit = record_prediction(vehicle_id, probability, risk, MODEL_VERSION, latency, explanation)
    return {
        "vehicle_id": vehicle_id,
        "fault_probability": round(probability, 4),
        "risk": risk,
        "explanation": explanation,
        "model_version": MODEL_VERSION,
        "latency_ms": round(latency, 2),
        "audit": audit,
    }

@app.post("/api/v1/telemetry")
def telemetry(data: Telemetry):
    event = {"event_type": "BrakeTelemetryReceived", **data.model_dump()}
    publish(event)
    return {"status": "accepted", "event": event, "subscribers": ["FeatureStoreConsumer"]}

@app.get("/api/v1/events")
def events():
    return {"events": recent_events(20)}

@app.get("/api/v1/audit")
def audit():
    return {"predictions": list(reversed(PREDICTION_LOG[-20:]))}

@app.get("/api/v1/monitoring")
def monitoring():
    avg_latency = round(sum(x["latency_ms"] for x in PREDICTION_LOG) / len(PREDICTION_LOG), 2) if PREDICTION_LOG else 0
    return {
        "availability": "99.5% target",
        "p95_latency_target_ms": 500,
        "observed_avg_latency_ms": avg_latency,
        "predictions_logged": len(PREDICTION_LOG),
        "feature_store_vehicles": len(FEATURE_STORE),
        "model_version": MODEL_VERSION,
    }

@app.get("/health")
def health():
    return {"status": "healthy", "services": {"api": "up", "feature_store": "up", "model_service": "up", "event_bus": "up"}}
