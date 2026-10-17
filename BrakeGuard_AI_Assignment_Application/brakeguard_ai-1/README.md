# BrakeGuard AI — Assignment I Prototype

A runnable prototype for AIML ZG546 Assignment I demonstrating Objective 1 requirements and Objective 2 architecture.

## What it demonstrates
- ML: RandomForest brake-fault risk classifier
- Non-ML: FastAPI API layer, request validation, risk thresholds, audit logging, health monitoring
- Feature Store: in-memory prototype
- Explanation: feature-influence based explanation
- Event-driven pattern: publish/subscribe `BrakeTelemetryReceived`
- Microservices pattern: service boundaries are represented as separate modules (`model_service`, `feature_service`, `audit_service`) and can be containerized/deployed independently; this prototype keeps them in one process for easy demonstration.
- Docker/Kubernetes-ready packaging concept

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Run tests
```bash
pytest -q
```

## APIs
- `GET /health`
- `GET /api/v1/vehicles/{vehicle_id}/features`
- `POST /api/v1/predictions?vehicle_id=BG-10245`
- `POST /api/v1/telemetry`
- `GET /api/v1/events`
- `GET /api/v1/monitoring`

## Assignment evidence
Suggested screenshots:
1. Main dashboard showing ML and non-ML components.
2. High-risk prediction with probability, explanation and model version.
3. Event-driven telemetry publication and resulting feature-store update.
4. FastAPI Swagger UI (`/docs`) showing the APIs.
5. Terminal showing `pytest -q` passing.
