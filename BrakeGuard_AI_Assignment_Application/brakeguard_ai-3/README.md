# BrakeGuard AI: Assignment II (implementation, code quality, QA)

AIML ZG546 Software Engineering for ML, Assignment II. This builds on the Assignment I prototype in
`../brakeguard_ai-2` and refactors it into a production-style, tested, lint-clean package with a
REST API.

**Group No:** 14. The member table is in `notebooks/Group14.ipynb` and in the report
(`reports/Group-14-Assignment_2_BrakeGuard_AI.docx` / `.pdf`).

## Layout

```
brakeguard/                 production package
  config.py                 Settings (env-overridable), risk thresholds, hyper-parameters
  logging_config.py         logging set-up (INFO / WARNING / ERROR conventions)
  exceptions.py             typed exception hierarchy
  schemas.py                TELEMETRY_SCHEMA: single source of truth for the data contract
  data/ingestion.py         TelemetryDataSource -> SyntheticTelemetrySource, CSVTelemetrySource
  data/validation.py        DataValidator -> DataQualityReport (schema, missing, range, dupes)
  features/engineering.py   FeatureEngineer (sklearn transformer; imputation + derived features)
  features/store.py         FeatureStore (thread-safe online store)
  models/trainer.py         ModelTrainer (pipeline, train/evaluate, loss curve)
  models/registry.py        ModelRegistry / ModelBundle (versioned persistence)
  models/predictor.py       BrakeFaultPredictor (inference + explanations)
  evaluation/metrics.py     recall, precision, F1, PR-AUC, ROC-AUC, Brier, ECE, log-loss
  monitoring/               AuditLog, EventBus, DriftDetector (PSI + KS)
  pipeline.py               TrainingPipeline (ingest -> validate -> train -> gate -> register)
  api/                      FastAPI app factory, routes, schemas, errors, API-key security, UI
app.py                      ASGI entry point (uvicorn app:app)
scripts/train.py            training CLI
scripts/quality_report.py   model/data-quality metrics + figures -> reports/metrics
scripts/build_notebooks.py  regenerates the notebooks
notebooks/01_research_prototype.ipynb   research-code version (for comparison)
notebooks/Group14.ipynb                implementation notebook
tests/unit | data_quality | ml | integration     96 pytest tests
docs/                       research-vs-production, production testing & security
reports/                    lint before/after, pytest+coverage, metrics, screenshots, final report
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
python -m scripts.train --metrics-out reports/metrics/training_metrics.json
```

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for Swagger. If no
model artefact exists, the API trains one on start-up and logs a WARNING.

To enable API-key authentication on `/api/v1/*`:

```bash
BRAKEGUARD_API_KEY=change-me uvicorn app:app
```

## Quality checks

```bash
pytest -v --cov=brakeguard
```

```bash
black --check brakeguard tests scripts app.py && isort --check-only brakeguard tests scripts app.py && flake8 brakeguard tests scripts app.py && pylint brakeguard scripts app.py
```

```bash
python -m scripts.quality_report
```

## REST API (v1)

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/health` | 200 liveness | |
| GET | `/ready` | 200 readiness | 503 model not loaded |
| GET | `/api/v1/model` | 200 version, metrics, thresholds | 401, 503 |
| POST | `/api/v1/predictions` | 200 prediction for telemetry in the body | 401, 422, 503 |
| POST | `/api/v1/predictions/batch` | 200, up to 500 items | 401, 422, 503 |
| GET | `/api/v1/vehicles` | 200 vehicle ids | 401 |
| GET | `/api/v1/vehicles/{id}/features` | 200 | 401, 404, 422 |
| POST | `/api/v1/vehicles/{id}/predictions` | 200 prediction from feature store | 401, 404, 503 |
| POST | `/api/v1/telemetry` | 202 event accepted | 401, 422 |
| GET | `/api/v1/events`, `/api/v1/audit` | 200 recent records | 401 |
| GET | `/api/v1/monitoring` | 200 latency, quality, drift | 401 |

Every error uses the same envelope:
`{"error": {"code", "message", "request_id", "details"}}`.

## Docker

```bash
docker compose up --build
```

The build stage runs black, isort, flake8 and pytest, then trains the model. The runtime image runs
as a non-root user with a read-only filesystem.
