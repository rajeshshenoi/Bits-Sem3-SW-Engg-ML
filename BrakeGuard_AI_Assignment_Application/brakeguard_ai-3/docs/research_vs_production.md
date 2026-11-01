# Research code vs production code (Task 2)

Component compared: **model training + inference**.

* Research version: `notebooks/01_research_prototype.ipynb` (the same logic as the Assignment I
  `model_service.py`, written as exploratory cells).
* Production version: the `brakeguard` package (`data/`, `features/`, `models/`, `pipeline.py`) and
  the CLI `scripts/train.py`.

| Concern | Research notebook | Production package |
|---|---|---|
| Structure | Linear cells sharing global variables; correctness depends on execution order | Single-responsibility classes: `SyntheticTelemetrySource` / `CSVTelemetrySource` → `DataValidator` → `FeatureEngineer` → `ModelTrainer` → `ModelRegistry` → `BrakeFaultPredictor`, wired by `TrainingPipeline` |
| Configuration | Magic numbers inline (`n=3000`, `0.70`, `random_state=42`, file names) | Frozen `Settings` dataclass, overridable via `BRAKEGUARD_*` environment variables |
| Data contract | Implicit column order in a NumPy array | `TELEMETRY_SCHEMA` is the single source of truth for validator, model and API |
| Data quality | None (450 % pad wear is scored) | `DataValidator` quality gate: schema, missing ratio, physical range, labels, duplicates |
| Feature engineering | Inline; would have to be copy-pasted into serving code | `FeatureEngineer` sklearn transformer, serialised inside the pipeline, so there is no training/serving skew |
| Model acceptance | Metrics printed, a human decides | Quality gate: recall ≥ 0.75 and PR-AUC ≥ 0.70, otherwise `ModelTrainingError` and nothing is registered |
| Persistence | `joblib.dump(model, "model_research.joblib")` (relative path, no metadata) | `ModelBundle` with version, metrics, timestamp, sklearn version, reference data for drift; corrupt or missing artefacts handled |
| Explanations | `importance × raw value`, dominated by `odometer_km` because of its units | `importance × |z-score|`, which is unit-independent |
| Error handling | Exceptions surface as tracebacks | Typed exception hierarchy mapped to HTTP status codes; CLI returns exit code 1 |
| Observability | `print` | `logging` with INFO / WARNING / ERROR, request IDs, latency headers |
| Tests | None | 96 pytest tests, 97 % line coverage |
| Code style | Not checked | black, isort, flake8 (0 issues), pylint 10.00/10 |

Notebooks remain the right tool for exploration. Once an idea works, it moves into the package and
gets tests, and the notebook becomes a client of that package, as `notebooks/Group14.ipynb` does.
