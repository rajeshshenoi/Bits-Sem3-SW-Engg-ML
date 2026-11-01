// Builds reports/Group14.docx from the generated evidence.
// Usage (from brakeguard_ai-3/):  node scripts/build_report.js
// Requires the `docx` npm package. Re-run scripts.quality_report / pytest first so numbers are current.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, LevelFormat, Footer, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const rd = (p) => fs.readFileSync(path.join(ROOT, p), "utf8");
const quality = JSON.parse(rd("reports/metrics/quality_report.json"));
const mq = quality.model_quality.test;
const mqTrain = quality.model_quality.train;
const dqBad = quality.data_quality.corrupted_batch;
const dqClean = quality.data_quality.clean_batch;
const drift = quality.drift;

const PAGE_W = 9360; // 6.5in content width (A4 with 1in margins ~ 9026; we use 1in on Letter-ish width)
const ACCENT = "1F3A5F";
const SHADE = "E8EEF6";

// ---------- helpers ----------
const p = (text, opts = {}) =>
  new Paragraph({ spacing: { after: 120 }, ...opts, children: runs(text) });
function runs(text) {
  // **bold** and `code` inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun(text.slice(last, m.index)));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, color: "7A2E0E" }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun(text.slice(last)));
  return out;
}
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const bullet = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60 }, children: runs(t) });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: t, italics: true, size: 18, color: "555555" })] });

function code(text, label) {
  const lines = text.replace(/\t/g, "    ").split("\n");
  const paras = [];
  if (label) paras.push(new Paragraph({ spacing: { before: 60, after: 0 }, children: [new TextRun({ text: label, bold: true, size: 17, color: "555555" })] }));
  lines.forEach((line, i) =>
    paras.push(new Paragraph({
      spacing: { after: 0, before: 0, line: 240 },
      shading: { type: ShadingType.CLEAR, fill: "F4F5F7", color: "auto" },
      border: i === 0 ? { top: { style: BorderStyle.SINGLE, size: 4, color: "C9CED6", space: 2 } }
        : i === lines.length - 1 ? { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9CED6", space: 2 } } : undefined,
      children: [new TextRun({ text: line.length ? line : " ", font: "Consolas", size: 16 })],
    }))
  );
  paras.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return paras;
}

// Extract `start`..`end` (exclusive) lines matching markers from a source file.
function excerpt(file, startMarker, endMarker, maxLines = 60) {
  const lines = rd(file).split("\n");
  const s = lines.findIndex((l) => l.includes(startMarker));
  if (s < 0) throw new Error(`marker not found in ${file}: ${startMarker}`);
  let e = endMarker ? lines.findIndex((l, i) => i > s && l.includes(endMarker)) : -1;
  if (e < 0) e = Math.min(lines.length, s + maxLines);
  return lines.slice(s, Math.min(e, s + maxLines)).join("\n").replace(/\s+$/, "");
}

function image(rel, maxWIn = 6.3, maxHIn = 8.3) {
  const buf = fs.readFileSync(path.join(ROOT, rel));
  // PNG width/height from IHDR
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  let wPx = maxWIn * 96, hPx = (wPx * h) / w;
  if (hPx > maxHIn * 96) { hPx = maxHIn * 96; wPx = (hPx * w) / h; }
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(wPx), height: Math.round(hPx) } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "B8C2CF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: SHADE, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: String(t).split("\n").map((line) => new Paragraph({ spacing: { after: 0 },
      children: head ? [new TextRun({ text: line, bold: true, size: 19 })] : runs(line).map((r) => r) })),
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });
const pct = (x) => (100 * x).toFixed(2) + " %";

// ---------- content ----------
const groupTable = table(
  ["Sl. No", "BITS ID", "Name", "Contribution of team member (Qualitative)", "Percentage Contribution (out of 100)"],
  [
    ["1", "<BITS ID>", "<Name>", "Refactoring into the brakeguard package (data, features, models, pipeline); research-vs-production comparison (Tasks 1, 2)", "<%>"],
    ["2", "<BITS ID>", "<Name>", "FastAPI REST API: schemas, status codes, error envelope, API-key security, dashboard (Task 5)", "<%>"],
    ["3", "<BITS ID>", "<Name>", "Test strategy and pytest suites: unit, data-quality, ML training/inference, integration; coverage (Tasks 6, 7)", "<%>"],
    ["4", "<BITS ID>", "<Name>", "Logging and error handling, black/isort/flake8/pylint, quality metrics and drift, production testing and security, report (Tasks 3, 4, 8, 9)", "<%>"],
  ],
  [700, 1300, 1500, 4260, 1600]
);

const children = [];
const add = (...xs) => xs.flat().forEach((x) => children.push(x));

// Title page
add(
  new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "AIML ZG546 - Software Engineering for Machine Learning", size: 26, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Assignment II", bold: true, size: 44, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 }, children: [new TextRun({ text: "BrakeGuard AI - Implementation, Code Quality and Testing of an ML-enabled Predictive-Maintenance System", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "Group No: 14", bold: true, size: 26 })] }),
  groupTable,
  gap(),
  p("Code: `BrakeGuard_AI_Assignment_Application/brakeguard_ai-3` (builds on the Assignment I prototype in `brakeguard_ai-2`). Implementation notebook: `notebooks/Group14.ipynb`."),
  pageBreak(),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents")] }),
  ...[
    "0. System overview",
    "Objective 1 - Implementation and code sharing [5 marks]",
    "    Task 1 - OOP / functional design of the ML application",
    "    Task 2 - Research code vs production code",
    "    Task 3 - Error handling and logging",
    "    Task 4 - Code formatting and linting (before / after)",
    "    Task 5 - REST API (FastAPI)",
    "Objective 2 - Quality assurance [5 marks]",
    "    Task 6 - Types of tests (pytest)",
    "    Task 7 - Tests specific to ML components (training, inference)",
    "    Task 8 - Model-quality and data-quality metrics",
    "    Task 9 - Testing in production and security",
    "Appendix A - How to run;  Appendix B - Deliverables",
  ].map((t) => new Paragraph({ spacing: { after: 80 }, indent: { left: t.startsWith("    ") ? 540 : 0 },
    children: [new TextRun({ text: t.trim(), bold: !t.startsWith("    "), size: 22 })] })),
  pageBreak()
);

// 0. Overview
add(
  h1("0. System overview"),
  p("BrakeGuard AI predicts the probability that a vehicle's brake system will develop a fault within the next 30 days / 1,000 km, using six telemetry signals: rotor temperature, line pressure, pad wear, caliper vibration, brake-fluid age and odometer. A Random-Forest classifier produces a probability, which is mapped to a LOW / MEDIUM / HIGH risk band (thresholds 0.40 / 0.70) with an explanation of the main drivers. The ML component sits inside non-ML components: a REST API, a feature store, an event bus for telemetry, an audit log and drift monitoring."),
  p("Assignment I delivered a working prototype (`brakeguard_ai-2`) with module-level globals, no data validation, one happy-path test file, no logging, and a dashboard bug (the audit table read `fault_probability` while the log stored `probability`). Assignment II rebuilds it as the production-style package below. The model family and hyper-parameters are kept, so the comparison isolates software-engineering changes."),
  table(["Package / module", "Responsibility", "Key classes / functions"], [
    ["`brakeguard/config.py`", "Immutable, environment-overridable settings", "`Settings`, `RiskThresholds`, `ModelHyperParams`"],
    ["`brakeguard/schemas.py`", "Single source of truth for the telemetry contract", "`TELEMETRY_SCHEMA`, `FeatureSpec`"],
    ["`brakeguard/data/`", "Data ingestion; data-quality validation", "`TelemetryDataSource` (ABC), `SyntheticTelemetrySource`, `CSVTelemetrySource`, `DataValidator`, `DataQualityReport`"],
    ["`brakeguard/features/`", "Feature engineering; online feature store", "`FeatureEngineer` (sklearn transformer), `FeatureStore`"],
    ["`brakeguard/models/`", "Training; versioned persistence; inference", "`ModelTrainer`, `ModelRegistry`, `ModelBundle`, `BrakeFaultPredictor`, `Prediction`"],
    ["`brakeguard/evaluation/`", "Model-quality metrics", "`evaluate_classifier`, `expected_calibration_error`"],
    ["`brakeguard/monitoring/`", "Audit log, pub/sub, drift", "`AuditLog`, `EventBus`, `DriftDetector`, `population_stability_index`"],
    ["`brakeguard/pipeline.py`", "Orchestration with quality gates", "`TrainingPipeline`, `load_or_train`"],
    ["`brakeguard/api/`", "REST layer", "`create_app`, routes, schemas, error handlers, API-key security, `ServiceContainer`"],
    ["`scripts/`", "CLIs", "`train.py`, `quality_report.py`, `build_notebooks.py`"],
  ], [2300, 2900, 4160]),
  gap()
);

// Objective 1
add(
  pageBreak(),
  h1("Objective 1 - Implementation and code sharing"),
  h2("Task 1 - OOP / functional design of the ML application"),
  p("The code is organised around the ML lifecycle, with one class per responsibility. The design follows these principles:"),
  bullet("**Single responsibility:** ingestion, validation, feature engineering, training, persistence and inference are separate classes. Each can be tested on its own (Section 6) and replaced independently."),
  bullet("**Dependency inversion:** `TrainingPipeline` depends on the abstract `TelemetryDataSource`, so synthetic data, a CSV export or a database are interchangeable. The integration test trains from a CSV without code changes."),
  bullet("**Composition over globals:** the API builds a `ServiceContainer` (predictor, feature store, event bus, audit log, drift detector) per application instance through the `create_app` factory. Tests create isolated apps with no model or with an API key, without shared mutable state."),
  bullet("**Functional core:** metrics (`evaluate_classifier`, `expected_calibration_error`, `population_stability_index`) are pure functions of their inputs, and settings and value objects are frozen dataclasses."),
  bullet("**Training/serving consistency:** `FeatureEngineer` is an sklearn transformer placed inside the model `Pipeline`. The same fitted transformation, including imputation medians and the derived features `pressure_deviation`, `thermal_wear_index` and `fluid_age_years`, is serialised with the model and replayed at inference time."),
  p("The training workflow with its two quality gates (data quality before training, model quality before registration):"),
  ...code(excerpt("brakeguard/pipeline.py", "    def run(self, register", "    def _check_quality_gates"), "brakeguard/pipeline.py - TrainingPipeline.run"),
  ...code(excerpt("brakeguard/features/engineering.py", "    def transform(self", "    def get_feature_names_out"), "brakeguard/features/engineering.py - FeatureEngineer.transform"),

  h2("Task 2 - Research code vs production code"),
  p("Component compared: **model training and inference**. The research version is `notebooks/01_research_prototype.ipynb`, written as typical exploratory cells with the same logic as Assignment I's `model_service.py`. The production version is the `brakeguard` package plus the `scripts/train.py` CLI. The notebook is kept on purpose, and its outputs show two defects the production code removes:"),
  bullet("Its explanation heuristic (`importance x raw value`) always ranks `odometer_km` first (score 7,681 vs 39 for the next feature) because odometer values are in tens of thousands of km. Production uses `importance x |z-score|`, and the top drivers for BG-10245 become thermal_wear_index, pad_wear_pct and vibration_rms."),
  bullet("It scores physically impossible input (`pad_wear_pct = 450` gives p = 0.94). Production rejects this with HTTP 422."),
  ...code(
`# research notebook - globals, magic numbers, implicit feature order, prints
model = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=4, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("recall", recall_score(y_test, pred))
joblib.dump(model, "model_research.joblib")          # relative path, no metadata
vehicle = [186.4, 72.1, 78.0, 3.2, 410, 84250]       # which number is which?
prob = model.predict_proba([vehicle])[0, 1]
risk = "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.40 else "LOW"`, "Research (notebook cell)"),
  ...code(
`# production - typed, validated, logged, versioned, tested
settings = get_settings()                                  # env-overridable, frozen
bundle = TrainingPipeline(settings=settings).run()         # validate -> train -> gate -> register
predictor = BrakeFaultPredictor(bundle, settings.thresholds)
prediction = predictor.predict(feature_store.get("BG-10245"))   # dict keyed by schema names
prediction.risk, prediction.fault_probability, prediction.top_factors`, "Production (package API)"),
  table(["Concern", "Research notebook", "Production package"], [
    ["Structure", "Linear cells and global variables; results depend on execution order", "Classes with single responsibilities, wired by `TrainingPipeline`"],
    ["Configuration", "Magic numbers and hard-coded paths", "`Settings` dataclass + `BRAKEGUARD_*` environment variables"],
    ["Data contract", "Implicit NumPy column order", "`TELEMETRY_SCHEMA` shared by validator, model and API"],
    ["Data quality", "None", "`DataValidator` gate (schema, missing, range, labels, duplicates)"],
    ["Model acceptance", "Human reads printed metrics", "Automatic gate: recall >= 0.75, PR-AUC >= 0.70"],
    ["Persistence", "Bare model, relative path", "`ModelBundle`: version, metrics, timestamp, sklearn version, drift reference"],
    ["Errors / logs", "Tracebacks / print", "Typed exceptions -> HTTP codes; `logging` with levels"],
    ["Tests / style", "None", "96 pytest tests (97 % coverage); pylint 10.00/10, flake8 0"],
  ], [1800, 3400, 4160]),
  gap(),

  h2("Task 3 - Error handling and logging"),
  p("All modules log through the `brakeguard.*` logger hierarchy configured in `logging_config.py` (format `time | level | module | message`). Levels have a fixed meaning: **INFO** for normal lifecycle events, **WARNING** for degraded but recoverable situations, and **ERROR** for failed operations where the caller receives an error. Failures raise typed exceptions from `exceptions.py`, which `api/errors.py` maps to HTTP status codes, so the API never returns a raw stack trace."),
  table(["Critical function", "INFO", "WARNING", "ERROR / exception"], [
    ["`CSVTelemetrySource.load`", "rows loaded", "file has no rows", "file missing / unparsable -> `DataIngestionError`"],
    ["`DataValidator.validate_or_raise`", "validation passed", "duplicates; missing values below gate; issues found", "data set rejected -> `DataValidationError`"],
    ["`FeatureEngineer.transform`", "-", "missing values imputed with training medians", "missing / non-numeric columns -> `FeatureEngineeringError` (422)"],
    ["`ModelTrainer.train`", "rows, positive rate, hold-out metrics", "-", "single-class target / fit failure -> `ModelTrainingError`"],
    ["`TrainingPipeline` quality gate", "model passed gates", "-", "model rejected -> `ModelTrainingError`, nothing registered"],
    ["`ModelRegistry.load` / `load_or_train`", "model loaded / registered", "sklearn version mismatch; artefact missing -> retrain fallback", "corrupt artefact -> `ModelRegistryError`"],
    ["`BrakeFaultPredictor.predict*`", "records scored", "-", "no model -> `ModelNotReadyError` (503)"],
    ["`FeatureStore.get`", "vehicle inserted / updated", "unknown vehicle", "`VehicleNotFoundError` (404)"],
    ["`EventBus.publish`", "subscriber registered", "event with no successful subscriber", "failing subscriber logged with traceback; others still run"],
    ["`AuditLog.record` / `DriftDetector`", "drift summary", "HIGH-risk prediction; PSI >= 0.10 / 0.25", "-"],
    ["HTTP middleware / handlers", "method, path, status, latency per request", "422 / 404 responses", "unexpected exception -> generic 500 with request id"],
  ], [2300, 1700, 2560, 2800]),
  gap(),
  ...code(excerpt("brakeguard/models/registry.py", "    def load(self)", null, 25), "brakeguard/models/registry.py - error handling with meaningful levels"),
  image("reports/screenshots/10_logging_levels.png"),
  caption("Figure 1 - Real log output: INFO lifecycle, WARNING for HIGH risk / drift / rejected input, ERROR for a missing data file (CLI exits with code 1) and a corrupt model artefact (automatic retraining fallback)."),

  h2("Task 4 - Code formatting and linting"),
  p("Tools: **black** (formatting, line length 110), **isort** (import order, black profile), **flake8** (PEP 8 / pyflakes) and **pylint** (static analysis). Configuration is in `pyproject.toml` and `.flake8`. The same commands run as a build gate in the Dockerfile. Two \"before\" baselines were captured: the Assignment I code base, and the new package before black and isort were run and findings fixed."),
  table(["Check", "Assignment I code (before)", "New package before fixes", "Final package (after)"], [
    ["black", "6 of 8 files would be reformatted", "7 of 44 files would be reformatted", "all 46 files unchanged"],
    ["isort", "3 files incorrectly sorted", "clean", "clean"],
    ["flake8", "37 issues (E231 x15, E302 x13, E501 x7, E305 x2)", "12 issues (E501)", "0 issues"],
    ["pylint", "7.25 / 10 (missing docstrings, long lines, too many args, ...)", "9.83 / 10", "10.00 / 10"],
  ], [1300, 3000, 2400, 2660]),
  gap(),
  p("Fixes applied to the package after the baseline: automatic reformatting (black) and import sorting (isort); renaming `VehicleId` to `VEHICLE_ID_PATH` (constant naming); removing variable shadowing (`audit` -> `entry`); marking intentionally unused arguments; and documenting why sklearn's `fit()`-time attributes and the dataclass field counts are acceptable. Suppressions are local and commented, never global. Raw reports are in `reports/lint/`."),
  image("reports/screenshots/report_crops/05a_lint_before_a1_tools.png", 6.3, 7.6),
  caption("Figure 2a - BEFORE: black, isort and flake8 on the Assignment I code base."),
  image("reports/screenshots/report_crops/05b_lint_before_a1_pylint.png", 6.3, 7.6),
  caption("Figure 2b - BEFORE: pylint on the Assignment I code base (7.25/10)."),
  image("reports/screenshots/06_lint_before_assignment2_package.png", 6.3, 7.6),
  caption("Figure 3 - BEFORE: the new package prior to black/isort and lint fixes (9.83/10, 12 flake8 issues)."),
  image("reports/screenshots/07_lint_after_assignment2_package.png"),
  caption("Figure 4 - AFTER: black and isort clean, flake8 0 issues, pylint 10.00/10."),

  h2("Task 5 - REST API (FastAPI)"),
  p("The API is created by `brakeguard.api.create_app()` and served by `uvicorn app:app`. It follows these design practices:"),
  bullet("**Resource-oriented, versioned URLs** under `/api/v1`, with nouns and HTTP verbs (`POST /predictions`, `GET /vehicles/{id}/features`). Operational probes (`/health` liveness, `/ready` readiness) are kept outside the versioned prefix."),
  bullet("**Explicit request/response schemas** (Pydantic) published in OpenAPI/Swagger. Field ranges are derived from `TELEMETRY_SCHEMA`, unknown fields are forbidden, and batch size is limited to 1-500."),
  bullet("**Meaningful status codes:** 200 OK, 202 Accepted (telemetry is processed asynchronously in production), 401 Unauthorized, 404 Not Found, 422 Unprocessable Content, 503 Service Unavailable (no model). All are documented per endpoint."),
  bullet("**Uniform error envelope** `{\"error\": {\"code\", \"message\", \"request_id\", \"details\"}}` for every non-2xx response."),
  bullet("**Traceability:** every response carries `X-Request-ID` and `X-Process-Time-ms`, and every prediction gets a `prediction_id` stored in the audit log with the model version."),
  bullet("**Security:** optional `X-API-Key` authentication on `/api/v1/*` (Section 9)."),
  table(["Method", "Endpoint", "Purpose", "Codes"], [
    ["GET", "/health, /ready", "Liveness / readiness probes", "200 / 200, 503"],
    ["GET", "/api/v1/model", "Model version, training time, metrics, thresholds", "200, 401, 503"],
    ["POST", "/api/v1/predictions", "Score telemetry in the request body", "200, 401, 422, 503"],
    ["POST", "/api/v1/predictions/batch", "Score up to 500 records (vectorised)", "200, 401, 422, 503"],
    ["GET", "/api/v1/vehicles", "List vehicles in the feature store", "200, 401"],
    ["GET", "/api/v1/vehicles/{id}/features", "Latest features of a vehicle", "200, 401, 404, 422"],
    ["POST", "/api/v1/vehicles/{id}/predictions", "Score a vehicle from the feature store", "200, 401, 404, 503"],
    ["POST", "/api/v1/telemetry", "Publish a BrakeTelemetryReceived event", "202, 401, 422"],
    ["GET", "/api/v1/events, /api/v1/audit", "Recent events / audited predictions", "200, 401"],
    ["GET", "/api/v1/monitoring", "Latency p50/p95, model quality, data drift", "200, 401"],
  ], [900, 3100, 3700, 1660]),
  gap(),
  ...code(excerpt("brakeguard/api/routes.py", '@api_router.post("/predictions", response_model', "@api_router.post(", 12), "brakeguard/api/routes.py - prediction endpoint"),
  image("reports/screenshots/report_crops/03_swagger_endpoints.png", 6.3, 7.9),
  caption("Figure 5 - Swagger UI (/docs): endpoints grouped by tag; lock icons mark API-key-protected routes."),
  image("reports/screenshots/report_crops/04_swagger_predict_schema.png", 6.3, 7.9),
  caption("Figure 6 - Request schema with example, and documented 200/401/422/503 responses for POST /api/v1/predictions."),
  image("reports/screenshots/09_api_curl_demo.png"),
  caption("Figure 7 - curl session: 200 prediction with explanation, 422 for pad wear of 450 %, 404 for an unknown vehicle, 202 for a telemetry event."),
  image("reports/screenshots/01_dashboard_full.png", 6.3, 8.0),
  caption("Figure 8 - Dashboard served by the API: prediction with drivers, hold-out model-quality metrics, latency, event-driven telemetry, live drift (a simulated heat-wave fleet drifts on brake temperature and pad wear only) and the audit log.")
);

// Objective 2
const t = { unit: 26, dq: 19, ml: 24, integ: 27 };
add(
  pageBreak(),
  h1("Objective 2 - Quality assurance"),
  h2("Task 6 - Types of tests (pytest)"),
  p(`The suite has **${t.unit + t.dq + t.ml + t.integ} tests in four categories**. All pass in about 12 s, with **97 % line coverage** of the \`brakeguard\` package. Session-scoped fixtures in \`tests/conftest.py\` train the production model once. Each API test gets a fresh app instance, so tests never share state.`),
  table(["Type", "Location", "#", "What is verified"], [
    ["Unit tests", "tests/unit/", String(t.unit), "Risk-threshold boundaries; FeatureEngineer (derived values, median imputation + WARNING log, missing/non-numeric columns); FeatureStore copies / unknown ids; EventBus isolation of failing subscribers; bounded AuditLog; ECE / metric correctness on hand-computed cases; registry round-trip, missing and corrupt artefacts"],
    ["Data-validation tests", "tests/data_quality/", String(t.dq), "Schema (missing columns, target), completeness gate, physical-range violations, non-numeric types, invalid labels, duplicates, empty data, CSV ingestion errors, PSI behaviour, DriftDetector on stable vs shifted fleets"],
    ["ML tests", "tests/ml/", String(t.ml), "Training behaviour (8) and inference behaviour (16); see Task 7"],
    ["Integration tests", "tests/integration/", String(t.integ), "HTTP -> services -> model end to end: every endpoint and status code, input-validation and auth rules, event-driven feature-store update, drift monitoring, CSV -> train -> register -> load, bad data blocks registration, retrain fallback, CLI exit codes"],
  ], [1700, 1700, 500, 5460]),
  gap(),
  image("reports/screenshots/08_pytest_results.png", 6.3, 8.0),
  caption("Figure 9 - pytest run (excerpt; full verbose log in reports/tests/pytest_report.txt) with the coverage table."),

  h2("Task 7 - Tests specific to ML components"),
  h3("7a. Testing model training (tests/ml/test_training.py)"),
  table(["Test", "Rationale"], [
    ["`test_model_can_overfit_a_small_batch`", "A high-capacity forest must memorise 32 samples (100 % training accuracy). Failure means misaligned labels or dropped features."],
    ["`test_training_loss_decreases_as_trees_are_added`", `Random forests have no epochs, so the analogue is training log-loss as trees are added (warm_start). It falls from ${quality.training_loss_curve["1"]} (1 tree) to ${quality.training_loss_curve["160"]} (160 trees) and must never rise by more than 0.01 (Figure 10).`],
    ["`test_model_beats_trivial_baseline`", "PR-AUC must exceed the positive rate (a random classifier's PR-AUC) by 0.2; ROC-AUC > 0.75."],
    ["`test_model_meets_quality_targets`", "Recall >= 0.75, ECE <= 0.10, Brier <= 0.20."],
    ["`test_training_is_reproducible`", "Same data and seed give identical metrics."],
    ["`test_no_severe_overfitting`", "Train-test ROC-AUC gap < 0.15."],
    ["`test_single_class_target_is_rejected`", "Degenerate data raises `ModelTrainingError`."],
    ["`test_quality_gate_rejects_model_trained_on_noise`", "With shuffled labels the pipeline refuses to register the model and logs an ERROR."],
  ], [3600, 5760]),
  gap(),
  ...code(excerpt("tests/ml/test_training.py", "def test_model_can_overfit_a_small_batch", "def test_model_beats_trivial_baseline"), "tests/ml/test_training.py"),
  image("reports/metrics/loss_curve.png", 4.6),
  caption("Figure 10 - Training log-loss as trees are added (basis of the 'loss decreases' test)."),
  h3("7b. Testing model inference (tests/ml/test_inference.py)"),
  p("These tests follow the CheckList approach: shape and range checks, minimum-functionality cases, invariance tests (changes that must not affect the output) and directional-expectation tests (changes that must move the output in a known direction)."),
  table(["Category", "Tests"], [
    ["Output shape / range", "Batch output shape = (n,); all probabilities in [0, 1] and not NaN; risk band consistent with thresholds; explanation has top-k known features; deterministic; batch == single predictions"],
    ["Minimum functionality", "Obviously failing brakes (230 degC, 92 % wear, 5.5 g vibration) -> HIGH; new brakes -> LOW"],
    ["Invariance", "Key order of the input; extra metadata (vehicle_id, driver name); 0.1 % sensor noise changes probabilities by < 0.01 on average"],
    ["Directional", "Higher pad wear, vibration, temperature or fluid age raises mean fleet probability (> +0.02); line pressure 25 bar or 105 bar both score above nominal 65 bar"],
  ], [2200, 7160]),
  gap(),
  ...code(excerpt("tests/ml/test_inference.py", "# -- directional expectations", null, 26), "tests/ml/test_inference.py - directional expectation tests"),

  h2("Task 8 - Model-quality and data-quality metrics"),
  p("All values below were measured by `python -m scripts.quality_report` (output in `reports/metrics/quality_report.json`) on a 600-row stratified hold-out set of the 3,000-row synthetic fleet data. The same model metrics are stored in the model bundle and served at `/api/v1/model` and on the dashboard."),
  h3("8a. Model quality"),
  table(["Metric", "Why it matters for BrakeGuard", "Hold-out", "Train", "Target / gate"], [
    ["Recall", "Share of real brake faults caught; a missed fault is the costly error", mq.recall.toFixed(3), mqTrain.recall.toFixed(3), ">= 0.75 (registration gate)"],
    ["Precision", "Share of alerts that are real; controls workshop false alarms", mq.precision.toFixed(3), mqTrain.precision.toFixed(3), "monitored"],
    ["F1", "Balance of recall and precision", mq.f1.toFixed(3), mqTrain.f1.toFixed(3), "monitored"],
    ["PR-AUC", "Threshold-free ranking quality on the positive class", mq.pr_auc.toFixed(3), mqTrain.pr_auc.toFixed(3), ">= 0.70 (registration gate)"],
    ["ROC-AUC", "Threshold-free discrimination", mq.roc_auc.toFixed(3), mqTrain.roc_auc.toFixed(3), "> 0.75 (test)"],
    ["Brier score", "Calibration + sharpness of probabilities (lower is better)", mq.brier_score.toFixed(3), mqTrain.brier_score.toFixed(3), "<= 0.20 (test)"],
    ["ECE (10 bins)", "Does a 70 % shown to a fleet manager mean 70 %?", mq.expected_calibration_error.toFixed(3), mqTrain.expected_calibration_error.toFixed(3), "<= 0.10 (test)"],
  ], [1300, 3500, 1000, 900, 2660]),
  gap(),
  p(`Recall is ${mq.recall.toFixed(3)} at threshold 0.5. Across other synthetic seeds hold-out recall varied between about 0.70 and 0.82, which is why recall is a gate rather than a single trusted number, and why it is re-checked online (Section 9). Calibration is good (ECE ${mq.expected_calibration_error.toFixed(3)}), so the probability shown on the dashboard can be read at face value.`),
  image("reports/metrics/calibration.png", 4.6),
  caption("Figure 11 - Reliability diagram: predicted probabilities track observed fault rates."),
  image("reports/metrics/precision_recall.png", 4.6),
  caption("Figure 12 - Precision-recall curve with the operating point at threshold 0.5."),
  h3("8b. Data quality"),
  p("`DataValidator` measures data-quality metrics for every batch, and `DriftDetector` compares live telemetry with the training reference. They were run on a clean batch, a deliberately corrupted batch (sensor dropouts, a fraction-vs-percent unit bug and duplicated rows), and a stable and a shifted fleet:"),
  table(["Metric", "Definition", "Clean batch", "Corrupted batch", "Gate"], [
    ["Schema validity", "All required columns present and numeric", String(dqClean.schema_valid), String(dqBad.schema_valid), "must be true"],
    ["Missing-value ratio", "Share of null cells per column", "0.00 %", `${pct(dqBad.missing_ratio.brake_temperature)} (brake_temperature)`, "<= 5 % per column"],
    ["Out-of-range ratio", "Values outside the physical range in TELEMETRY_SCHEMA", "0.00 %", `${pct(dqBad.out_of_range_ratio.pad_wear_pct)} (pad_wear_pct)`, "<= 1 % per column"],
    ["Duplicate-row ratio", "Exact duplicate records", "0.00 %", pct(dqBad.duplicate_ratio), "warning"],
    ["Gate result", "", dqClean.passed ? "PASSED" : "FAILED", dqBad.passed ? "PASSED" : "REJECTED", ""],
  ], [1700, 2900, 1300, 2000, 1460]),
  gap(),
  table(["Drift metric", "Stable fleet (max over features)", "Shifted fleet", "Thresholds"], [
    ["PSI (10 reference-quantile bins)", Math.max(...Object.values(drift.stable_fleet.psi)).toFixed(3), `brake_temperature ${drift.shifted_fleet.psi.brake_temperature.toFixed(2)}, pad_wear_pct ${drift.shifted_fleet.psi.pad_wear_pct.toFixed(2)}; others <= ${Math.max(drift.shifted_fleet.psi.brake_pressure, drift.shifted_fleet.psi.vibration_rms, drift.shifted_fleet.psi.fluid_age_days, drift.shifted_fleet.psi.odometer_km).toFixed(3)}`, "warning 0.10, alert 0.25"],
    ["Kolmogorov-Smirnov p-value", `min ${Math.min(...Object.values(drift.stable_fleet.ks_pvalue)).toFixed(3)}`, `brake_temperature ${drift.shifted_fleet.ks_pvalue.brake_temperature < 1e-6 ? "< 1e-6" : drift.shifted_fleet.ks_pvalue.brake_temperature.toExponential(1)}`, "< 0.01 confirms shift"],
    ["Drifted features", drift.stable_fleet.drifted_features.length ? drift.stable_fleet.drifted_features.join(", ") : "none", drift.shifted_fleet.drifted_features.join(", "), ""],
  ], [2400, 2200, 3200, 1560]),
  gap(),
  p("While building the live drift monitor, PSI over 10 bins on a 30-40-sample window flagged every feature as drifting, because the estimate is dominated by noise. The minimum window was therefore made a setting (`BRAKEGUARD_DRIFT_MIN_SAMPLES`, default 200), and an integration test checks that an in-distribution fleet of 250 vehicles raises no drift alert."),
  image("reports/metrics/drift_psi.png", 5.4),
  caption("Figure 13 - PSI per feature: the shifted (heat-wave) fleet drifts on temperature and pad wear only."),

  h2("Task 9 - Testing in production and security"),
  h3("Testing / experimentation in production: shadow -> canary -> A/B"),
  p("Offline metrics come from synthetic data, and brake faults are only confirmed at the workshop weeks later. A new model version (the challenger) is therefore rolled out against the current champion in three stages:"),
  table(["Stage", "Traffic", "Exit criteria"], [
    ["1. Shadow (2-4 weeks)", "100 % of requests mirrored to the challenger; only the champion's answer is returned and the challenger's is logged", "No errors; p95 latency < 500 ms; >= 90 % agreement on HIGH/LOW; challenger recall >= champion on vehicles whose outcome is already known"],
    ["2. Canary (1-2 weeks)", "5 % -> 25 % -> 50 % of vehicles, sticky by vehicle_id hash", "Error rate and latency not worse; HIGH-alert rate within +/-20 % of champion; automatic rollback on breach"],
    ["3. A/B, then full rollout", "50/50, then 100 %", "After the 30-day label horizon: recall not significantly lower (one-sided test, alpha 0.05), false-alarm rate lower or equal; old version kept in the registry for one-click rollback"],
  ], [2000, 3500, 3860]),
  gap(),
  p("Shadow mode comes first because a missed brake fault is a safety incident, so a challenger must not influence a real maintenance decision before its live behaviour is known. The code base already has the needed pieces: `ModelRegistry` versions artefacts, `AuditLog` records `prediction_id` and `model_version` for joining with workshop outcomes, and `/api/v1/monitoring` reports the latency and drift guard-rail metrics that pause or roll back a rollout."),
  h3("Security consideration: input validation and model access control"),
  p("**Threat.** Crafted or faulty telemetry could suppress a real HIGH alert (an evasion attack), poison the feature store and drift statistics, or be used to query the model at high volume to reverse-engineer it. The Assignment I API and the research notebook accept any value; the notebook scores 450 % pad wear as a valid reading."),
  bullet("**Schema-driven input validation:** physical ranges per field, `extra=\"forbid\"`, a `vehicle_id` pattern and a batch cap of 500. Violations return 422 and are logged as WARNING. Covered by 6 integration tests."),
  bullet("**Model access control:** when `BRAKEGUARD_API_KEY` is set, `/api/v1/*` requires an `X-API-Key` header, compared in constant time with `hmac.compare_digest`. The key comes only from the environment. Probes stay public for orchestration."),
  bullet("**Safe failure:** generic 500 responses without stack traces; details are logged under the `request_id`."),
  bullet("**Bounded state** (audit, event and drift windows) against memory exhaustion, and drift alerts as a detection control for coordinated data injection."),
  bullet("**Hardened deployment:** multi-stage Docker build with lint and tests as a gate, non-root user, read-only filesystem."),
  p("Next steps for production: mTLS from vehicle gateways, per-client rate limiting at the gateway, and signed model artefacts. Unpickling an untrusted joblib file executes code, so only a file whose hash matches the registry entry should be loaded. Full write-up: `docs/production_testing_and_security.md`.")
);

// Appendix
add(
  pageBreak(),
  h1("Appendix A - How to run"),
  ...code(
`cd BrakeGuard_AI_Assignment_Application/brakeguard_ai-3
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements-dev.txt
python -m scripts.train --metrics-out reports/metrics/training_metrics.json
uvicorn app:app --reload                                  # dashboard http://127.0.0.1:8000, Swagger /docs
pytest -v --cov=brakeguard                                # 96 tests
black --check brakeguard tests scripts app.py && isort --check-only brakeguard tests scripts app.py
flake8 brakeguard tests scripts app.py && pylint brakeguard scripts app.py
python -m scripts.quality_report                          # metrics + figures
docker compose up --build                                 # lint + tests run inside the image build`),
  h1("Appendix B - Deliverables in the repository"),
  table(["Path", "Content"], [
    ["brakeguard/", "Production package (about 2,000 lines across 29 modules)"],
    ["tests/", "96 pytest tests in unit / data_quality / ml / integration"],
    ["notebooks/Group14.ipynb", "Executed implementation notebook with group details"],
    ["notebooks/01_research_prototype.ipynb", "Research-code version used for the Task 2 comparison"],
    ["docs/", "research_vs_production.md, production_testing_and_security.md"],
    ["reports/lint/", "Lint reports: before (Assignment I), before (new package), after"],
    ["reports/tests/", "pytest verbose log + coverage, summary"],
    ["reports/metrics/", "quality_report.json, training_metrics.json, figures"],
    ["reports/api/", "curl transcript, logging evidence"],
    ["reports/screenshots/", "Dashboard, Swagger, lint, tests, API, logging screenshots"],
  ], [3600, 5760])
);

const doc = new Document({
  creator: "Group 14",
  title: "AIML ZG546 Assignment II - BrakeGuard AI",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: ACCENT, font: "Calibri" }, paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: ACCENT, font: "Calibri" }, paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "333333", font: "Calibri" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } },
  ] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "Group 14 - AIML ZG546 Assignment II - BrakeGuard AI - page ", size: 16, color: "777777" }),
      new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "777777" }) ] })] }) },
    children,
  }],
});

const out = path.join(ROOT, "reports", "Group14.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out); });
