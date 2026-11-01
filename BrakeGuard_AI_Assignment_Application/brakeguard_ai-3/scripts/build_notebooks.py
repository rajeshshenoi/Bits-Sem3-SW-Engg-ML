"""Generate the two notebooks with nbformat (run, then execute with nbconvert).

* ``notebooks/01_research_prototype.ipynb`` - deliberately written in
  "research code" style (globals, magic numbers, prints, no tests).
* ``notebooks/Group14.ipynb`` - the implementation notebook required by the
  assignment; it drives the production package end to end.
"""

# Long lines below are notebook cell *content* (markdown tables, demo code), not program logic.
# pylint: disable=line-too-long

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

GROUP_TABLE = """| Sl. No | BITS ID | Name | Contribution of team member (Qualitative) | Percentage Contribution (out of 100) |
|---|---|---|---|---|
| 1 | 2025ae05802 | SAI KRISHNA VIJAYA RAM K | Refactoring into the `brakeguard` package (data, features, models), research-vs-production comparison | 100% |
| 2 | 2025ae05999 | S. S SRINIVAS RAO | FastAPI REST API design (schemas, status codes, error envelope, API-key security), dashboard | 100% |
| 3 | 2025ae05192 | ABHISHEK ARORA | Test strategy and pytest suites (unit, data-quality, ML training/inference, integration), coverage | 100% |
| 4 | 2025ae05200 | RAJESH. R. SHENOI | Logging & error handling, linting/formatting, quality metrics & drift, production-testing & security write-up, report | 100% |
"""


def md(text: str):
    """Markdown cell."""
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    """Code cell."""
    return nbf.v4.new_code_cell(text.strip())


def research_notebook():
    """Notebook written in deliberate research-code style."""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# BrakeGuard - research prototype (RESEARCH CODE)

This notebook is the *research* version of the brake-fault model, written the way exploratory
notebooks usually are. It is kept **on purpose** to contrast with the production package
`brakeguard/` (see the report, section 1.2). Typical research-code traits are marked `# (R#)`:

* (R1) global mutable state, cells must be run in order
* (R2) magic numbers and hard-coded paths
* (R3) `print` instead of logging, no error handling
* (R4) no input validation / schema
* (R5) training and inference logic copy-pasted, no tests
* (R6) a scale-dependent explanation heuristic that nobody reviewed
"""),
        code("""
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score
"""),
        code("""
# (R1)(R2) globals + magic numbers
n = 3000
rng = np.random.default_rng(42)
temp = rng.normal(155, 35, n).clip(70, 280)
pressure = rng.normal(65, 15, n).clip(20, 110)
wear = rng.beta(2.0, 2.5, n) * 100
vibration = rng.gamma(2.0, 0.8, n).clip(0.1, 8)
fluid_age = rng.gamma(2.2, 120, n).clip(5, 900)
odo = rng.uniform(5000, 180000, n)
score = 0.022*(temp-120) + 0.028*np.abs(pressure-65) + 0.065*(wear-35) + 0.55*(vibration-1.2) + 0.0022*(fluid_age-180) + 0.000008*(odo-60000)
y = (rng.random(n) < 1/(1+np.exp(-(score-2.2)))).astype(int)
X = np.column_stack([temp, pressure, wear, vibration, fluid_age, odo])
X.shape, y.mean()
"""),
        code("""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=4, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]
pred = (p >= 0.5).astype(int)
# (R3) prints, metrics computed by hand each time
print("recall", recall_score(y_test, pred))
print("precision", precision_score(y_test, pred))
print("f1", f1_score(y_test, pred))
print("pr_auc", average_precision_score(y_test, p))
"""),
        code("""
# (R2) hard-coded relative path - breaks when run from another directory
joblib.dump(model, "model_research.joblib")
"""),
        code("""
# (R4)(R5) inference: feature order implicit, no validation, thresholds copy-pasted
vehicle = [186.4, 72.1, 78.0, 3.2, 410, 84250]
prob = model.predict_proba([vehicle])[0, 1]
risk = "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.40 else "LOW"
print(prob, risk)
"""),
        code("""
# (R6) explanation = importance * |raw value| -> odometer (tens of thousands of km) always "wins"
names = ["brake_temperature", "brake_pressure", "pad_wear_pct", "vibration_rms", "fluid_age_days", "odometer_km"]
sorted(zip(names, model.feature_importances_ * np.abs(vehicle)), key=lambda t: -t[1])[:3]
"""),
        code("""
# (R4) nothing stops physically impossible input: pad wear of 450 % is happily scored
model.predict_proba([[186.4, 72.1, 450.0, 3.2, 410, 84250]])[0, 1]
"""),
        code("""
import os
os.remove("model_research.joblib")  # tidy up the side effect of the cell above
"""),
    ]
    return nb


def group_notebook():
    """Implementation notebook driving the production package."""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"""
# AIML ZG546 - Software Engineering for Machine Learning - Assignment II
## BrakeGuard AI: implementation, code quality and testing

**Group No:** 14

{GROUP_TABLE}

This notebook drives the **production package** `brakeguard/` (it contains no model logic of its
own). Every cell imports from the package, so what is shown here is exactly what the API serves.
"""),
        code("""
import json, logging, sys, subprocess
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from brakeguard.config import get_settings
from brakeguard.logging_config import configure_logging
settings = get_settings()
configure_logging("INFO")
print(settings.model_version, settings.thresholds)
"""),
        md("## 1. Data ingestion and data-quality validation (Task 1, Task 8b)"),
        code("""
from brakeguard.data import SyntheticTelemetrySource, DataValidator
frame = SyntheticTelemetrySource(n_samples=3000, seed=42).load()
report = DataValidator(max_missing_ratio=settings.max_missing_ratio).validate(frame)
frame.head()
"""),
        code("""
{k: v for k, v in report.to_dict().items() if k in ("n_rows", "schema_valid", "passed", "overall_missing_ratio", "overall_out_of_range_ratio", "duplicate_ratio")}
"""),
        md(
            "A corrupted batch (sensor drop-outs + a unit bug) is rejected by the same validator - note the WARNING/ERROR logs:"
        ),
        code("""
import numpy as np
from brakeguard.exceptions import DataValidationError
bad = frame.copy()
bad.loc[bad.index[:300], "brake_temperature"] = np.nan
bad.loc[bad.index[300:400], "pad_wear_pct"] *= 100
try:
    DataValidator().validate_or_raise(bad)
except DataValidationError as exc:
    print("Rejected:", exc.issues)
"""),
        md("## 2. Training pipeline with quality gates (Task 1, Task 8a)"),
        code("""
from dataclasses import replace
from brakeguard.pipeline import TrainingPipeline
nb_settings = replace(settings, model_path=ROOT / "artifacts" / "notebook_model.joblib")
bundle = TrainingPipeline(settings=nb_settings).run(register=True)
bundle.metrics
"""),
        md("## 3. Inference with explanations (Task 1)"),
        code("""
from brakeguard.models import BrakeFaultPredictor
from brakeguard.features import FeatureStore
predictor = BrakeFaultPredictor(bundle, settings.thresholds)
store = FeatureStore()
for vid in store.vehicle_ids():
    p = predictor.predict(store.get(vid))
    print(f"{vid}: {p.fault_probability:.3f} {p.risk:<6} | {p.explanation}")
"""),
        md(
            "Production input validation: the impossible input that the research notebook scored is rejected by the API schema."
        ),
        code("""
from pydantic import ValidationError
from brakeguard.api.schemas import TelemetryFeatures
try:
    TelemetryFeatures(brake_temperature=186.4, brake_pressure=72.1, pad_wear_pct=450, vibration_rms=3.2, fluid_age_days=410, odometer_km=84250)
except ValidationError as exc:
    print(exc.errors()[0]["loc"], exc.errors()[0]["msg"])
"""),
        md("## 4. REST API (Task 5) - exercised in-process with FastAPI's TestClient"),
        code("""
from fastapi.testclient import TestClient
from brakeguard.api import create_app
client = TestClient(create_app(settings=nb_settings, bundle=bundle))
r = client.post("/api/v1/predictions", json={"vehicle_id": "BG-10245", "telemetry": store.get("BG-10245")})
print(r.status_code); print(json.dumps(r.json(), indent=2))
"""),
        code("""
for method, path, body in [
    ("get", "/ready", None),
    ("post", "/api/v1/telemetry", {"vehicle_id": "BG-55555", **store.get("BG-10245")}),
    ("get", "/api/v1/vehicles/BG-99999/features", None),
    ("post", "/api/v1/predictions", {"telemetry": {**store.get("BG-10245"), "pad_wear_pct": 450}}),
]:
    resp = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    print(f"{method.upper():4} {path:42} -> {resp.status_code}  {str(resp.json())[:90]}")
"""),
        md("## 5. Model-quality and data-quality metrics, drift (Task 8)"),
        code("""
subprocess.run([sys.executable, "-m", "scripts.quality_report"], cwd=ROOT, check=True, capture_output=True)
q = json.loads((ROOT / "reports/metrics/quality_report.json").read_text())
print("Model quality:", {k: q["model_quality"]["test"][k] for k in ("recall", "precision", "f1", "pr_auc", "brier_score", "expected_calibration_error")})
print("Corrupted batch issues:", q["data_quality"]["corrupted_batch"]["issues"])
print("Shifted fleet drifted features:", q["drift"]["shifted_fleet"]["drifted_features"])
"""),
        code("""
from IPython.display import Image, display
for name in ("calibration.png", "precision_recall.png", "loss_curve.png", "drift_psi.png"):
    display(Image(filename=str(ROOT / "reports/metrics" / name), width=520))
"""),
        md("## 6. Test suite (Tasks 6 and 7)"),
        code("""
out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
print(out.stdout[-600:])
"""),
        code("""
(ROOT / "artifacts" / "notebook_model.joblib").unlink(missing_ok=True)
"""),
    ]
    return nb


def main() -> None:
    """Write both notebooks."""
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(research_notebook(), NB_DIR / "01_research_prototype.ipynb")
    nbf.write(group_notebook(), NB_DIR / "Group14.ipynb")


if __name__ == "__main__":
    main()
