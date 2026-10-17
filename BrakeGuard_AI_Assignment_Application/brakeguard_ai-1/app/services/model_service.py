"""ML model service for BrakeGuard AI.
Prototype model: RandomForest trained on synthetic brake telemetry.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split

FEATURES = ["brake_temperature", "brake_pressure", "pad_wear_pct", "vibration_rms", "fluid_age_days", "odometer_km"]
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "brakeguard_model.joblib"


def _make_training_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(155, 35, n).clip(70, 280)
    pressure = rng.normal(65, 15, n).clip(20, 110)
    wear = rng.beta(2.0, 2.5, n) * 100
    vibration = rng.gamma(2.0, 0.8, n).clip(0.1, 8)
    fluid_age = rng.gamma(2.2, 120, n).clip(5, 900)
    odo = rng.uniform(5000, 180000, n)
    score = (
        0.022 * (temp - 120) +
        0.028 * np.abs(pressure - 65) +
        0.065 * (wear - 35) +
        0.55 * (vibration - 1.2) +
        0.0022 * (fluid_age - 180) +
        0.000008 * (odo - 60000)
    )
    prob = 1 / (1 + np.exp(-(score - 2.2)))
    y = (rng.random(n) < prob).astype(int)
    X = np.column_stack([temp, pressure, wear, vibration, fluid_age, odo])
    return X, y


def train_model():
    X, y = _make_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=4, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "recall": round(float(recall_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred)), 3),
        "f1": round(float(f1_score(y_test, pred)), 3),
        "pr_auc": round(float(average_precision_score(y_test, p)), 3),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics, "features": FEATURES, "version": "brakeguard-rf-v1.0"}, MODEL_PATH)
    return model, metrics


def load_model():
    if not MODEL_PATH.exists():
        return train_model()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["metrics"]

MODEL, METRICS = load_model()
MODEL_VERSION = "brakeguard-rf-v1.0"


def predict(features: dict):
    x = np.array([[features[f] for f in FEATURES]], dtype=float)
    p = float(MODEL.predict_proba(x)[0, 1])
    risk = "HIGH" if p >= 0.70 else "MEDIUM" if p >= 0.40 else "LOW"
    importances = MODEL.feature_importances_
    contributions = sorted(zip(FEATURES, importances * np.abs(x[0])), key=lambda z: z[1], reverse=True)
    return p, risk, contributions
