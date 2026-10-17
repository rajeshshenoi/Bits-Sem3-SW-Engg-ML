"""Feature-service / feature-store prototype."""
from typing import Dict

FEATURE_STORE: Dict[str, dict] = {
    "BG-10245": {
        "brake_temperature": 186.4, "brake_pressure": 72.1, "pad_wear_pct": 78.0,
        "vibration_rms": 3.2, "fluid_age_days": 410, "odometer_km": 84250
    },
    "BG-20481": {
        "brake_temperature": 121.5, "brake_pressure": 63.8, "pad_wear_pct": 31.0,
        "vibration_rms": 0.9, "fluid_age_days": 120, "odometer_km": 41200
    },
    "BG-30991": {
        "brake_temperature": 164.0, "brake_pressure": 68.5, "pad_wear_pct": 58.0,
        "vibration_rms": 2.0, "fluid_age_days": 260, "odometer_km": 97300
    },
}


def get_features(vehicle_id: str):
    return FEATURE_STORE.get(vehicle_id)


def update_features(vehicle_id: str, telemetry: dict):
    FEATURE_STORE[vehicle_id] = {k: telemetry[k] for k in telemetry if k != "vehicle_id" and k != "timestamp" and k != "event_type"}
