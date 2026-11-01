"""Single source of truth for the telemetry schema.

Both the data validator and the API request models are derived from
:data:`TELEMETRY_SCHEMA`, so the training data contract and the serving
contract can never silently diverge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Physical type and plausible range of a raw telemetry signal."""

    name: str
    unit: str
    min_value: float
    max_value: float
    description: str


TELEMETRY_SCHEMA: tuple[FeatureSpec, ...] = (
    FeatureSpec("brake_temperature", "degC", 0.0, 400.0, "Rotor temperature"),
    FeatureSpec("brake_pressure", "bar", 0.0, 200.0, "Hydraulic line pressure"),
    FeatureSpec("pad_wear_pct", "%", 0.0, 100.0, "Brake-pad wear"),
    FeatureSpec("vibration_rms", "g", 0.0, 20.0, "RMS vibration at the caliper"),
    FeatureSpec("fluid_age_days", "days", 0.0, 3650.0, "Days since fluid change"),
    FeatureSpec("odometer_km", "km", 0.0, 1_000_000.0, "Vehicle odometer"),
)

RAW_FEATURES: tuple[str, ...] = tuple(spec.name for spec in TELEMETRY_SCHEMA)
TARGET_COLUMN = "fault_within_30d"
ID_COLUMN = "vehicle_id"
SCHEMA_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in TELEMETRY_SCHEMA}
