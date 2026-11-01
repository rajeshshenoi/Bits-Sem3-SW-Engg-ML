"""Online feature store (in-memory prototype, thread-safe).

In Assignment I the store was a module-level ``dict`` mutated from several
places. It is now an object with an explicit interface, which makes it
injectable into the API and trivially replaceable by Redis/Feast later.
"""

from __future__ import annotations

from copy import deepcopy
from threading import Lock

from ..exceptions import VehicleNotFoundError
from ..logging_config import get_logger
from ..schemas import RAW_FEATURES

logger = get_logger(__name__)

DEFAULT_FLEET: dict[str, dict[str, float]] = {
    "BG-10245": {
        "brake_temperature": 186.4,
        "brake_pressure": 72.1,
        "pad_wear_pct": 78.0,
        "vibration_rms": 3.2,
        "fluid_age_days": 410,
        "odometer_km": 84250,
    },
    "BG-20481": {
        "brake_temperature": 121.5,
        "brake_pressure": 63.8,
        "pad_wear_pct": 31.0,
        "vibration_rms": 0.9,
        "fluid_age_days": 120,
        "odometer_km": 41200,
    },
    "BG-30991": {
        "brake_temperature": 164.0,
        "brake_pressure": 68.5,
        "pad_wear_pct": 58.0,
        "vibration_rms": 2.0,
        "fluid_age_days": 260,
        "odometer_km": 97300,
    },
}


class FeatureStore:
    """Latest feature vector per vehicle."""

    def __init__(self, seed: dict[str, dict[str, float]] | None = None) -> None:
        self._data: dict[str, dict[str, float]] = deepcopy(seed if seed is not None else DEFAULT_FLEET)
        self._lock = Lock()

    def get(self, vehicle_id: str) -> dict[str, float]:
        """Return the features of ``vehicle_id`` or raise ``VehicleNotFoundError``."""
        with self._lock:
            features = self._data.get(vehicle_id)
        if features is None:
            logger.warning("Vehicle %s not found in feature store", vehicle_id)
            raise VehicleNotFoundError(f"Vehicle '{vehicle_id}' not found in feature store")
        return dict(features)

    def upsert(self, vehicle_id: str, telemetry: dict) -> dict[str, float]:
        """Insert/replace the feature vector of a vehicle from telemetry."""
        features = {name: float(telemetry[name]) for name in RAW_FEATURES}
        with self._lock:
            is_new = vehicle_id not in self._data
            self._data[vehicle_id] = features
        logger.info("Feature store %s vehicle %s", "inserted" if is_new else "updated", vehicle_id)
        return dict(features)

    def vehicle_ids(self) -> list[str]:
        """Sorted list of known vehicles."""
        with self._lock:
            return sorted(self._data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, vehicle_id: object) -> bool:
        with self._lock:
            return vehicle_id in self._data
