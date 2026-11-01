"""Service container: wires the domain services used by the API."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from ..config import Settings
from ..features import FeatureStore
from ..logging_config import get_logger
from ..models import BrakeFaultPredictor, ModelBundle
from ..monitoring import AuditLog, DriftDetector, DriftReport, EventBus
from ..schemas import RAW_FEATURES

logger = get_logger(__name__)

TELEMETRY_EVENT = "BrakeTelemetryReceived"


@dataclass
class ServiceContainer:  # pylint: disable=too-many-instance-attributes
    """All long-lived services of one API instance (no module globals)."""

    settings: Settings
    bundle: ModelBundle | None
    feature_store: FeatureStore = field(default_factory=FeatureStore)
    event_bus: EventBus = field(default_factory=EventBus)
    audit_log: AuditLog = field(default_factory=AuditLog)
    telemetry_window: deque = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self) -> None:
        self.predictor = BrakeFaultPredictor(self.bundle, self.settings.thresholds)
        self.drift_detector = (
            DriftDetector(self.bundle.reference_frame, self.settings.psi_warning, self.settings.psi_alert)
            if self.bundle is not None
            else None
        )
        self.event_bus.subscribe(TELEMETRY_EVENT, self._update_feature_store, "FeatureStoreConsumer")
        self.event_bus.subscribe(TELEMETRY_EVENT, self._record_for_drift, "DriftMonitorConsumer")

    def _update_feature_store(self, event: dict) -> None:
        self.feature_store.upsert(event["vehicle_id"], event)

    def _record_for_drift(self, event: dict) -> None:
        self.telemetry_window.append({name: float(event[name]) for name in RAW_FEATURES})

    def drift_report(self) -> DriftReport | None:
        """Drift of recent live telemetry vs training data (``None`` if too few samples)."""
        if self.drift_detector is None or len(self.telemetry_window) < self.settings.drift_min_samples:
            return None
        return self.drift_detector.evaluate(pd.DataFrame(list(self.telemetry_window)))
