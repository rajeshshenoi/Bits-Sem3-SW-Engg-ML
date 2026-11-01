"""Bounded, thread-safe audit log of served predictions."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from ..logging_config import get_logger

logger = get_logger(__name__)


class AuditLog:
    """Keeps the most recent ``max_entries`` predictions in memory.

    Assignment I used an unbounded module-level list (a memory leak in a
    long-running service). A ``deque(maxlen=...)`` bounds memory; in
    production the same records would also be shipped to a log store.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = Lock()
        self._total = 0

    def record(  # pylint: disable=too-many-arguments
        self,
        *,
        vehicle_id: str,
        fault_probability: float,
        risk: str,
        model_version: str,
        latency_ms: float,
        explanation: str,
    ) -> dict:
        """Append a prediction record and return it."""
        entry = {
            "prediction_id": str(uuid4()),
            "vehicle_id": vehicle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fault_probability": round(fault_probability, 4),
            "risk": risk,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 3),
            "explanation": explanation,
        }
        with self._lock:
            self._entries.append(entry)
            self._total += 1
        if risk == "HIGH":
            logger.warning("HIGH brake-fault risk for %s (p=%.3f)", vehicle_id, fault_probability)
        return entry

    def recent(self, limit: int = 20) -> list[dict]:
        """Newest-first list of at most ``limit`` records."""
        with self._lock:
            items = list(self._entries)
        return list(reversed(items[-limit:]))

    def latencies(self) -> list[float]:
        """Latencies of the retained records."""
        with self._lock:
            return [e["latency_ms"] for e in self._entries]

    @property
    def total(self) -> int:
        """Number of predictions recorded since start-up."""
        return self._total
