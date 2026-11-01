"""In-process publish/subscribe event bus (Kafka stand-in)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Callable

from ..logging_config import get_logger

logger = get_logger(__name__)

Handler = Callable[[dict], None]


class EventBus:
    """Synchronous event bus with error isolation between subscribers."""

    def __init__(self, max_events: int = 500) -> None:
        self._subscribers: dict[str, list[tuple[str, Handler]]] = {}
        self._log: deque[dict] = deque(maxlen=max_events)
        self._lock = Lock()

    def subscribe(self, event_type: str, handler: Handler, name: str | None = None) -> None:
        """Register ``handler`` for ``event_type``."""
        label = name or getattr(handler, "__name__", "handler")
        self._subscribers.setdefault(event_type, []).append((label, handler))
        logger.info("Subscriber '%s' registered for %s", label, event_type)

    def subscribers(self, event_type: str) -> list[str]:
        """Names of subscribers of ``event_type``."""
        return [label for label, _ in self._subscribers.get(event_type, [])]

    def publish(self, event: dict) -> dict:
        """Deliver an event; a failing subscriber never breaks the others."""
        if "event_type" not in event:
            raise ValueError("event must contain 'event_type'")
        event = dict(event)
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        delivered, failed = [], []
        for label, handler in self._subscribers.get(event["event_type"], []):
            try:
                handler(event)
                delivered.append(label)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Subscriber '%s' failed on %s", label, event["event_type"])
                failed.append(label)
        event["delivered_to"] = delivered
        event["failed_subscribers"] = failed
        with self._lock:
            self._log.append(event)
        if not delivered:
            logger.warning("Event %s had no successful subscriber", event["event_type"])
        return event

    def recent(self, limit: int = 20) -> list[dict]:
        """Newest-first list of recent events."""
        with self._lock:
            items = list(self._log)
        return list(reversed(items[-limit:]))
