"""Lightweight event-driven broker prototype used to demonstrate publish/subscribe."""
from datetime import datetime, timezone
from typing import Callable

SUBSCRIBERS: dict[str, list[Callable]] = {}
EVENT_LOG: list[dict] = []


def subscribe(event_type: str, handler: Callable):
    SUBSCRIBERS.setdefault(event_type, []).append(handler)


def publish(event: dict):
    event = dict(event)
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    EVENT_LOG.append(event)
    for handler in SUBSCRIBERS.get(event.get("event_type"), []):
        handler(event)


def recent_events(limit=8):
    return list(reversed(EVENT_LOG[-limit:]))
