"""ASGI entry point: ``uvicorn app:app``."""

from brakeguard.api import create_app

app = create_app()
